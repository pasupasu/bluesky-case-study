"""Convert data/user_posts.tar.gz into a folder of Parquet files without extracting it.

We read the archive as a stream, glue many users' files together into a batch, parse each
batch with polars and write it out as one Parquet part file.

Usage:
    uv run python -m bluesky_case_study.posts            # full run
    uv run python -m bluesky_case_study.posts --limit 5000 --out data/user_posts_test

The run can be resumed: part files that already exist are skipped. The archive
still has to be decompressed from the start (gzip can't seek), but skipped
batches aren't parsed or written, so catching up is quick.
"""

import argparse
import io
import json
import os
import tarfile
import time
from pathlib import Path

import polars as pl

from bluesky_case_study.paths import DATA_DIR

ARCHIVE = DATA_DIR / "user_posts.tar.gz"
OUT_DIR = DATA_DIR / "user_posts"

POST_SCHEMA = {
    "post_id": pl.Int64,
    "user_id": pl.Int64,
    "instance": pl.String,
    "date": pl.Int64,  # YYYYMMDDHHMM as a number, converted to a datetime below
    "text": pl.String,
    "langs": pl.List(pl.String),
    "like_count": pl.Int64,
    "reply_count": pl.Int64,
    "repost_count": pl.Int64,
    "reply_to": pl.Int64,
    "replied_author": pl.Int64,
    "thread_root": pl.Int64,
    "thread_root_author": pl.Int64,
    "repost_from": pl.Int64,
    "reposted_author": pl.Int64,
    "quotes": pl.Int64,
    "quoted_author": pl.Int64,
    "labels": pl.List(pl.String),
    "sent_label": pl.Int64,
    "sent_score": pl.Float64,
}


def _drop_bad_lines(
    chunks: list[tuple[str, bytes]], rejects_path: Path
) -> tuple[bytes, int]:
    """Keep only lines that parse as JSON objects; log the rest to rejects_path.

    Only used when polars fails on a whole batch, so the slow Python check runs
    on a handful of batches rather than all 125 GB.
    """
    good: list[bytes] = []
    n_bad = 0
    with open(rejects_path, "a", encoding="utf-8") as rejects:
        for name, data in chunks:
            for line_no, line in enumerate(data.split(b"\n"), start=1):
                if not line.strip():
                    continue
                try:
                    ok = isinstance(json.loads(line), dict)
                except ValueError:  # covers bad JSON and bad UTF-8
                    ok = False
                if ok:
                    good.append(line)
                else:
                    n_bad += 1
                    rejects.write(
                        json.dumps(
                            {
                                "file": name,
                                "line": line_no,
                                "raw": line[:2000].decode("utf-8", "replace"),
                            }
                        )
                        + "\n"
                    )
    return b"\n".join(good) + b"\n", n_bad


def _parse_batch(raw: bytes) -> pl.DataFrame:
    # The ndjson reader only accepts 64-bit numbers, so shrink small columns after.
    df = pl.read_ndjson(io.BytesIO(raw), schema=POST_SCHEMA)
    return df.with_columns(
        # strict=False so one malformed date becomes null instead of killing a long run
        pl.col("date").cast(pl.String).str.to_datetime("%Y%m%d%H%M", strict=False),
        pl.col("sent_label").cast(pl.Int8),
        pl.col("sent_score").cast(pl.Float32),
    )


def _write_part(df: pl.DataFrame, path: Path) -> None:
    # Write to a temp name first so an interrupted run never leaves a half-written
    # part file that a resumed run would mistake for a finished one.
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, path)


def convert(
    archive: Path = ARCHIVE,
    out_dir: Path = OUT_DIR,
    batch_bytes: int = 256 * 1024**2,
    limit: int | None = None,
) -> None:
    """Stream the archive into out_dir/part-NNNNN.parquet files.

    batch_bytes is how much raw JSON goes into each part file. Peak memory is a
    few times this, so lower it if RAM is tight.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.tmp"):
        stale.unlink()

    # Not *.parquet, so load_posts() won't pick it up.
    rejects_path = out_dir / "rejected_lines.jsonl"

    total_compressed = archive.stat().st_size
    start = time.monotonic()
    chunks: list[tuple[str, bytes]] = []  # (file name in archive, contents)
    chunk_size = 0
    part = 0
    n_files = 0
    n_rows = 0
    n_rejected = 0
    skipped = 0

    def flush(raw_file) -> None:
        nonlocal chunks, chunk_size, part, n_rows, n_rejected, skipped
        path = out_dir / f"part-{part:05d}.parquet"
        if path.exists():
            skipped += 1
        else:
            try:
                df = _parse_batch(b"".join(data for _, data in chunks))
            # polars raises ComputeError, or a Rust PanicException, which is a
            # BaseException rather than an Exception.
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                raw, n_bad = _drop_bad_lines(chunks, rejects_path)
                n_rejected += n_bad
                print(
                    f"part {part:5d}: dropped {n_bad} malformed line(s), "
                    f"logged to {rejects_path.name}",
                    flush=True,
                )
                df = _parse_batch(raw)
            _write_part(df, path)
            n_rows += df.height
        part += 1
        chunks, chunk_size = [], 0

        done = raw_file.tell() / total_compressed
        elapsed = time.monotonic() - start
        eta = elapsed / done - elapsed if done > 0 else 0
        print(
            f"part {part:5d} | {done:6.1%} | {n_files:,} users | {n_rows:,} posts "
            f"| {elapsed / 60:5.1f} min elapsed, ~{eta / 60:5.1f} min left",
            flush=True,
        )

    with open(archive, "rb") as raw_file:
        # "r|gz" reads the archive strictly front to back, with no seeking.
        with tarfile.open(fileobj=raw_file, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".jsonl"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                if data and not data.endswith(b"\n"):
                    data += b"\n"
                chunks.append((member.name, data))
                chunk_size += len(data)
                n_files += 1

                if chunk_size >= batch_bytes:
                    flush(raw_file)
                if limit is not None and n_files >= limit:
                    break

            if chunks:
                flush(raw_file)

    print(
        f"Done: {part} part files in {out_dir} "
        f"({skipped} already existed and were skipped). "
        f"{n_rejected} malformed line(s) dropped this run."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--batch-mb", type=int, default=256, help="raw JSON per part file (MB)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many user files"
    )
    args = parser.parse_args()
    convert(args.archive, args.out, args.batch_mb * 1024**2, args.limit)


if __name__ == "__main__":
    main()
