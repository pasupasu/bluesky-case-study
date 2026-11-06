from pathlib import Path

# src/bluesky_case_study/paths.py -> go up 3 levels to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"