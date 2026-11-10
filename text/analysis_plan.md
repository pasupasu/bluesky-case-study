# Analysis plan: content-and-stance pairs

Written on 2026-11-08, **before** labelling the corpus with Haiku. The stance labels and the test score are fixed. What this plan fixes is how they're used. Anything decided after the corpus labels are in should be reported as exploratory.

This file is deliberately separate from `codebook.md`. The codebook is part of the model's prompt, so editing it would change the frozen prompt version.

## Research question

Are posts with similar content *and* stance received differently within a community depending on whether their author is in-group or out-group?

## Pairs (content)

- Built by `pairs.py`: `all-mpnet-base-v2` embeddings, FAISS HNSW, 20 nearest neighbours, cosine similarity **≥ 0.80**, unordered pairs.
- Only pairs between different authors. Posts with no words left after removing links and mentions are dropped.
- Flags kept for robustness checks rather than as filters: `short` (fewer than 4 words), `near_duplicate` (≥ 0.98), `linked` (one post replies to or quotes the other), `days_apart`.

## Stance (per post)

- Labels come from `claude-haiku-5-5` at low effort, using the frozen codebook (prompt version `216304cab33e`, see `FROZEN_VERSION` in `stance.py`).
- Measured accuracy:

| | Accuracy | Kappa | Pro-Palestinian / pro-Israeli / other |
|---|---|---|---|
| Dev (200 posts, tuned on) | 84% | 0.79 | 90% |
| **Test (198 posts, held out)** | **65%** (95% CI 59–72%) | 0.54 | 76% |

- Test is the number to report; dev is an upper bound.
- The side-level split was chosen after seeing the test results. That is stated here and should be stated in the write-up.

## Stance match (per pair) — fixed rule

- `side` maps `pro_palestinian` → PAL, `pro_israeli` → ISR, and every other label → other.
- **A pair is a stance match (`same_side`) when both posts are PAL or both are ISR.**
- Pairs where either side is other, and PAL–ISR pairs, are not matches. They can serve as comparison groups.

## Analyses

1. **Primary: PAL–PAL pairs.** On test, a post Haiku labels `pro_palestinian` is truly `pro_israeli` 1% of the time, so a PAL–PAL pair contains a truly opposite-side post roughly 3% of the time. About a quarter of these posts are really "other" (`unclear`, `anti_war_humanitarian`…). That weakens the stance match but doesn't reverse it.
2. **Exploratory: ISR–ISR pairs.** On test, 4 of the 16 posts labelled `pro_israeli` were truly `pro_palestinian`, so a quarter may be on the opposite side. Before reporting ISR–ISR results, hand-label about 50 randomly drawn ISR–ISR corpus pairs ("are both posts pro-Israeli?") to measure how many truly are.
3. **Sensitivity checks**, each reported next to the primary result:
   - only pairs with both labels at confidence ≥ 2 (`uncertain == False`);
   - seven-label match (`same_stance`) instead of `same_side`;
   - dropping `short`, `near_duplicate` and `linked` pairs;
   - a stricter similarity threshold (≥ 0.85);
   - capping or down-weighting hub posts (posts in many pairs), so that a few slogans don't dominate.
4. **Do labelling errors differ by group?** Haiku never sees the author, so errors should not depend on in-group vs out-group, and should make any difference smaller rather than larger. Check this: join dev and test authors to their Leiden communities, and compare Haiku's error rate across communities and between authors in and outside the community being analysed.

## Reception (fixed 2026-11-08, implemented in `reception.py`)

No classifier for replies or quotes. Reception is measured from what the data records directly:

- **Communities.** Leiden on the **full active interaction graph**: every user with at least one reply, repost or quote edge (weight > 0), about 1.66M users. The graph is undirected, weights are summed, the objective is modularity, and the seed is 42, the same settings as `notebooks/exploration.ipynb`. Saved to `data/communities.parquet`. The notebook's top-100k sample (`data/communities_top100k.parquet`) is kept for comparison only.
- **In-group / out-group.** For community C, a post's author is in-group if they are a member of C, and out-group otherwise: either in another community or with no interaction edges at all. `out_assigned` records which. A contrast is one (pair, C) row where exactly one of the two authors is in C. Pairs whose authors share a community give no contrast.
- **Outcomes**, counted only from members of C, with self-engagement dropped:
  - **positive:** reposts;
  - **volume:** replies and quotes. Their direction is unknown, so they are reported as volume, not as negative;
  - **negative tone:** replies with the dataset's `sent_label == 0`. This is a weak proxy, because tone isn't disagreement;
  - **likes:** community-blind (likes have no liker ids), per follower overall, as a secondary measure.
- **Exposure.** The author's followers who are members of C (from `followers.csv`, where src follows dst). Outcomes are compared as rates per in-community follower, because out-group authors usually have far fewer.
- **Controls.** Post date (counts are cumulative at collection time), `days_apart`, and the pair flags listed above.

### Model (fixed 2026-11-08, `reception model`)

**Unit.** A contrast $r$ is one content pair seen from one community $C$. It has an in-group post $i$ (author in $C$) and an out-group post $o$ (author not in $C$).

**Notation.**

| Symbol | Meaning |
|---|---|
| $y_{rp}$ | engagement with post $p \in \{i, o\}$ from members of $C$ (reposts, replies, quotes or negative-tone replies, one outcome at a time) |
| $\text{in}_p$ | 1 for the in-group post, 0 for the out-group post |
| $F^{C}_p$ | the author's followers who are members of $C$ (exposure within the community) |
| $F^{\text{all}}_p$ | the author's followers overall |
| $a_p$ | post age: time from the post's date to the end of collection, in 30-day units |
| $\Delta_r x = x_i - x_o$ | in-group minus out-group difference within contrast $r$ |

**Model: Poisson with a fixed effect for each contrast.**

$$
\log \mathbb{E}[y_{rp}] = \alpha_r + \beta\,\text{in}_p + \gamma \log\!\left(1 + F^{C}_p\right) + \delta \log\!\left(1 + F^{\text{all}}_p\right) + \lambda\, a_p
$$

- $\alpha_r$ absorbs everything the two posts share: content, topic, stance, and when the pair was found. It is never estimated.
- $\beta$ is the in-group effect. $\gamma$, $\delta$ and $\lambda$ are estimated from the data.
- Exposure enters as a covariate, not a fixed offset, because members of $C$ also engage with authors they don't follow. A fixed offset would force $\gamma = 1$.

**What is actually fitted.** Let $n_r = y_{ri} + y_{ro}$ be the contrast's total engagement from $C$. Conditioning on $n_r$ removes $\alpha_r$. What remains is a binomial model of the in-group post's share (the standard result for fixed-effect Poisson):

$$
y_{ri} \mid n_r \sim \operatorname{Binomial}(n_r,\ \pi_r)
$$

$$
\operatorname{logit} \pi_r = \beta + \gamma\, \Delta_r \log\!\left(1 + F^{C}\right) + \delta\, \Delta_r \log\!\left(1 + F^{\text{all}}\right) + \lambda\, \Delta_r a
$$

In words: only how engagement is *split* between the two posts matters, not how much the pair got in total. Contrasts with $n_r = 0$ carry no information and drop out.

**The quantity of interest** is the rate ratio

$$
\text{RR} = e^{\beta}
$$

the in-group post's expected engagement from $C$ divided by the out-group post's, when both authors have the same followers in $C$, the same followers overall, and their posts are the same age. Equivalently, the in-group post's expected share of the pair's engagement at equal exposure is $\pi = \text{RR} / (1 + \text{RR})$:

| $\text{RR} = e^{\beta}$ | In-group share $\pi$ | Reading |
|---|---|---|
| 0.5 | 33% | out-group post gets twice the engagement |
| 1 | 50% | **no in-group difference** |
| 1.5 | 60% | in-group post gets 50% more |
| 2 | 67% | in-group post gets twice the engagement |

**Reading the other coefficients.** Each is a log rate ratio per unit difference:
- $e^{\gamma}$: how much more engagement an author gets from $C$ when they have $e \approx 2.7$ times as many followers in $C$ ($\Delta \log(1 + F^C) = 1$). For example, $\gamma = 0.5$ means $e^{0.5} \approx 1.65$ times as much.
- $e^{\delta}$: the same for followers overall.
- $e^{\lambda}$: the change per 30 days of extra age (counts are cumulative, so older posts have had longer to collect engagement).

**Standard errors.** Clustered two ways, by in-group author and by out-group author (authors appear in many contrasts). There are too few communities with real weight to cluster by community, so communities are reported separately instead (below).

**Outcomes.**
- **Primary:** reposts.
- **Secondary:** replies, quotes, and replies with negative tone. Replies and quotes measure response volume; they don't say whether the response was approving.

**Sample.**
- **Primary:** PAL–PAL contrasts (`side_pair == "PAL-PAL"`) from communities with at least 100 contrasts.
- **Per community:** $e^{\beta}$ for each community with at least 1,000 contrasts.

**Robustness** (each reported next to the primary result):
1. **Exposure-matched:** only contrasts where the two authors' followers in $C$ differ by less than a factor of $e \approx 2.7$:

   $$\left|\Delta_r \log\!\left(1 + F^{C}\right)\right| \le 1$$

   This relies far less on how the model handles exposure. Without it, the gap between in-group and out-group authors (median 106 vs 15 followers in C, across all contrasts) has to be bridged entirely by the model's assumed form.
2. **Exposure modelled flexibly:** with $x_r = \Delta_r \log(1 + F^C)$, replace $\gamma\, x_r$ by a piecewise-linear spline with knots $\kappa_1, \kappa_2, \kappa_3$ at the quartiles of $x_r$:

   $$s(x) = \theta_0\, x + \sum_{k=1}^{3} \theta_k \left[ (x - \kappa_k)_+ - (-\kappa_k)_+ \right], \qquad (u)_+ = \max(u, 0)$$

   Every term is zero at $x = 0$, so $s(0) = 0$ and $e^{\beta}$ is still the in-group effect at equal exposure. (Dummy bins were dropped before any real-data run: on simulated data, their intercept picked up the exposure gap and reported 3.4 where the truth was 1.0.)
3. Both stance labels at confidence ≥ 2.
4. Dropping `short`, `near_duplicate` and `linked` pairs.
5. ISR–ISR contrasts, as exploratory (see the stance section).
6. **Opposite-side pairs (PAL–ISR) as a comparison:** the in-group effect when the content is similar but the stance differs. If $e^{\beta}$ is similar for same-side and opposite-side pairs, the effect is about who the author is, not what they said.

**Reading the result.** Report $e^{\beta}$ with a 95% confidence interval, $e^{\hat\beta \pm 1.96\,\widehat{\text{se}}(\hat\beta)}$. An interval that includes 1 means no detectable in-group difference. The headline uses the primary outcome and sample only. Everything else is secondary or robustness and is reported as such, without picking the best-looking one.

## Change log

| Date | Change |
|---|---|
| 2026-11-08 | First version, written after the test score and before the corpus run. |
| 2026-11-08 | Reception definitions fixed (no reply/quote classifier); `reception.py` added. |
| 2026-11-08 | Model fixed: conditional (fixed-effect) Poisson per contrast, exposure as a covariate, two-way clustered standard errors, robustness list. Written after a pipeline check that printed descriptive in/out means across all pairs, with no stance and no model. That check showed raw counts and per-follower rates pointing in opposite directions, which is why exposure is modelled and an exposure-matched check is required. |
| 2026-11-08 | Communities: switched from the notebook's top-100k sample to the full active graph (about 1.66M users), before any stance-filtered or modelled results. |
| 2026-11-09 | Model section reformatted: equations in LaTeX, a notation table, and worked readings of $e^{\beta}$ and the other coefficients. No change to the specification. Age is stated in 30-day units, as `fit_contrasts` has always used (this rescales $\lambda$ only). Corrected a descriptive figure in robustness check 1: median followers in $C$ are 106 (in-group) vs 15 (out-group) across all contrasts; the earlier text had "8 vs 113". |
| 2026-11-09 | ISR–ISR hand check (`isr_check.py`), set up after the model results were seen: 50 pairs drawn uniformly from all ISR–ISR corpus pairs (seed `isr-pairs-v1`); each post coded ISR / PAL / other, blind to author and community. Reported: share of pairs with both posts truly ISR (Wilson 95% CI), share with a PAL post, and, descriptively, the share weighted by each pair's ISR–ISR contrasts. 18 of the 50 pairs enter the model's contrasts. |
