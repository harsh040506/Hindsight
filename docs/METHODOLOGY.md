# Methodology

Everything Hindsight claims rests on four steps. This document explains each,
states what it cannot do, and names the assumption it would fail under.

---

## 1. Cohort-relative scoring

### The problem

View counts are not comparable across a catalog. Three effects dominate any
per-video difference:

- **Age.** A video published eleven months ago has had eleven months to
  accumulate views.
- **Channel size.** The channel this tool was built against went from 80 to
  531 subscribers over the period. Later uploads started with a larger
  audience before the content mattered at all.
- **Platform conditions.** Algorithmic treatment of a channel drifts.

Sort a year of uploads by views and you have mostly measured publish date.

### Rejected alternatives

| Approach | Why not |
|---|---|
| Views per day since publish | Punishes evergreen content, rewards spikes. A video with 1000 views in week one and none since beats one gaining 50/week forever. |
| Fit a growth curve, take residuals | Findings inherit every assumption in the curve. Choosing exponential vs linear changes which videos look good. |
| Only analyse the last 30 days | Discards 90% of the evidence, and 30 days on most channels is not enough videos to clear any significance threshold. |
| Normalise by subscriber count at publish | Subscriber history is not exposed by the API, and subscribers are a poor proxy for reach on a shorts channel anyway. |

### The approach

For the video at position *i* in publication order, its **cohort** is the
`COHORT_HALF_WIDTH` videos published immediately before it and the same number
immediately after. Its score is its percentile rank within that cohort.

```
score(v) = ( |{c ∈ cohort : views(c) < views(v)}|
           + 0.5 · |{c ∈ cohort : views(c) = views(v)}| ) / |cohort| × 100
```

Default half-width is 25, so each video is judged against its 50 nearest
neighbours in time.

Ties are counted at half weight. This keeps the scale symmetric: a video in a
cohort where everything scored identically lands at 50, not 0. On catalogs
with many low-view videos sharing exact counts, counting only "strictly below"
would push a whole cluster to 0 and invent differences between videos that
performed the same.

### Why it works

Anything that moves slowly relative to the cohort window is shared by every
member of the cohort and cancels out of a percentile. Channel growth,
seasonality, algorithmic drift, and view accumulation with age are all
slow in exactly this sense. What survives is variation *within* a cohort —
which is the per-video choices.

This is verified rather than asserted. `tests/test_cohort.py::TestGrowthCancellation`
builds a catalog where baseline views grow 10× across the period and every
third video carries a known +40% effect, then asserts:

- raw view counts make the later era look 2×+ better (the failure mode),
- cohort percentiles show no difference between eras (growth removed),
- cohort percentiles still recover the +40% effect (signal preserved).

### Edge handling

Videos near either end of the catalog cannot have a symmetric cohort. The
oldest video's neighbours are all newer, and on a growing channel newer means
more views — so its percentile is biased downward by its position in the list
rather than by anything about the video. Those videos are scored but flagged
and **excluded from the analysis** by default.

Videos younger than `MIN_AGE_DAYS` (default 14) are also excluded: they have
not finished accumulating views and would drag down every bucket they land in.

### What this cannot do

Cohort scoring **cannot detect anything that changed slowly and monotonically
across the whole catalog**, because that is precisely what it removes. If you
gradually improved your thumbnails over a year, Hindsight will not see it.

This is the correct trade. A slow channel-wide drift is not a lever you can
pull on tomorrow's upload.

---

## 2. Permutation testing

### Why not a t-test

YouTube view counts are violently right-skewed. On the reference channel the
mean is 337 and the median 140 — the mean sits at the 68th percentile. Any
method assuming normally distributed outcomes reads that skew as signal.

Working on cohort percentiles already bounds the outcome, but the difference
between two bucket means still has no clean parametric distribution. So
Hindsight does not assume one.

### The test

For a bucket *A* and the rest of the catalog *B*:

1. Compute the observed difference in means, `|mean(A) − mean(B)|`.
2. Pool all values, randomly re-split into groups of the original sizes,
   recompute the difference. Repeat 10,000 times.
3. The p-value is the fraction of random splits producing a gap at least as
   large as the observed one.

The null hypothesis is *"the labels are exchangeable"* — which is exactly the
question being asked: could this gap have arisen from arbitrarily assigning
these videos to these buckets?

### The +1 correction

```
p = (hits + 1) / (iters + 1)
```

Without the added 1, a test that never sees a more extreme permutation reports
p = 0, claiming more certainty than 10,000 resamples can support (Phipson &
Smyth, 2010). The smallest p-value Hindsight can return is 1/10001.

### Comparison choice

Each bucket is tested **one-vs-rest** — against every other video the feature
applies to, not against one hand-picked rival. Choosing a comparison bucket
would let the analysis flatter itself by picking the weakest opponent.

---

## 3. False discovery rate control

Hindsight tests roughly forty feature buckets per run. At α = 0.05 you expect
**two false findings per analysis from noise alone**. Reporting those to a
creator who then rewrites their content strategy is the single most harmful
thing this tool could do.

Every p-value in a run is corrected together using **Benjamini-Hochberg**:

1. Sort the *m* p-values ascending.
2. Scale each: `q(i) = p(i) · m / i`.
3. Enforce monotonicity with a cumulative minimum from the largest rank down.

A q-value of 0.05 means *"if I act on every finding at or below this
threshold, about 5% of what I act on will be flukes"*. That is the question
that matters when choosing which lever to pull.

**Why not Bonferroni.** Bonferroni controls the probability of *any* false
positive, which is far stricter than needed here and would suppress nearly
every real effect at m = 40. A creator does not need certainty that zero
findings are wrong; they need to know roughly what fraction are.

`TestBenjaminiHochberg::test_correction_suppresses_lone_marginal_result`
pins the behaviour: one p = 0.04 among 39 nulls must not survive.

---

## 4. Era-split replication

A p-value cannot answer the question a sceptical creator should ask: *did this
show up because of a habit I had during one period?*

Suppose 01:00 uploads underperform. If the channel posted at 01:00 only during
one three-month stretch when it was also small or between formats, then
"01:00" is standing in for "that era" and rescheduling achieves nothing.
Cohort scoring suppresses most of this — a video is only ever compared to its
immediate neighbours — but it cannot suppress a habit that persisted across
whole cohorts.

So every surviving finding is **re-tested independently on the first and
second half of the catalog**. An effect appearing in two disjoint samples
separated by months is a property of the choice; one appearing in a single
half is a property of the period.

Findings that fail to replicate are **kept and labelled**, not dropped. The
correct response to them is to run the experiment, not to act.

On the reference channel both timing findings replicate: +12.9 early / +14.0
late for the 12:00 effect.

---

## 5. Power analysis and experiment sizing

Sample size per arm comes from the standard two-sample formula:

```
n = 2 · (z(1−α/2) + z(power))² · σ² / Δ²
```

with σ estimated from the observed spread of cohort percentiles. Because
percentiles are bounded, σ is stable (≈ 29, close to the 28.9 of a uniform
distribution) even on channels whose raw views span three orders of magnitude.

The inverse normal CDF is Acklam's rational approximation, accurate to
~1.15e-9 — implemented in-repo rather than adding scipy for two z-scores.

### Sizing to a budget, not an effect

Fixing the minimum detectable effect answers *"how long until I can detect a
shift this small?"* and routinely returns over a year. On the reference
channel, detecting 5 percentile points needs 524 videos per arm — 272 days at
its cadence.

So Hindsight inverts the question by default: given your **actual upload
rate** and a time budget, what is the smallest effect you could detect by
then? At 90 days and 3.85 uploads/day that is 173 videos per arm and an 8.7
point resolution — and the plan says so explicitly, alongside a table of what
other budgets would buy.

This number is usually the most sobering output the tool produces. Saying it
out loud is the point: it stops people concluding anything from the six-video
"test" they were about to run.

---

## Assumptions, stated plainly

Hindsight's findings are valid **if**:

1. Videos published close together in time faced comparable conditions. Fails
   if a single video went viral and dragged its neighbours, or during a
   platform-wide anomaly affecting part of a cohort.
2. Uploads are frequent enough that a 50-video cohort spans a short period.
   On a channel posting weekly, a cohort spans a year and the method degrades
   toward raw comparison. Hindsight is built for high-cadence channels.
3. The lever is genuinely independent of the confounders being frozen. If you
   only ever wrote long titles for one topic, "long title" and "that topic"
   cannot be separated observationally — which is what `design` is for.

None of these make a finding causal. **The catalog was not randomised.** Every
result is an association strong enough to be worth testing properly, which is
why the pipeline ends in an experiment rather than a conclusion.

---

## References

- Phipson, B. & Smyth, G. K. (2010). *Permutation p-values should never be
  zero.* Statistical Applications in Genetics and Molecular Biology, 9(1).
- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the false discovery rate.*
  Journal of the Royal Statistical Society B, 57(1), 289–300.
- Acklam, P. J. (2003). *An algorithm for computing the inverse normal
  cumulative distribution function.*
