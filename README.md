# Hindsight

**Your channel already ran the experiment. Nobody read the results.**

If you publish on a schedule, you have hundreds of videos sitting in your
uploads tab — every one of them a test of a title, a length, a posting time.
That is a dataset. Almost nobody mines it, because comparing videos honestly
is harder than it looks: a video from last July has had a year to accumulate
views on a smaller channel, so sorting by view count mostly tells you *when
you published*.

Hindsight reads your back catalog, works out which of your choices actually
moved the numbers, and then designs the experiment that settles the next
question.

```bash
pip install -e .
hindsight demo --tz 5.5   # no credentials, no API key, no setup
```

---

## What it found on a real channel

A real shorts channel: 949 videos, ~317,000 lifetime views, thirteen months of
uploads. 938 are public, and 845 of those clear the analysis filters.

| | |
|---|---|
| **Publishing at 01:00 costs 13.9 percentile points** | median **70 views** · n=83 · q=0.018 |
| **Publishing at 12:00 gains 13.5 percentile points** | median **323 views** · n=58 · q=0.041 |

That is a **4.6× difference in median views** driven by nothing but the hour
of day — on a channel that had been publishing for thirteen months without
noticing. Both results survive correction for multiple testing, and both
replicate independently on the older and newer halves of the catalog, months
apart.

Reproduce all of it, with no credentials:

```bash
hindsight demo --tz 5.5
```

That catalog ships with the package. **The `--tz` matters**: hours are reported
in the timezone you pass, and the channel's own is UTC+5:30. Plain
`hindsight demo` runs in UTC and reports the same two findings shifted — 19:00
and 07:00 — because the bucket edges land differently.

It also found that **every one of the 845 analysed videos carries the same five
tags**. A full year of uploads, and that lever had never once been varied.
Hindsight reports that as an opportunity rather than a null result, and sizes
the experiment to test it.

---

## Why this is hard to do correctly

Three problems sit between "I have 949 videos" and "I know what works", and
most analytics tooling walks into all three.

**1. You cannot compare raw view counts.** Older videos have had longer to
accumulate views, and they were published to a smaller channel. Views-per-day
overcorrects and punishes evergreen content; fitting a growth curve makes your
findings inherit the curve's assumptions.

Hindsight scores each video as a **percentile against the 50 videos published
nearest to it in time**. Those neighbours share the same channel size, the
same season, the same algorithmic weather and the same age, so all of it
cancels. What survives is the part that varied *within* the cohort — the
per-video choices.

**2. Test forty things and two will look significant by luck.** Hindsight
tests around forty feature buckets per run. At p<0.05 you would expect two
false findings from noise alone, and a creator who rewrites their strategy
around those is worse off than before they started.

Every p-value in a run is corrected together using **Benjamini-Hochberg**, and
significance is judged on the corrected q-value. Findings are also **re-tested
independently on the older and newer halves** of the catalog: an effect that
appears in one era is a habit, not a lever, and gets labelled as such.

**3. "No finding" means three completely different things.** A lever can be
measured and show nothing; it can have never been varied; or it can have
varied so lopsidedly that nothing is measurable. Collapsing these into "no
effect" is how a channel concludes tags don't matter after 949 uploads that
all used identical tags.

Hindsight separates them explicitly, and treats the untested ones as the most
valuable output it produces — they are the only levers guaranteed to still
have headroom.

---

## The loop

A dashboard tells you what happened. Hindsight tells you what to change, then
holds itself accountable for whether the change worked.

```
ingest  →  analyze  →  design  →  (publish)  →  verdict
   ↑                                                │
   └────────────────────────────────────────────────┘
```

`hindsight design` picks the lever with the most unrealised value and emits a
real experimental plan:

- **Arms** to compare — and for a never-varied lever it constructs the
  challenger, mining candidate tags from your own best-performing titles.
- **Sample size** from a power calculation, sized to read out within a time
  budget at *your* current upload rate rather than to a fixed effect that
  might take a year.
- **Variables to freeze**, pinned to their best known values, so a confirmed
  effect cannot confound the new test.
- **Interleaved assignment**, because running arm A for a week and arm B the
  next tests one week against another.

Then `hindsight verdict` reads the channel back and calls it — classifying
each published video by the value it actually carries, so the readout stays
honest even if your pipeline drifted from the plan.

An inconclusive result reports the smallest effect the test could have
resolved. "No effect" and "no effect this test could see" are different
claims, and only one of them is usually true.

---

## Running it

### Zero setup — bundled real data

```bash
hindsight demo --tz 5.5
```

Runs the entire pipeline against a genuine 949-video catalog shipped with the
package, and writes a standalone HTML report. Publish times, view counts,
durations and tags are **real and unmodified**; video ids, channel identity
and title text are synthetic so the dataset identifies nobody. Everything
downstream of ingestion is the same code that runs on live data.

A pre-rendered copy of that report is checked in at
[examples/hindsight-demo.html](examples/hindsight-demo.html) if you would
rather just look at the output.

### Any public channel — API key only

No OAuth, no consent screen:

```bash
export YOUTUBE_API_KEY=...        # console.cloud.google.com → Credentials
hindsight ingest --channel @somehandle
hindsight report
```

### Your own channel — OAuth

```bash
hindsight auth login              # or reuse an existing token
hindsight ingest --token path/to/token.json
hindsight analyze --tz 5.5        # publish-hour advice in your local time
hindsight report
hindsight design
```

Full setup notes, including the optional YouTube Analytics upgrade for
retention and CTR data, are in [docs/SETUP.md](docs/SETUP.md).

---

## Safety

**Hindsight cannot modify your channel.** It requests only `youtube.readonly`,
calls only read endpoints, and has no upload, edit, delete or publish path
anywhere in the codebase. Pointing it at a production channel is safe by
construction, not by discipline.

Credentials are never written to the repo; `.gitignore` covers token files,
API keys and the local catalog.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | The statistics: cohort scoring, permutation testing, FDR control, power analysis — and the limits of each |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map, data model, and why each design decision went the way it did |
| [docs/SETUP.md](docs/SETUP.md) | Credentials for all three modes, quota costs, troubleshooting |
| [docs/CLI.md](docs/CLI.md) | Every command, flag and output format |

---

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest                      # 175 tests
```

The test suite is mostly property-based rather than golden-value, because the
methods are stochastic. The one that matters most is
`TestGrowthCancellation` in `tests/test_cohort.py`: it builds a synthetic
catalog where views grow 10× over the period *and* a feature has a known
effect, then asserts that raw view counts give the wrong answer while cohort
percentiles recover the real one. That is Hindsight's central claim, written
as a falsifiable test.

Statistics are implemented in-repo rather than imported — permutation testing,
bootstrap intervals, Benjamini-Hochberg and the power calculation are all in
`src/hindsight/stats.py`. numpy is used to make the resampling fast, not to
supply the methods.

---

## Honest limits

- **These are associations, not proofs.** Your catalog was not randomised, so
  every finding is a well-supported hypothesis about a lever. That is exactly
  why the tool ends by designing an experiment instead of declaring victory.
- **Cohort scoring cannot see slow channel-wide drift**, because removing it
  is the entire mechanism. That is the right trade: a year-long trend is not
  something you can act on in tomorrow's upload.
- **Retention and click-through are not used.** They need the YouTube
  Analytics API, which requires both an extra OAuth scope and the API enabled
  on your Cloud project. Hindsight detects whether that path is available and
  works fully without it.
- **Small channels will find nothing, correctly.** Under a few hundred videos
  most buckets will not clear the size floor. The tool will say so rather than
  inventing findings.

---

## License

MIT.
