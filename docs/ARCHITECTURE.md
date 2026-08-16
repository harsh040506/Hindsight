# Architecture

## Shape

Hindsight is a pipeline of pure transformations wrapped in a thin CLI. Only
one module talks to the network, only one touches disk, and everything in
between is deterministic given its inputs.

```
                    ┌──────────┐
   YouTube API ────▶│  ingest  │────▶ SQLite catalog
                    └──────────┘            │
                                            ▼
   bundled dataset ─────────────────▶ ┌──────────┐
        (demo)                        │  cohort  │  percentile scoring
                                      └────┬─────┘
                                           ▼
                                      ┌──────────┐
                                      │ features │  choices → buckets
                                      └────┬─────┘
                                           ▼
                                      ┌──────────┐
                                      │  stats   │  permutation, FDR, power
                                      └────┬─────┘
                                           ▼
                                      ┌──────────┐
                                      │ analyze  │  classify + rank
                                      └────┬─────┘
                                     ┌─────┴─────┐
                                     ▼           ▼
                               ┌──────────┐ ┌──────────┐
                               │  report  │ │experiment│
                               └──────────┘ └──────────┘
                                  HTML        design/verdict
```

## Modules

| Module | Responsibility | Talks to |
|---|---|---|
| `config.py` | Tunables, paths, credential discovery | env |
| `auth.py` | OAuth and API-key session construction | Google |
| `ingest.py` | Catalog fetch with retry/backoff | Google, `db` |
| `db.py` | SQLite schema and queries | disk |
| `demo.py` | Bundled dataset + anonymisation | disk |
| `cohort.py` | Cohort-relative percentile scoring | — |
| `features.py` | Video record → controllable choices | — |
| `stats.py` | Permutation, bootstrap, BH, power | — |
| `analyze.py` | Orchestration and lever classification | above |
| `experiment.py` | Design and readout | `analyze`, `stats` |
| `report.py` | HTML and JSON output | `analyze` |
| `cli.py` | Commands and terminal output | all |

`cohort`, `features` and `stats` have no I/O and no dependency on each other.
They are the parts most worth trusting, so they are the parts easiest to test
in isolation.

---

## Key decisions

### Stat snapshots are append-only

Every ingest writes a new row per video into `stat_snapshots` rather than
overwriting a `views` column.

One ingest gives a photograph. Several give a time series — which is what lets
`verdict` say "this arm gained views faster" rather than merely "this arm has
more views today". It also makes the tool robust to a video being deleted or
made private: the history survives.

Cost is a few hundred KB per ingest. Video *metadata* is upserted instead,
because titles can be edited after publish and the current value is the one
that matters for attribution.

### The database is a cache, not a source of truth

Deleting `catalog.db` loses nothing `hindsight ingest` cannot rebuild. This is
why the schema check refuses to migrate: rebuilding is cheap and correct,
while a migration bug would silently corrupt the evidence behind a finding.

### Features are categorical, always

A 33-second video is in bucket `33s`; a 412-character description is in
`351-500 chars`. Numeric features are bucketed rather than correlated.

Two reasons. It keeps one code path through the analysis. More importantly it
keeps findings in the same shape as the decision — nobody sets "description
length to 0.34 standard deviations above mean", they pick a length.

### Bucketing adapts to the channel

`adaptive_buckets` inspects the distribution before choosing a strategy. If a
few discrete values cover ≥80% of the catalog — which is what an automated
pipeline with fixed render settings produces — it buckets on those exact
values. Otherwise it falls back to quartiles labelled with their real bounds.

A tool shipping hardcoded thresholds ("short = under 60s") gives a shorts
channel one bucket and zero findings.

### Every feature must be a lever

A feature is only included if a creator could decide differently next time.
This excludes plenty of things that correlate with views — video id, day of
the month, subscriber count at publish. A finding you cannot act on is a
distraction.

Each feature carries a `lever` string phrased as the action, not the
statistic: *"Schedule the upload for a different hour"*, not `publish_hour`.

### Features can supersede each other

Some levers cannot move independently. Changing which tags you use necessarily
changes how many. `Feature.supersedes` records this, and the experiment ranker
drops any candidate a higher-ranked one subsumes — otherwise a channel that
never varied its tags gets told to run two separate experiments that answer
the same question and confound each other.

### Three outcomes, not two

`analyze._classify` distinguishes:

- **tested** — varied enough to measure; the result is trustworthy either way
- **untested** — every video made the same choice; no variation, so no finding
- **underpowered** — varied, but too lopsidedly (837 vs 8) to compare

This distinction is the reason the tool exists. Collapsing it into "no
finding" is how a channel concludes tags don't matter after 949 identically
tagged uploads.

### Analysis is seeded

`RANDOM_SEED` fixes permutation and bootstrap draws, so the same catalog
always produces the same numbers. Reproducibility matters when someone is
about to make content decisions from the output.

---

## Safety

**Hindsight cannot write to YouTube.** This is structural, not procedural:

- `config.YOUTUBE_SCOPES` contains only `youtube.readonly`.
- `ingest.py` calls exactly three endpoints — `channels.list`,
  `playlistItems.list`, `videos.list`. All read.
- No module imports `MediaFileUpload` or any insert/update/delete method.
- API-key mode has no write capability at all.

If you point it at a token that happens to carry upload scope (because another
tool minted it), Hindsight still only calls read endpoints.

`.gitignore` covers `*_token.json`, `client_secrets.json`, `.env` and the
local catalog. Refreshed OAuth tokens are written back to their original file
and nowhere else.

---

## Failure handling

`ingest._execute` wraps every API call. Transient failures — 5xx, 429 — are
retried with exponential backoff. **Quota exhaustion and permission errors are
not retried**; they raise immediately with a message saying what to do,
because retrying them only burns time.

Quota cost is `1 + 2·ceil(N/50)` units for N videos — about 41 for a
1000-video channel against a 10,000/day default. Ingesting daily is cheap, and
daily ingestion is what builds the time series `verdict` reads.

---

## The demo path

`hindsight demo` bypasses `auth` and `ingest` entirely and loads a bundled
JSON snapshot in exactly the shape `db.load_videos_with_latest_stats` returns.
Every downstream stage is unaware it is running on demo data.

That equivalence is deliberate: a demo mode running a different code path
proves nothing about the tool. See `demo.py` for what is real in the dataset
(timing, views, likes, durations, tags) and what is synthetic (ids, channel
identity, title text), and why each choice was made.
