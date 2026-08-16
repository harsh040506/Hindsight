# CLI reference

```
hindsight [--verbose] COMMAND [OPTIONS]
```

All commands accept `-h/--help`.

---

## `hindsight demo`

Run the full pipeline on the bundled catalog. No credentials.

| Flag | Default | Meaning |
|---|---|---|
| `--tz FLOAT` | `0.0` | Hours offset from UTC for publish-time features |
| `--out PATH` | `out/hindsight-demo.html` | Report destination |
| `--open / --no-open` | `--open` | Launch a browser when done |

```bash
hindsight demo --tz 5.5
```

---

## `hindsight auth list`

Find reusable OAuth token files.

| Flag | Default | Meaning |
|---|---|---|
| `--dir PATH` | `.` | Directory tree to scan |

## `hindsight auth login`

Run the OAuth flow and write a token. Requires an interactive desktop session.

| Flag | Default | Meaning |
|---|---|---|
| `--client-secrets PATH` | `client_secrets.json` | OAuth client config |
| `--out PATH` | `hindsight_token.json` | Where to write the token |

---

## `hindsight ingest`

Pull a channel's catalog into the local database. Read-only.

| Flag | Default | Meaning |
|---|---|---|
| `--token PATH` | auto-discovered | OAuth token file |
| `--api-key KEY` | `$YOUTUBE_API_KEY` | API key for public-channel mode |
| `--channel REF` | — | Channel id, `@handle` or URL. Required with `--api-key` |
| `--limit N` | all | Fetch only the newest N videos |

```bash
hindsight ingest --api-key $YOUTUBE_API_KEY --channel @some_handle
hindsight ingest --token assets/my_channel_token.json
hindsight ingest --limit 200                    # quick smoke test
```

Re-running is cheap and appends a new stat snapshot, building the time series
that `verdict` reads.

---

## Shared analysis options

Accepted by `analyze`, `report`, `design` and `verdict`:

| Flag | Default | Meaning |
|---|---|---|
| `--channel REF` | inferred | Which stored channel to use |
| `--tz FLOAT` | `0.0` | Hours offset from UTC for publish-time features |
| `--cohort N` | `25` | Cohort half-width in videos |
| `--min-age N` | `14` | Exclude videos younger than N days |
| `--min-bucket N` | `20` | Minimum videos in a bucket before testing it |

---

## `hindsight analyze`

Score the catalog and print what moved views.

| Flag | Meaning |
|---|---|
| `--json` | Emit machine-readable JSON instead of the terminal report |

```bash
hindsight analyze --tz 5.5
hindsight analyze --json > findings.json
```

Output sections: catalog summary, findings surviving FDR correction (with
confidence intervals and replication status), levers never tested, and levers
that varied too thinly to measure.

---

## `hindsight report`

Render the analysis as a standalone HTML page.

| Flag | Default | Meaning |
|---|---|---|
| `--out PATH` | `out/hindsight-<slug>.html` | Report destination |
| `--open / --no-open` | `--open` | Launch a browser |
| `--with-experiment / --no-experiment` | with | Include a designed next experiment |

The page has no external dependencies — it opens offline and survives being
emailed.

---

## `hindsight design`

Design the next A/B test.

| Flag | Default | Meaning |
|---|---|---|
| `--feature KEY` | auto | Force a specific lever |
| `--min-effect FLOAT` | auto | Smallest effect to detect, in percentile points |
| `--within-days FLOAT` | `90` | Time budget the test should read out within |
| `--save / --no-save` | `--save` | Record it so `verdict` can read it back |
| `--out PATH` | `out/<experiment_id>.json` | Manifest destination |

By default the test is **sized to the time budget**: given your upload rate,
Hindsight picks the smallest effect resolvable within `--within-days`. Pass
`--min-effect` to fix the effect instead and accept whatever duration follows.

```bash
hindsight design                          # auto-pick the best lever
hindsight design --within-days 30         # I need an answer this month
hindsight design --min-effect 5 --feature duration
```

The emitted JSON manifest is machine-consumable — arms, per-arm counts,
variables to hold constant, and the readout date — so a generator can act on
it directly.

---

## `hindsight verdict EXPERIMENT_ID`

Read out a running experiment against freshly ingested data.

```bash
hindsight ingest --token ...      # refresh first
hindsight verdict exp_20260816_tag_set
```

Three outcomes:

- **conclusive** — a difference survived testing; the winning arm is named and
  the experiment is marked concluded
- **inconclusive** — no detectable difference, reported alongside the smallest
  effect this test could have resolved
- **insufficient-data** — not enough eligible videos per arm yet

Arm membership is read from what was actually published, not from the plan, so
the readout stays honest if your pipeline drifted.

---

## `hindsight experiments`

List designed and concluded experiments.

| Flag | Meaning |
|---|---|
| `--channel REF` | Filter to one channel |

---

## Typical sessions

**Trying it out**

```bash
hindsight demo --tz 5.5
```

**Analysing a public channel**

```bash
export YOUTUBE_API_KEY=...
hindsight ingest --channel @some_handle
hindsight report --tz 5.5
```

**Running the loop on your own channel**

```bash
hindsight ingest --token my_token.json
hindsight analyze --tz 5.5
hindsight design --within-days 60
# ... publish according to the manifest ...
hindsight ingest --token my_token.json
hindsight verdict exp_20260816_tag_set
```
