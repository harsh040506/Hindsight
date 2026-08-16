# Setup

Three ways to run Hindsight, in increasing order of effort and access.

---

## 1. Demo — no credentials

```bash
python -m venv .venv
.venv/bin/pip install -e .          # Windows: .venv\Scripts\pip install -e .
hindsight demo --tz 5.5
```

Runs the whole pipeline on a bundled 949-video catalog and opens an HTML
report. Nothing to configure.

`--tz` is the timezone publish hours are reported in. The bundled channel
publishes on UTC+5:30, so `--tz 5.5` is what reproduces the numbers quoted in
the README; without it the same findings appear shifted into UTC.

Useful flags:

```bash
hindsight demo --tz 5.5             # publish-hour findings in your local time
hindsight demo --no-open            # don't launch a browser
hindsight demo --out report.html    # choose the output path
```

---

## 2. API key — any public channel

Read any public channel without OAuth. This is the easiest way to point
Hindsight at your own channel or someone else's.

**Get a key:**

1. Go to [console.cloud.google.com](https://console.cloud.google.com).
2. Create a project (or pick one).
3. **APIs & Services → Library** → enable **YouTube Data API v3**.
4. **APIs & Services → Credentials → Create credentials → API key**.

No OAuth consent screen is required.

```bash
export YOUTUBE_API_KEY=AIza...              # Windows: $env:YOUTUBE_API_KEY="AIza..."
hindsight ingest --channel @somehandle
hindsight analyze --tz 5.5
hindsight report
```

`--channel` accepts any of:

```
UCxxxxxxxxxxxxxxxxxxxxxx
@some_handle
https://www.youtube.com/@some_handle
https://www.youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxxxx
```

**Limits.** Public statistics only — which is everything the analysis uses.
Private and unlisted videos are invisible, but those are excluded from
analysis anyway since they were never distributed.

---

## 3. OAuth — your own channel

Needed only if you want Hindsight to see unlisted videos, or you plan to
enable the Analytics upgrade below.

### Reuse an existing token

If another tool already minted an OAuth token for your channel, Hindsight can
use it as long as it carries `youtube.readonly`:

```bash
hindsight auth list --dir /path/to/your/channels
hindsight ingest --token /path/to/channel_token.json
```

`auth list` scans for `*_token.json` and `token.json`, skipping virtualenvs.

### Mint a new one

1. In the Cloud console, **APIs & Services → Credentials → Create credentials
   → OAuth client ID → Desktop app**.
2. Download the JSON as `client_secrets.json`.
3. Run:

```bash
hindsight auth login
hindsight ingest --token hindsight_token.json
```

This opens a browser and starts a temporary local server to catch the
redirect, so it needs an interactive desktop session.

Expired tokens refresh automatically and are written back in place.

---

## Optional: retention and click-through

Hindsight works fully on public statistics. If you additionally want retention
and impression CTR, the YouTube **Analytics** API needs two things — and it is
common to have one without the other, so Hindsight probes for both and reports
which is missing.

1. **Enable the API.** Cloud console → **APIs & Services → Library** →
   **YouTube Analytics API** → Enable. Without this you get
   `accessNotConfigured` even with a perfect token.
2. **Re-authorise with the scope.** `hindsight auth login` requests
   `yt-analytics.readonly` as well. An existing token minted by another tool
   almost certainly lacks it, and scopes cannot be added to an issued token —
   you must mint a new one.

Check what you have:

```bash
hindsight ingest --token ... -v
```

The output states whether retention data is available and, if not, which of
the two conditions failed.

---

## Configuration

Optional environment variables, or a `.env` file in the working directory:

| Variable | Default | Purpose |
|---|---|---|
| `YOUTUBE_API_KEY` | — | API key for public-channel mode |
| `HINDSIGHT_HOME` | `.hindsight` | Where the catalog cache lives |
| `HINDSIGHT_OUT` | `out` | Where reports and manifests are written |
| `HINDSIGHT_TOKEN_DIR` | `.` | Directory scanned for token files |
| `HINDSIGHT_CLIENT_SECRETS` | `client_secrets.json` | OAuth client config |

Analysis parameters (cohort width, significance threshold, minimum bucket
size) live in `src/hindsight/config.py`, each with a comment explaining what
changing it does. Most are also exposed as CLI flags.

---

## Quota

Ingest costs `1 + 2·ceil(N/50)` units for N videos:

| Channel size | Units | Share of default 10,000/day |
|---|---|---|
| 100 videos | 5 | 0.05% |
| 1,000 videos | 41 | 0.4% |
| 10,000 videos | 401 | 4% |

Re-running daily is cheap and builds the stat history that `hindsight verdict`
needs to read out experiments.

---

## Troubleshooting

**`No OAuth token found and no --api-key given`**
Start with `hindsight demo`, or supply `--api-key` with `--channel`.

**`accessNotConfigured`**
The relevant API is not enabled on the Cloud project behind your key or OAuth
client. Enable YouTube Data API v3 (and YouTube Analytics API if you want
retention data).

**`quotaExceeded`**
Daily quota resets at midnight Pacific. Hindsight does not retry this — it
stops immediately rather than burning time.

**`No public channel found for '@handle'`**
Handles are case-sensitive and need the leading `@`. Try the `UC...` channel
id instead.

**`The credentials are valid but no channel is attached`**
You authorised a Google account that does not own a channel. Common with
Brand Accounts — re-run `hindsight auth login` and pick the channel's account
at the consent screen.

**Analysis reports no findings**
Often correct. Under a few hundred videos most buckets will not clear the
20-video floor. Check the "never tested" and "varied too thinly" sections —
they are usually where the value is on a small catalog.
