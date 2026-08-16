"""
Configuration, paths, and credential discovery.

Everything tunable lives here as a module-level constant with a comment
explaining what moving it does. Values that are secret (API keys) or
machine-specific (where your OAuth tokens live) come from the environment,
never from this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Where Hindsight keeps its local state. Everything here is a cache derived
# from the YouTube API and can be deleted and rebuilt with `hindsight ingest`.
STATE_DIR = Path(os.getenv("HINDSIGHT_HOME", ".hindsight")).expanduser()

# SQLite catalog: channels, videos, and an append-only log of stat snapshots.
DB_PATH = STATE_DIR / "catalog.db"

# Rendered reports and experiment manifests land here.
OUT_DIR = Path(os.getenv("HINDSIGHT_OUT", "out")).expanduser()


# --------------------------------------------------------------------------
# YouTube API
# --------------------------------------------------------------------------

# Read-only scopes ONLY. Hindsight never requests upload/edit permission --
# it physically cannot modify your channel. If you point it at a token that
# happens to carry write scopes (because another tool created that token),
# Hindsight still only ever calls read endpoints.
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
]

# Optional upgrade. If your token carries this scope AND the YouTube Analytics
# API is enabled on your Google Cloud project, Hindsight will additionally
# pull retention and impression-CTR data. Without it, everything still works
# on public statistics (views/likes/comments). See docs/SETUP.md#analytics.
ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
ANALYTICS_API_SERVICE_NAME = "youtubeAnalytics"
ANALYTICS_API_VERSION = "v2"

# The Data API caps `videos.list` at 50 ids per call. Do not raise this.
API_BATCH_SIZE = 50

# Retries for transient (5xx / rate-limit) API failures, with exponential
# backoff starting at BACKOFF_BASE_S. Quota-exceeded errors are NOT retried --
# they are a hard stop until the quota window resets.
API_MAX_RETRIES = 5
BACKOFF_BASE_S = 1.5


# --------------------------------------------------------------------------
# Analysis defaults
# --------------------------------------------------------------------------

# Cohort half-width. Each video is scored against the COHORT_HALF_WIDTH videos
# published immediately before it and the same number after it. This is the
# core of the normalization -- see docs/METHODOLOGY.md.
#
# Smaller: tighter control for channel growth, noisier percentiles.
# Larger:  smoother percentiles, but slow channel-wide trends leak back in.
COHORT_HALF_WIDTH = 25

# Videos younger than this are excluded from analysis. They have not finished
# accumulating views, and including them drags down every bucket they land in.
MIN_AGE_DAYS = 14

# A feature bucket needs at least this many videos before Hindsight will
# report a lift for it. Below this, percentile differences are mostly noise.
MIN_BUCKET_N = 20

# Permutation-test iterations. 10k gives p-values stable to ~0.005, which is
# far more precision than any content decision needs.
PERMUTATION_ITERS = 10_000

# Findings at or below this p-value are reported as signal; everything else is
# reported as "not distinguishable from noise" and explicitly labelled so.
ALPHA = 0.05

# Bootstrap resamples for confidence intervals on the reported lift.
BOOTSTRAP_ITERS = 5_000

# Fixed seed so a given catalog always produces the same numbers. Reproducible
# output matters when you are about to make content decisions from it.
RANDOM_SEED = 20260816


# --------------------------------------------------------------------------
# Experiment design defaults
# --------------------------------------------------------------------------

# Target statistical power for `hindsight design`. The sample-size calculation
# answers: how many videos per arm to detect the effect we think is there?
TARGET_POWER = 0.80

# Smallest effect worth chasing, in cohort-percentile points. A 5-point shift
# in median percentile is roughly "this video now beats 55% of its neighbours
# instead of 50%" -- small but real and compounding over hundreds of uploads.
# Nothing is ever sized below this, however generous the time budget.
MIN_DETECTABLE_EFFECT = 5.0

# Default time budget for a designed experiment. Experiments are sized to read
# out within this window at the channel's current upload rate, rather than
# being sized to a fixed effect and taking however long that takes -- which on
# a channel posting a few times a day is routinely over a year.
DEFAULT_BUDGET_DAYS = 90.0


# --------------------------------------------------------------------------
# Credential discovery
# --------------------------------------------------------------------------

# Directory tree to scan for `*_token.json` OAuth files when no explicit
# --token is given. Set HINDSIGHT_TOKEN_DIR to your channel-assets root.
TOKEN_SEARCH_DIR = Path(os.getenv("HINDSIGHT_TOKEN_DIR", ".")).expanduser()

# OAuth client config, used only by `hindsight auth login` to mint a new token.
CLIENT_SECRETS_FILE = Path(
    os.getenv("HINDSIGHT_CLIENT_SECRETS", "client_secrets.json")
).expanduser()


@dataclass(frozen=True)
class TokenRef:
    """A discovered OAuth token file and the channel slug inferred from it."""

    path: Path
    slug: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.slug} ({self.path})"


def discover_tokens(search_dir: Path | None = None) -> list[TokenRef]:
    """
    Find OAuth token files under `search_dir`.

    Looks for files named `*_token.json` or `token.json`, skipping virtualenvs
    and version-control directories. The channel slug is the filename with the
    `_token.json` suffix stripped, which matches the layout produced by most
    multi-channel upload scripts.

    Returns tokens sorted by slug so `--channel` selection is stable between
    runs regardless of filesystem ordering.
    """
    root = (search_dir or TOKEN_SEARCH_DIR).expanduser()
    if not root.is_dir():
        return []

    skip = {".venv", "venv", "env", ".git", "node_modules", "__pycache__"}
    found: list[TokenRef] = []

    for path in root.rglob("*token*.json"):
        if any(part in skip for part in path.parts):
            continue
        name = path.name
        if name == "token.json":
            slug = path.parent.name
        elif name.endswith("_token.json"):
            slug = name[: -len("_token.json")]
        else:
            continue
        found.append(TokenRef(path=path.resolve(), slug=slug))

    return sorted(found, key=lambda t: t.slug)


def ensure_dirs() -> None:
    """Create the state and output directories if they do not exist."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
