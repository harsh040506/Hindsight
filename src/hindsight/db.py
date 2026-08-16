"""
SQLite storage for the catalog.

Two design decisions worth stating up front:

1. **Stat snapshots are append-only.** Every ingest writes a new row per video
   into `stat_snapshots` rather than overwriting a `views` column. One ingest
   gives you a photograph; several give you a time series, which is what makes
   `hindsight verdict` able to say "this arm gained views faster" instead of
   just "this arm has more views". The cost is a few hundred KB per ingest.

2. **Video metadata is upserted.** Titles and descriptions can be edited after
   publish, and the current value is the one that matters for attribution, so
   the latest ingest wins.

The database is a pure cache. Deleting it loses nothing that `hindsight
ingest` cannot rebuild from the API.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import config

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channels (
    channel_id        TEXT PRIMARY KEY,
    slug              TEXT,
    title             TEXT,
    subscriber_count  INTEGER,
    view_count        INTEGER,
    video_count       INTEGER,
    uploads_playlist  TEXT,
    last_ingest_utc   TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    video_id       TEXT PRIMARY KEY,
    channel_id     TEXT NOT NULL,
    title          TEXT,
    description    TEXT,
    published_utc  TEXT NOT NULL,
    duration_s     INTEGER,
    privacy        TEXT,
    tags_json      TEXT,
    category_id    TEXT,
    thumbnail_url  TEXT,
    first_seen_utc TEXT,
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
);

CREATE INDEX IF NOT EXISTS idx_videos_channel_pub
    ON videos(channel_id, published_utc);

-- Append-only. One row per (video, ingest run).
CREATE TABLE IF NOT EXISTS stat_snapshots (
    video_id     TEXT NOT NULL,
    observed_utc TEXT NOT NULL,
    views        INTEGER NOT NULL DEFAULT 0,
    likes        INTEGER NOT NULL DEFAULT 0,
    comments     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (video_id, observed_utc),
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_video
    ON stat_snapshots(video_id, observed_utc DESC);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id  TEXT PRIMARY KEY,
    channel_id     TEXT NOT NULL,
    feature        TEXT NOT NULL,
    hypothesis     TEXT,
    arms_json      TEXT NOT NULL,
    min_per_arm    INTEGER NOT NULL,
    created_utc    TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'designed',
    concluded_utc  TEXT,
    verdict_json   TEXT
);
"""


# --------------------------------------------------------------------------
# Connection handling
# --------------------------------------------------------------------------


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """
    Open the catalog database, creating and migrating it if necessary.

    WAL mode is enabled so a long ingest does not block a concurrent read
    (useful when you leave a report open while re-ingesting).
    """
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)

    stored = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()
    if stored is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    elif int(stored["value"]) != SCHEMA_VERSION:
        raise RuntimeError(
            f"Catalog at {path} uses schema v{stored['value']}, but this "
            f"version of Hindsight expects v{SCHEMA_VERSION}. The catalog is "
            f"a rebuildable cache -- delete it and re-run `hindsight ingest`."
        )

    return conn


@contextmanager
def session(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager that commits on success and rolls back on error."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def utcnow_iso() -> str:
    """Current UTC time as a second-resolution ISO string (our row key)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


def upsert_channel(conn: sqlite3.Connection, channel: dict[str, Any]) -> None:
    """Insert or update a channel row and stamp its last ingest time."""
    conn.execute(
        """
        INSERT INTO channels (channel_id, slug, title, subscriber_count,
                              view_count, video_count, uploads_playlist,
                              last_ingest_utc)
        VALUES (:channel_id, :slug, :title, :subscriber_count, :view_count,
                :video_count, :uploads_playlist, :last_ingest_utc)
        ON CONFLICT(channel_id) DO UPDATE SET
            slug             = excluded.slug,
            title            = excluded.title,
            subscriber_count = excluded.subscriber_count,
            view_count       = excluded.view_count,
            video_count      = excluded.video_count,
            uploads_playlist = excluded.uploads_playlist,
            last_ingest_utc  = excluded.last_ingest_utc
        """,
        {**channel, "last_ingest_utc": utcnow_iso()},
    )


def upsert_videos(conn: sqlite3.Connection, videos: Iterable[dict[str, Any]]) -> int:
    """
    Insert or update video metadata rows.

    `first_seen_utc` is preserved across updates -- it records when Hindsight
    first observed the video, which is not the same as its publish time and is
    useful for spotting back-dated or re-published uploads.
    """
    now = utcnow_iso()
    rows = list(videos)
    conn.executemany(
        """
        INSERT INTO videos (video_id, channel_id, title, description,
                            published_utc, duration_s, privacy, tags_json,
                            category_id, thumbnail_url, first_seen_utc)
        VALUES (:video_id, :channel_id, :title, :description, :published_utc,
                :duration_s, :privacy, :tags_json, :category_id,
                :thumbnail_url, :first_seen_utc)
        ON CONFLICT(video_id) DO UPDATE SET
            title         = excluded.title,
            description   = excluded.description,
            published_utc = excluded.published_utc,
            duration_s    = excluded.duration_s,
            privacy       = excluded.privacy,
            tags_json     = excluded.tags_json,
            category_id   = excluded.category_id,
            thumbnail_url = excluded.thumbnail_url
        """,
        [{**v, "first_seen_utc": now} for v in rows],
    )
    return len(rows)


def insert_snapshots(
    conn: sqlite3.Connection, snapshots: Iterable[dict[str, Any]], observed_utc: str
) -> int:
    """
    Append one stat snapshot per video for this ingest run.

    All rows in a run share `observed_utc` so a run can be selected as a unit.
    `INSERT OR REPLACE` keeps a re-run within the same second idempotent.
    """
    rows = list(snapshots)
    conn.executemany(
        """
        INSERT OR REPLACE INTO stat_snapshots
            (video_id, observed_utc, views, likes, comments)
        VALUES (:video_id, :observed_utc, :views, :likes, :comments)
        """,
        [{**s, "observed_utc": observed_utc} for s in rows],
    )
    return len(rows)


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def get_channel(conn: sqlite3.Connection, channel_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM channels WHERE channel_id = ?", (channel_id,)
    ).fetchone()


def list_channels(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM channels ORDER BY slug").fetchall()


def resolve_channel(conn: sqlite3.Connection, ref: str | None) -> sqlite3.Row:
    """
    Resolve a channel by id, slug, or -- if only one exists -- implicitly.

    Raises a ValueError carrying an actionable message rather than returning
    None, because every caller would otherwise write the same error handling.
    """
    channels = list_channels(conn)
    if not channels:
        raise ValueError("No channels in the catalog yet. Run `hindsight ingest` first.")

    if ref is None:
        if len(channels) == 1:
            return channels[0]
        slugs = ", ".join(c["slug"] or c["channel_id"] for c in channels)
        raise ValueError(
            f"Catalog holds {len(channels)} channels; pass --channel to pick one "
            f"of: {slugs}"
        )

    for c in channels:
        if ref in (c["channel_id"], c["slug"]):
            return c

    slugs = ", ".join(c["slug"] or c["channel_id"] for c in channels)
    raise ValueError(f"No channel matching {ref!r}. Known channels: {slugs}")


def latest_observation_time(conn: sqlite3.Connection, channel_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT MAX(s.observed_utc) AS t
        FROM stat_snapshots s
        JOIN videos v ON v.video_id = s.video_id
        WHERE v.channel_id = ?
        """,
        (channel_id,),
    ).fetchone()
    return row["t"] if row else None


def load_videos_with_latest_stats(
    conn: sqlite3.Connection, channel_id: str, include_private: bool = False
) -> list[dict[str, Any]]:
    """
    Load every video for a channel joined to its most recent stat snapshot.

    Private and unlisted videos are excluded by default: they were never
    distributed, so their view counts describe YouTube's access controls
    rather than the content, and including them would poison every cohort.
    """
    privacy_clause = "" if include_private else "AND v.privacy = 'public'"
    rows = conn.execute(
        f"""
        SELECT v.*, s.views, s.likes, s.comments, s.observed_utc
        FROM videos v
        JOIN (
            SELECT video_id, views, likes, comments, observed_utc,
                   ROW_NUMBER() OVER (
                       PARTITION BY video_id ORDER BY observed_utc DESC
                   ) AS rn
            FROM stat_snapshots
        ) s ON s.video_id = v.video_id AND s.rn = 1
        WHERE v.channel_id = ? {privacy_clause}
        ORDER BY v.published_utc ASC
        """,
        (channel_id,),
    ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d.pop("tags_json") or "[]")
        out.append(d)
    return out


def snapshot_history(
    conn: sqlite3.Connection, video_ids: list[str]
) -> dict[str, list[sqlite3.Row]]:
    """Full snapshot time series for the given videos, oldest first."""
    if not video_ids:
        return {}
    placeholders = ",".join("?" * len(video_ids))
    rows = conn.execute(
        f"""
        SELECT video_id, observed_utc, views, likes, comments
        FROM stat_snapshots
        WHERE video_id IN ({placeholders})
        ORDER BY video_id, observed_utc ASC
        """,
        video_ids,
    ).fetchall()

    history: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        history.setdefault(r["video_id"], []).append(r)
    return history


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def save_experiment(conn: sqlite3.Connection, exp: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO experiments (experiment_id, channel_id, feature, hypothesis,
                                 arms_json, min_per_arm, created_utc, status)
        VALUES (:experiment_id, :channel_id, :feature, :hypothesis, :arms_json,
                :min_per_arm, :created_utc, :status)
        ON CONFLICT(experiment_id) DO UPDATE SET
            feature     = excluded.feature,
            hypothesis  = excluded.hypothesis,
            arms_json   = excluded.arms_json,
            min_per_arm = excluded.min_per_arm,
            status      = excluded.status
        """,
        exp,
    )


def list_experiments(
    conn: sqlite3.Connection, channel_id: str | None = None
) -> list[sqlite3.Row]:
    if channel_id:
        return conn.execute(
            "SELECT * FROM experiments WHERE channel_id = ? ORDER BY created_utc DESC",
            (channel_id,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM experiments ORDER BY created_utc DESC"
    ).fetchall()


def get_experiment(conn: sqlite3.Connection, experiment_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
    ).fetchone()


def conclude_experiment(
    conn: sqlite3.Connection, experiment_id: str, verdict: dict[str, Any]
) -> None:
    conn.execute(
        """
        UPDATE experiments
        SET status = 'concluded', concluded_utc = ?, verdict_json = ?
        WHERE experiment_id = ?
        """,
        (utcnow_iso(), json.dumps(verdict), experiment_id),
    )
