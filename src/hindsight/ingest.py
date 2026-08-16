"""
Catalog ingestion: YouTube Data API -> local SQLite.

The API surface used here is entirely read-only:

    channels.list        identify the channel and find its uploads playlist
    playlistItems.list   enumerate every video id (50 per page)
    videos.list          fetch metadata + public statistics (50 per call)

Quota cost is roughly `1 + ceil(N/50) + ceil(N/50)` units for N videos, so a
1000-video channel costs about 41 units against a 10,000/day default quota.
Ingesting is cheap; you can re-run it daily to build the time series that
`hindsight verdict` needs.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from googleapiclient.errors import HttpError

from . import config, db
from .auth import Session

log = logging.getLogger(__name__)

# Reasons that mean "stop, waiting will not help today".
_FATAL_REASONS = {"quotaExceeded", "dailyLimitExceeded", "accessNotConfigured",
                  "forbidden", "authError"}

_ISO_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


class IngestError(RuntimeError):
    """Raised when ingestion cannot proceed (quota, permissions, bad channel)."""


@dataclass
class IngestResult:
    channel_id: str
    channel_title: str
    slug: str
    videos_seen: int = 0
    snapshots_written: int = 0
    observed_utc: str = ""
    api_calls: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_iso_duration(value: str | None) -> int:
    """
    Convert an ISO-8601 duration ("PT1M33S") to whole seconds.

    YouTube reports `P0D` for live broadcasts and videos still processing;
    that parses to 0, and analysis treats 0-length videos as unusable rather
    than as genuinely zero-second content.
    """
    if not value:
        return 0
    m = _ISO_DURATION.match(value)
    if not m:
        log.debug("Unparseable duration %r", value)
        return 0
    parts = {k: int(v) for k, v in m.groupdict(default="0").items()}
    return (
        parts["days"] * 86400
        + parts["hours"] * 3600
        + parts["minutes"] * 60
        + parts["seconds"]
    )


def _execute(request: Callable[[], Any], what: str) -> Any:
    """
    Execute an API request with bounded exponential backoff.

    Transient failures (5xx, rate limiting) are retried. Quota exhaustion and
    permission errors are not -- they raise immediately with a message that
    says what to do, because retrying them just burns time.
    """
    from .auth import _http_reason

    delay = config.BACKOFF_BASE_S
    last: Exception | None = None

    for attempt in range(1, config.API_MAX_RETRIES + 1):
        try:
            return request()
        except HttpError as exc:
            reason = _http_reason(exc)
            status = exc.resp.status if exc.resp else 0
            last = exc

            if any(r in reason for r in _FATAL_REASONS):
                raise IngestError(
                    f"{what} failed permanently -- {reason}\n"
                    f"If this is a quota error, the daily limit resets at "
                    f"midnight Pacific. If it is 'accessNotConfigured', enable "
                    f"the YouTube Data API v3 on your Google Cloud project."
                ) from exc

            if status and status < 500 and status != 429:
                raise IngestError(f"{what} failed -- {reason}") from exc

            log.warning(
                "%s failed (attempt %d/%d, %s); retrying in %.1fs",
                what, attempt, config.API_MAX_RETRIES, reason, delay,
            )
        except (TimeoutError, ConnectionError) as exc:  # pragma: no cover
            last = exc
            log.warning("%s network error (attempt %d): %s", what, attempt, exc)

        time.sleep(delay)
        delay *= 2

    raise IngestError(f"{what} failed after {config.API_MAX_RETRIES} attempts: {last}")


def fetch_channel(session: Session) -> tuple[dict[str, Any], int]:
    """
    Fetch the target channel record. Returns (channel, api_calls_used).

    In OAuth mode this is the authenticated user's own channel. In API-key
    mode it is whatever public channel was named, looked up by id or handle.
    """
    part = "snippet,statistics,contentDetails"
    ref = session.channel_ref

    if ref is None:
        selector = {"mine": True}
    elif ref.startswith("@"):
        selector = {"forHandle": ref}
    else:
        selector = {"id": ref}

    resp = _execute(
        lambda: session.youtube.channels().list(part=part, **selector).execute(),
        "channels.list",
    )
    items = resp.get("items") or []

    if not items:
        if ref is None:
            raise IngestError(
                "The credentials are valid but no channel is attached to them. "
                "Confirm you authorised the Google account that owns the channel."
            )
        raise IngestError(
            f"No public channel found for {ref!r}. Check the handle or id -- "
            f"handles are case-sensitive and must include the leading '@'."
        )

    item = items[0]
    stats = item.get("statistics", {})
    return {
        "channel_id": item["id"],
        "slug": session.slug,
        "title": item["snippet"]["title"],
        "subscriber_count": int(stats.get("subscriberCount", 0) or 0),
        "view_count": int(stats.get("viewCount", 0) or 0),
        "video_count": int(stats.get("videoCount", 0) or 0),
        "uploads_playlist": item["contentDetails"]["relatedPlaylists"]["uploads"],
    }, 1


def iter_video_ids(session: Session, uploads_playlist: str) -> Iterator[list[str]]:
    """
    Yield pages of video ids from the uploads playlist.

    Yields in pages rather than one flat list so the caller can pipeline
    metadata fetches and show progress on very large channels.
    """
    page_token: str | None = None
    while True:
        resp = _execute(
            lambda t=page_token: session.youtube.playlistItems()
            .list(
                part="contentDetails",
                playlistId=uploads_playlist,
                maxResults=config.API_BATCH_SIZE,
                pageToken=t,
            )
            .execute(),
            "playlistItems.list",
        )
        ids = [i["contentDetails"]["videoId"] for i in resp.get("items", [])]
        if ids:
            yield ids
        page_token = resp.get("nextPageToken")
        if not page_token:
            return


def fetch_video_details(
    session: Session, video_ids: list[str], channel_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fetch metadata and statistics for up to API_BATCH_SIZE video ids.

    Returns (video_rows, snapshot_rows). Videos with hidden statistics (the
    owner can disable public counts) yield 0s; analysis filters those out
    rather than treating them as genuine zeros.
    """
    import json as _json

    resp = _execute(
        lambda: session.youtube.videos()
        .list(part="snippet,statistics,contentDetails,status", id=",".join(video_ids))
        .execute(),
        "videos.list",
    )

    videos: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    for item in resp.get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        thumbs = snippet.get("thumbnails", {})
        best = thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium") or {}

        videos.append({
            "video_id": item["id"],
            "channel_id": channel_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "published_utc": snippet.get("publishedAt", ""),
            "duration_s": parse_iso_duration(
                item.get("contentDetails", {}).get("duration")
            ),
            "privacy": item.get("status", {}).get("privacyStatus", "unknown"),
            "tags_json": _json.dumps(snippet.get("tags") or []),
            "category_id": snippet.get("categoryId", ""),
            "thumbnail_url": best.get("url", ""),
        })

        snapshots.append({
            "video_id": item["id"],
            "views": int(stats.get("viewCount", 0) or 0),
            "likes": int(stats.get("likeCount", 0) or 0),
            "comments": int(stats.get("commentCount", 0) or 0),
        })

    return videos, snapshots


def ingest_channel(
    session: Session,
    db_path: Any = None,
    limit: int | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> IngestResult:
    """
    Pull a channel's full catalog into the local database.

    `limit` caps how many videos are fetched (newest first) -- useful for a
    quick smoke test against a large channel. `progress` is called with
    (videos_done, videos_total_or_None) as pages complete.
    """
    channel, calls = fetch_channel(session)
    result = IngestResult(
        channel_id=channel["channel_id"],
        channel_title=channel["title"],
        slug=channel["slug"],
        api_calls=calls,
    )

    observed = db.utcnow_iso()
    result.observed_utc = observed
    expected = channel["video_count"] if limit is None else min(limit, channel["video_count"])

    with db.session(db_path) as conn:
        db.upsert_channel(conn, channel)

        collected = 0
        for page in iter_video_ids(session, channel["uploads_playlist"]):
            if limit is not None:
                page = page[: max(0, limit - collected)]
                if not page:
                    break

            result.api_calls += 1
            videos, snapshots = fetch_video_details(session, page, channel["channel_id"])
            result.api_calls += 1

            db.upsert_videos(conn, videos)
            db.insert_snapshots(conn, snapshots, observed)

            collected += len(page)
            result.videos_seen += len(videos)
            result.snapshots_written += len(snapshots)

            if progress:
                progress(result.videos_seen, expected)

            if limit is not None and collected >= limit:
                break

    missing = result.videos_seen - result.snapshots_written
    if missing:  # pragma: no cover - defensive
        result.warnings.append(f"{missing} videos returned metadata but no statistics")

    if limit is None and channel["video_count"] > result.videos_seen:
        result.warnings.append(
            f"Channel reports {channel['video_count']} videos but the uploads "
            f"playlist returned {result.videos_seen}. The difference is normally "
            f"videos that are deleted, private-by-owner, or still processing."
        )

    return result
