"""
Demo mode: the whole pipeline, on a real catalog, with no credentials.

WHY THIS EXISTS

Every part of Hindsight that matters is downstream of the data. The ingest
step is the only piece that needs Google credentials, and it is also the least
interesting piece. Requiring an OAuth consent screen before anyone can see
what the tool concludes is a bad trade -- so the repository ships a real
catalog snapshot and `hindsight demo` runs cohort scoring, feature analysis,
significance testing, the report, and experiment design against it with no
setup at all.

WHAT IS REAL IN THE BUNDLED DATASET

The snapshot is a genuine 949-video catalog from a shorts channel with roughly
317,000 lifetime views, spanning thirteen months. 938 of those are public and
845 clear the analysis filters. These fields are unmodified:

    published_utc   exact publish timestamps
    views           lifetime view counts
    likes           lifetime like counts
    comments        lifetime comment counts
    duration_s      real render lengths
    tags            the real tag list
    description     length only (the text is replaced by filler of equal size)

The headline finding demo mode reproduces -- that publish hour is associated
with a large, era-stable difference in views -- rests entirely on those real
fields.

WHAT IS SYNTHETIC, AND WHY

    video_id        hashed
    channel name    replaced
    title           regenerated

Titles are synthesised rather than shipped verbatim because a quote title is
searchable: publishing 949 real ones would identify the source channel and its
performance numbers no matter what the id column said. The surrogate titles
are built to preserve the *structure* the analysis reads -- word count, emoji
presence and choice, clause count, second-person address -- so every feature
extractor is exercised on realistic input and the code path under demo is the
same code path that runs on live data.

The honest consequence: title-derived findings in demo mode describe the
surrogate vocabulary, not the original channel's. Timing, duration, tag and
description-length findings are real. The report labels this rather than
letting the reader assume otherwise.
"""

from __future__ import annotations

import hashlib
import json
import random
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

DATASET_NAME = "demo_catalog.json"

# Vocabulary for surrogate titles. Drawn from public-domain philosophical
# terminology so the generated text reads plausibly and the discovered-keyword
# feature has something real to find, without reproducing any source title.
_NOUNS = [
    "silence", "wisdom", "shadow", "reason", "virtue", "solitude", "memory",
    "courage", "doubt", "clarity", "freedom", "meaning", "stillness", "fate",
    "reflection", "burden", "wonder", "patience", "conscience", "longing",
    "illusion", "presence", "restraint", "surrender", "intention", "distance",
]
_ADJECTIVES = [
    "quiet", "deepest", "honest", "hidden", "fragile", "certain", "ancient",
    "restless", "ordinary", "unspoken", "difficult", "luminous", "sober",
]
_VERBS = [
    "reveals", "demands", "outlasts", "precedes", "shapes", "dissolves",
    "answers", "resists", "sharpens", "survives", "begins", "endures",
]
_CONNECTORS = ["but", "yet", "and", "because", "until", "unless"]
_OPENERS = ["the", "true", "every", "your", "our", "real", "a"]


def _rng_for(seed_material: str) -> random.Random:
    """Deterministic RNG keyed to a video, so exports are reproducible."""
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _surrogate_title(original: str, rng: random.Random) -> str:
    """
    Build a title that matches the original's measurable structure.

    Preserves word count, emoji (the exact emoji character is kept, since it
    is not identifying and the "which emoji" feature depends on it), clause
    count, and whether the title addresses the viewer directly.
    """
    from .features import emojis_in, words_in

    words = words_in(original)
    target = max(3, len(words))
    emoji = emojis_in(original)
    commas = original.count(",")
    second_person = bool(
        {"you", "your", "yours", "yourself"} & set(words)
    )

    out: list[str] = [rng.choice(_OPENERS) if not second_person else "your"]
    while len(out) < target:
        remaining = target - len(out)
        if remaining > 3 and rng.random() < 0.35:
            out += [rng.choice(_NOUNS), rng.choice(_VERBS), rng.choice(_ADJECTIVES)]
        elif remaining > 1 and rng.random() < 0.5:
            out += [rng.choice(_ADJECTIVES), rng.choice(_NOUNS)]
        else:
            out.append(rng.choice(_NOUNS))
    out = out[:target]

    # Re-insert the same number of clause breaks the original had.
    text = " ".join(out)
    if commas:
        parts = text.split(" ")
        for _ in range(min(commas, max(0, len(parts) - 2))):
            idx = rng.randrange(1, len(parts) - 1)
            if not parts[idx].endswith(","):
                parts[idx] += ","
        text = " ".join(parts)

    if second_person and "your" not in text:
        text = "your " + text
    # Preserve terminal punctuation: the "sentence form" feature reads it, and
    # flattening every surrogate to a full stop would make a lever that really
    # did vary look as though it never had.
    stripped = (original or "").rstrip().rstrip("".join(emoji)).rstrip()
    ending = stripped[-1] if stripped and stripped[-1] in "?!." else "."

    text = text[0].upper() + text[1:] + ending
    if emoji:
        text += " " + emoji[0]
    return text


def anonymize(
    videos: Sequence[dict[str, Any]],
    channel: dict[str, Any],
    channel_slug: str = "demo_philosophy_channel",
    salt: str = "hindsight-demo-v1",
) -> dict[str, Any]:
    """
    Convert a live catalog into a shippable, non-identifying dataset.

    Deterministic: the same input always produces the same output, so the
    bundled dataset can be regenerated and diffed.
    """
    channel_id = "UC" + hashlib.sha256(
        (channel["channel_id"] + salt).encode()
    ).hexdigest()[:22]

    out_videos: list[dict[str, Any]] = []
    for v in videos:
        vid = hashlib.sha256((v["video_id"] + salt).encode()).hexdigest()[:11]
        rng = _rng_for(v["video_id"] + salt)
        desc_len = len(v.get("description") or "")

        # Only two things are ever read off a description: its length, and
        # whether it contains a call to action. Filler preserves the first;
        # re-inserting a real CTA phrase when the original had one preserves
        # the second. Dropping it would make a lever that was genuinely used
        # on 99% of uploads register as never used at all.
        from .features import _cta_bucket
        has_cta = _cta_bucket(v.get("description") or "") == "has CTA"
        marker = "Follow @demo_channel for more. " if has_cta else ""
        description = (marker + "x" * max(0, desc_len - len(marker)))[:desc_len or 0]

        out_videos.append({
            "video_id": vid,
            "channel_id": channel_id,
            "title": _surrogate_title(v.get("title") or "", rng),
            "description": description,
            "published_utc": v["published_utc"],
            "duration_s": v.get("duration_s") or 0,
            "privacy": v.get("privacy") or "public",
            "tags": list(v.get("tags") or []),
            "category_id": v.get("category_id") or "",
            "thumbnail_url": "",
            "views": int(v.get("views") or 0),
            "likes": int(v.get("likes") or 0),
            "comments": int(v.get("comments") or 0),
        })

    return {
        "_about": (
            "Anonymised real catalog snapshot bundled with Hindsight so the "
            "tool can be run with no credentials. Publish times, view counts, "
            "like counts, comment counts, durations and tags are unmodified "
            "real data. Video ids, channel identity and titles are synthetic "
            "-- see src/hindsight/demo.py for exactly how and why."
        ),
        "channel": {
            "channel_id": channel_id,
            "slug": channel_slug,
            "title": "Demo Philosophy Channel",
            "subscriber_count": int(channel.get("subscriber_count") or 0),
            "view_count": int(channel.get("view_count") or 0),
            "video_count": len(out_videos),
            "uploads_playlist": "",
            "last_ingest_utc": None,
        },
        "videos": out_videos,
    }


def dataset_path() -> Path:
    """Filesystem path to the bundled dataset."""
    return Path(str(resources.files("hindsight").joinpath(DATASET_NAME)))


def load_demo() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load the bundled catalog.

    Returns (videos, channel) in exactly the shape `db.load_videos_with_
    latest_stats` produces, so every downstream stage is unaware it is running
    on demo data. That equivalence is the point: demo mode must exercise the
    real pipeline, or it proves nothing.
    """
    path = dataset_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled demo dataset missing at {path}. Regenerate it with "
            f"`hindsight export-demo` from a live catalog."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = payload["videos"]
    for v in videos:
        v.setdefault("observed_utc", None)
    return videos, payload["channel"]


def write_dataset(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Write an anonymised dataset to disk."""
    target = path or dataset_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    return target
