"""
Feature extraction: turning a published video into the set of choices that
produced it.

Every feature here answers one question: "what did the creator decide, and
could they decide differently next time?" That test excludes a lot of things
that correlate with views but are not levers -- the video id, the subscriber
count on the day, whether it happened to be a Tuesday in a month when the
channel was trending. A finding you cannot act on is a distraction.

Features are expressed as *categorical buckets* regardless of the underlying
type. A 33-second video is in bucket "33s"; a 412-character description is in
bucket "351-500 chars". This keeps one code path in the analysis and, more
importantly, it keeps the reported finding in the same shape as the decision:
nobody sets "description length = 0.34 standard deviations above mean", they
pick a length.

BUCKETING IS ADAPTIVE

Numeric thresholds are derived from the channel's own distribution, never
hardcoded. A shorts channel and a long-form channel have nothing in common on
an absolute scale, and a tool that ships with "short = under 60s" baked in
gives the long-form channel one bucket and no findings. `adaptive_buckets`
picks exact-value buckets when the data clusters on a few discrete values
(which is what an automated pipeline with fixed render settings produces) and
falls back to quartiles when it is genuinely continuous.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence

# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

# Emoji ranges covering the blocks that actually appear in video titles:
# pictographs, symbols, transport, dingbats, supplemental, and extended-A.
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002190-\U000021FF"
    "]"
)

_WORD = re.compile(r"[a-z']+")

# Function words carry no topical signal; excluding them keeps the discovered
# keyword features about subject matter instead of grammar.
_STOPWORDS = frozenset("""
a an and are as at be been being but by can cannot could did do does doing for
from had has have having he her hers him his how i if in into is it its me more
most my no nor not of on once only or other our ours out over own same she
should so some such than that the their theirs them then there these they this
those through to too under until up very was we were what when where which
while who whom why will with you your yours it's don't won't
""".split())


def has_emoji(text: str) -> bool:
    return bool(_EMOJI.search(text or ""))


def emojis_in(text: str) -> list[str]:
    return _EMOJI.findall(text or "")


def words_in(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def content_words(text: str) -> list[str]:
    return [w for w in words_in(text) if w not in _STOPWORDS and len(w) > 2]


# --------------------------------------------------------------------------
# Feature definition
# --------------------------------------------------------------------------


@dataclass
class Feature:
    """
    One controllable dimension of a video, and how to read it off a record.

    `lever` is the sentence the report shows a creator: it must describe the
    action they would take, not the statistic. "Change how long the video is"
    is a lever; "duration_s" is a column name.
    """

    key: str
    label: str
    lever: str
    group: str                                    # metadata | title | timing
    extract: Callable[[dict[str, Any]], str | None]
    description: str = ""
    buckets: list[str] = field(default_factory=list)

    # Keys of features this one mechanically subsumes. Some levers cannot be
    # moved independently: changing the tag *set* from five tags to fifteen
    # also changes the tag *count*, so proposing both as separate experiments
    # would spend two test cycles answering one question -- and the two tests
    # would be confounded with each other. The experiment ranker drops any
    # candidate that a higher-ranked candidate supersedes.
    supersedes: tuple[str, ...] = ()


def adaptive_buckets(
    values: Sequence[float],
    max_exact: int = 8,
    exact_coverage: float = 0.80,
) -> Callable[[float], str | None]:
    """
    Build a labeller for a numeric feature, adapted to how the data is shaped.

    If at most `max_exact` distinct values account for `exact_coverage` of the
    catalog, label by exact value -- an automated pipeline rendering everything
    at 12s, 17s or 33s should be analysed on those three real settings, not on
    quartiles that split them arbitrarily.

    Otherwise fall back to quartile ranges, labelled with their actual bounds
    so the reader can see what the bucket contains.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return lambda v: None

    counts = Counter(clean)
    common = counts.most_common(max_exact)
    coverage = sum(c for _, c in common) / len(clean)

    if coverage >= exact_coverage and len(counts) > 1:
        keep = {v for v, _ in common}
        return lambda v: (_fmt_num(v) if v in keep else None)

    srt = sorted(clean)

    def q(frac: float) -> float:
        return srt[min(len(srt) - 1, int(len(srt) * frac))]

    q1, q2, q3 = q(0.25), q(0.50), q(0.75)
    if q1 == q3:  # degenerate: almost all one value
        return lambda v: _fmt_num(v) if v == q2 else None

    def label(v: float) -> str | None:
        if v is None:
            return None
        if v <= q1:
            return f"<= {_fmt_num(q1)}"
        if v <= q2:
            return f"{_fmt_num(q1)}-{_fmt_num(q2)}"
        if v <= q3:
            return f"{_fmt_num(q2)}-{_fmt_num(q3)}"
        return f"> {_fmt_num(q3)}"

    return label


def _fmt_num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _published(video: dict[str, Any]) -> datetime | None:
    raw = video.get("published_utc")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:  # pragma: no cover - malformed API data
        return None


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------


def build_features(
    videos: Sequence[dict[str, Any]],
    timezone_offset_h: float = 0.0,
    keyword_count: int = 12,
    min_keyword_docs: int = 25,
) -> list[Feature]:
    """
    Construct the feature set for a specific catalog.

    Bucket boundaries and the discovered keyword vocabulary both depend on the
    channel's own data, so features are built per-catalog rather than declared
    as module-level constants.

    `timezone_offset_h` shifts publish-hour features into the creator's local
    time. The API reports UTC; "post at 18:00" is only actionable advice if
    18:00 means what the creator thinks it means.
    """
    features: list[Feature] = []

    # -- Metadata levers ---------------------------------------------------

    dur_label = adaptive_buckets([v.get("duration_s") or 0 for v in videos])
    features.append(Feature(
        key="duration",
        label="Video length",
        lever="Render the video at a different length",
        group="metadata",
        extract=lambda v: (
            None if not v.get("duration_s") else _suffix(dur_label(v["duration_s"]), "s")
        ),
        description="Runtime in seconds, bucketed to the lengths this channel "
                    "actually renders.",
    ))

    desc_label = adaptive_buckets([len(v.get("description") or "") for v in videos])
    features.append(Feature(
        key="description_length",
        label="Description length",
        lever="Write a longer or shorter description",
        group="metadata",
        extract=lambda v: _suffix(desc_label(len(v.get("description") or "")), " chars"),
        description="Total characters in the description, including hashtags.",
    ))

    tag_label = adaptive_buckets([len(v.get("tags") or []) for v in videos])
    features.append(Feature(
        key="tag_count",
        label="Number of tags",
        lever="Add or remove tags",
        group="metadata",
        extract=lambda v: _suffix(tag_label(len(v.get("tags") or [])), " tags"),
        description="How many tags were attached at upload.",
    ))

    features.append(Feature(
        key="tag_set",
        label="Tag set used",
        lever="Use a different set of tags",
        group="metadata",
        extract=lambda v: ", ".join(sorted(v.get("tags") or [])[:5]) or "(none)",
        description="The exact combination of tags, as a single choice.",
        # Testing which tags you use necessarily changes how many you use, so
        # the two cannot be run as independent experiments.
        supersedes=("tag_count",),
    ))

    features.append(Feature(
        key="description_cta",
        label="Call to action in description",
        lever="Include or drop the comment/share prompt",
        group="metadata",
        extract=lambda v: _cta_bucket(v.get("description") or ""),
        description="Whether the description asks the viewer to comment, share, "
                    "or follow.",
    ))

    # -- Title levers ------------------------------------------------------

    title_words = adaptive_buckets([len(words_in(v.get("title") or "")) for v in videos])
    features.append(Feature(
        key="title_words",
        label="Title word count",
        lever="Write a shorter or longer title",
        group="title",
        extract=lambda v: _suffix(title_words(len(words_in(v.get("title") or ""))), " words"),
        description="Words in the title. For a quote channel this is the length "
                    "of the quote itself.",
    ))

    title_chars = adaptive_buckets([len(v.get("title") or "") for v in videos])
    features.append(Feature(
        key="title_chars",
        label="Title character count",
        lever="Tighten or expand the title text",
        group="title",
        extract=lambda v: _suffix(title_chars(len(v.get("title") or "")), " chars"),
        description="Characters in the title, which drives whether it truncates "
                    "in the feed.",
    ))

    features.append(Feature(
        key="title_emoji",
        label="Emoji in title",
        lever="Add or remove the trailing emoji",
        group="title",
        extract=lambda v: "with emoji" if has_emoji(v.get("title") or "") else "no emoji",
        description="Whether the title carries any emoji at all.",
    ))

    emoji_counts = Counter(
        e for v in videos for e in set(emojis_in(v.get("title") or ""))
    )
    frequent_emoji = {e for e, c in emoji_counts.items() if c >= min_keyword_docs}
    if len(frequent_emoji) > 1:
        features.append(Feature(
            key="title_emoji_which",
            label="Which emoji",
            lever="Switch to a different emoji",
            group="title",
            extract=lambda v: _which_emoji(v.get("title") or "", frequent_emoji),
            description="Which specific emoji was used, among those used often "
                        "enough to compare.",
        ))

    features.append(Feature(
        key="title_first_word",
        label="Opening word",
        lever="Open the title on a different word",
        group="title",
        extract=lambda v: _first_word(v.get("title") or ""),
        description="The first word of the title -- the first thing a scrolling "
                    "viewer reads.",
    ))

    features.append(Feature(
        key="title_person",
        label="Second-person address",
        lever="Address the viewer as 'you' or keep it impersonal",
        group="title",
        extract=lambda v: (
            "addresses 'you'"
            if {"you", "your", "yours", "yourself"} & set(words_in(v.get("title") or ""))
            else "impersonal"
        ),
        description="Whether the title speaks directly to the viewer.",
    ))

    features.append(Feature(
        key="title_form",
        label="Sentence form",
        lever="Phrase it as a question, a command, or a statement",
        group="title",
        extract=lambda v: _sentence_form(v.get("title") or ""),
        description="Question, exclamation, or plain declarative statement.",
    ))

    features.append(Feature(
        key="title_clauses",
        label="Clause structure",
        lever="Use a single clause or a contrasting pair",
        group="title",
        extract=lambda v: _clause_bucket(v.get("title") or ""),
        description="Whether the title is one clause or several -- the "
                    "'X, but Y' construction versus a flat statement.",
    ))

    # Discovered vocabulary: which concepts show up often enough to test.
    doc_freq = Counter()
    for v in videos:
        doc_freq.update(set(content_words(v.get("title") or "")))
    keywords = [w for w, c in doc_freq.most_common(keyword_count * 3)
                if c >= min_keyword_docs][:keyword_count]

    for word in keywords:
        features.append(Feature(
            key=f"kw_{word}",
            label=f"Title contains '{word}'",
            lever=f"Write about '{word}' or avoid it",
            group="title",
            extract=(lambda w: lambda v: (
                f"contains '{w}'" if w in words_in(v.get("title") or "")
                else f"no '{w}'"
            ))(word),
            description=f"Whether the title uses the word '{word}'.",
        ))

    # -- Timing levers -----------------------------------------------------

    features.append(Feature(
        key="publish_hour",
        label="Publish hour",
        lever="Schedule the upload for a different hour",
        group="timing",
        extract=lambda v: _hour_bucket(v, timezone_offset_h),
        description="Hour of day the video went live, in the creator's local time.",
    ))

    features.append(Feature(
        key="publish_dow",
        label="Day of week",
        lever="Schedule the upload for a different day",
        group="timing",
        extract=lambda v: _dow(v, timezone_offset_h),
        description="Weekday the video went live, in the creator's local time.",
    ))

    features.append(Feature(
        key="publish_slot",
        label="Time of day",
        lever="Move the upload to a different part of the day",
        group="timing",
        extract=lambda v: _slot(v, timezone_offset_h),
        description="Coarse daypart: night, morning, afternoon, or evening.",
    ))

    return features


# --------------------------------------------------------------------------
# Individual extractors
# --------------------------------------------------------------------------


def _suffix(label: str | None, suffix: str) -> str | None:
    """Append a unit to a bucket label without mangling range labels."""
    if label is None:
        return None
    if label.startswith("<= ") or label.startswith("> "):
        return label + suffix
    if "-" in label:
        lo, hi = label.split("-", 1)
        return f"{lo}-{hi}{suffix}"
    return label + suffix


def _which_emoji(title: str, allowed: set[str]) -> str | None:
    found = [e for e in emojis_in(title) if e in allowed]
    if not found:
        return "no frequent emoji" if not emojis_in(title) else None
    return found[0]


def _first_word(title: str) -> str | None:
    ws = words_in(title)
    return ws[0] if ws else None


def _sentence_form(title: str) -> str:
    t = (title or "").strip()
    if t.endswith("?"):
        return "question"
    if t.endswith("!"):
        return "exclamation"
    return "statement"


def _clause_bucket(title: str) -> str:
    commas = (title or "").count(",")
    if commas == 0:
        return "single clause"
    if commas == 1:
        return "two clauses"
    return "three or more clauses"


def _cta_bucket(description: str) -> str:
    low = (description or "").lower()
    signals = ("comment ", "share this", "send this", "follow @", "subscribe",
               "tag someone", "save this")
    return "has CTA" if any(s in low for s in signals) else "no CTA"


def _local_dt(video: dict[str, Any], offset_h: float) -> datetime | None:
    dt = _published(video)
    if dt is None:
        return None
    from datetime import timedelta
    return dt + timedelta(hours=offset_h)


def _hour_bucket(video: dict[str, Any], offset_h: float) -> str | None:
    dt = _local_dt(video, offset_h)
    return f"{dt.hour:02d}:00" if dt else None


def _dow(video: dict[str, Any], offset_h: float) -> str | None:
    dt = _local_dt(video, offset_h)
    if dt is None:
        return None
    return ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"][dt.weekday()]


def _slot(video: dict[str, Any], offset_h: float) -> str | None:
    dt = _local_dt(video, offset_h)
    if dt is None:
        return None
    h = dt.hour
    if h < 6:
        return "night (00-06)"
    if h < 12:
        return "morning (06-12)"
    if h < 18:
        return "afternoon (12-18)"
    return "evening (18-24)"
