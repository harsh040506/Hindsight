"""
HTML report generation.

The report is a single self-contained file: no CDN, no external stylesheet, no
chart library. It opens from disk, works offline, and survives being emailed
to someone. Charts are inline SVG generated here in Python, so rendering does
not depend on JavaScript running -- the interaction layer adds tooltips on top
of a chart that is already fully legible without them.

Colour follows the roles in the design system rather than being chosen per
chart. The spotlight chart encodes *polarity* -- each bucket sits above or
below the 50th percentile, which is the "average video for its moment"
baseline -- so it uses the diverging blue/red pair with a neutral midpoint.
The distribution chart encodes a single magnitude and uses one sequential hue.
Both palettes were validated with the design system's checker in light and
dark mode before being written in.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from jinja2 import Environment, PackageLoader, select_autoescape

from . import __version__, analyze as analyze_mod, config
from .analyze import AnalysisResult, Finding
from .experiment import ExperimentPlan

# Design-system roles actually used by the charts. Both palettes validated:
# diverging poles pass all six checks in light and dark; the sequential ramp
# is the reference blue ramp used monotonically.
PALETTE = {
    "pos_light": "#2a78d6", "pos_dark": "#3987e5",   # above the 50 baseline
    "neg_light": "#d03b3b", "neg_dark": "#e66767",   # below the 50 baseline
    "seq_light": "#5598e7", "seq_dark": "#2a78d6",   # single-magnitude bars
}


def _env() -> Environment:
    env = Environment(
        loader=PackageLoader("hindsight", "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["commafy"] = lambda v: f"{int(v):,}"
    env.filters["signed"] = lambda v: f"{v:+.1f}"
    env.filters["pct"] = lambda v: f"{v:.1f}"
    return env


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def spotlight_chart(
    result: AnalysisResult, finding: Finding | None, width: int = 720, height: int = 300
) -> str | None:
    """
    Diverging bar chart for the feature behind the headline finding.

    Every bucket of that feature is drawn as a deviation from 50 -- the score
    of a video that performed exactly like its neighbours. Bars above the line
    beat their moment; bars below lost to it. Buckets whose result survived
    correction are labelled directly, so significance is carried by a label
    and not by colour alone.

    Returns None when there is no finding to spotlight, in which case the
    template omits the section rather than drawing an empty axis.
    """
    if finding is None:
        return None

    fr = next(
        (f for f in result.feature_results if f.feature.key == finding.feature_key),
        None,
    )
    if fr is None or not fr.tested_buckets:
        return None

    buckets = sorted(fr.tested_buckets, key=lambda b: b.bucket)
    if len(buckets) < 2:
        return None

    pad_l, pad_r, pad_t, pad_b = 46, 16, 28, 54
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    deviations = [b.mean_percentile - 50.0 for b in buckets]
    extent = max(12.0, max(abs(d) for d in deviations) * 1.25)
    mid_y = pad_t + plot_h / 2

    def y_for(dev: float) -> float:
        return mid_y - (dev / extent) * (plot_h / 2)

    gap = 2.0
    slot = plot_w / len(buckets)
    bar_w = max(4.0, slot - gap)
    radius = min(4.0, bar_w / 2)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'aria-label="Mean cohort percentile by {html.escape(fr.feature.label)}">'
    ]

    # Gridlines at quarter steps of the extent, recessive.
    for frac in (-1.0, -0.5, 0.5, 1.0):
        y = y_for(extent * frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'class="grid"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 3.5:.1f}" class="axis-label" '
            f'text-anchor="end">{50 + extent * frac:.0f}</text>'
        )

    for i, (b, dev) in enumerate(zip(buckets, deviations)):
        x = pad_l + i * slot + gap / 2
        y_top = y_for(dev) if dev >= 0 else mid_y
        h = abs(y_for(dev) - mid_y)
        cls = "bar-pos" if dev >= 0 else "bar-neg"
        if not b.significant:
            cls += " bar-muted"

        # Rounded ends only on the data end; the baseline end stays square.
        r = min(radius, h)
        if dev >= 0:
            path = (
                f"M{x:.1f},{y_top + h:.1f} V{y_top + r:.1f} "
                f"Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
                f"H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{y_top:.1f} "
                f"{x + bar_w:.1f},{y_top + r:.1f} V{y_top + h:.1f} Z"
            )
        else:
            y_bot = mid_y + h
            path = (
                f"M{x:.1f},{mid_y:.1f} V{y_bot - r:.1f} "
                f"Q{x:.1f},{y_bot:.1f} {x + r:.1f},{y_bot:.1f} "
                f"H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{y_bot:.1f} "
                f"{x + bar_w:.1f},{y_bot - r:.1f} V{mid_y:.1f} Z"
            )

        tip = (
            f"{b.bucket} — {b.mean_percentile:.1f} mean percentile, "
            f"{b.n} videos, median {b.median_views:,.0f} views"
            + (f", q={b.test.q_value:.3f}" if b.test else "")
        )
        parts.append(
            f'<path d="{path}" class="{cls}" tabindex="0" '
            f'data-tip="{html.escape(tip)}"><title>{html.escape(tip)}</title></path>'
        )

        if b.significant:
            ly = (y_top - 6) if dev >= 0 else (mid_y + h + 14)
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{ly:.1f}" class="bar-value" '
                f'text-anchor="middle">{b.mean_percentile:.0f}</text>'
            )

        if len(buckets) <= 16 or i % 2 == 0:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{height - pad_b + 18:.1f}" '
                f'class="axis-label" text-anchor="middle">'
                f'{html.escape(_short(b.bucket, 9))}</text>'
            )

    # The 50 baseline sits above the bars so it always reads.
    parts.append(
        f'<line x1="{pad_l}" y1="{mid_y:.1f}" x2="{pad_l + plot_w}" '
        f'y2="{mid_y:.1f}" class="baseline"/>'
    )
    parts.append(
        f'<text x="{pad_l - 8}" y="{mid_y + 3.5:.1f}" class="axis-label strong" '
        f'text-anchor="end">50</text>'
    )
    parts.append(
        f'<text x="{pad_l}" y="{height - 8}" class="axis-title">'
        f'{html.escape(fr.feature.label)} — mean cohort percentile '
        f'(50 = performed like its neighbours)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def distribution_chart(
    result: AnalysisResult, width: int = 720, height: int = 200, bins: int = 34
) -> str:
    """
    Histogram of raw view counts across the catalog.

    This chart exists to justify the method rather than to make a
    recommendation. The shape it shows -- a hard left pile with a long right
    tail -- is why Hindsight never compares raw view counts and never assumes
    a normal distribution anywhere in the analysis.
    """
    values = [s.metric_value for s in result.scored if s.eligible]
    if not values:
        return ""

    hi = max(values)
    lo = min(values)
    span = max(hi - lo, 1.0)
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / span * bins))
        counts[idx] += 1
    peak = max(counts) or 1

    pad_l, pad_r, pad_t, pad_b = 46, 16, 14, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    gap = 2.0
    slot = plot_w / bins
    bar_w = max(2.0, slot - gap)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'aria-label="Distribution of view counts across the catalog">'
    ]

    baseline_y = pad_t + plot_h
    for i, c in enumerate(counts):
        if c == 0:
            continue
        h = (c / peak) * plot_h
        x = pad_l + i * slot + gap / 2
        y = baseline_y - h
        r = min(4.0, bar_w / 2, h)
        bin_lo = lo + span * i / bins
        bin_hi = lo + span * (i + 1) / bins
        tip = f"{c} videos between {bin_lo:,.0f} and {bin_hi:,.0f} views"
        path = (
            f"M{x:.1f},{baseline_y:.1f} V{y + r:.1f} "
            f"Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{y:.1f} "
            f"{x + bar_w:.1f},{y + r:.1f} V{baseline_y:.1f} Z"
        )
        parts.append(
            f'<path d="{path}" class="bar-seq" tabindex="0" '
            f'data-tip="{html.escape(tip)}"><title>{html.escape(tip)}</title></path>'
        )

    median = result.summary["median_metric"]
    mx = pad_l + ((median - lo) / span) * plot_w
    parts.append(
        f'<line x1="{mx:.1f}" y1="{pad_t}" x2="{mx:.1f}" y2="{baseline_y:.1f}" '
        f'class="marker"/>'
    )
    parts.append(
        f'<text x="{mx + 6:.1f}" y="{pad_t + 12:.1f}" class="bar-value">'
        f'median {median:,.0f}</text>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{baseline_y:.1f}" x2="{pad_l + plot_w}" '
        f'y2="{baseline_y:.1f}" class="baseline"/>'
    )
    parts.append(
        f'<text x="{pad_l}" y="{height - 22:.1f}" class="axis-label">'
        f'{lo:,.0f}</text>'
    )
    parts.append(
        f'<text x="{pad_l + plot_w}" y="{height - 22:.1f}" class="axis-label" '
        f'text-anchor="end">{hi:,.0f}</text>'
    )
    parts.append(
        f'<text x="{pad_l}" y="{height - 6:.1f}" class="axis-title">'
        f'Lifetime views per video — the long tail is why raw view counts are '
        f'never compared directly</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def build_context(
    result: AnalysisResult,
    plan: ExperimentPlan | None = None,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """Assemble everything the template needs."""
    headline = result.findings[0] if result.findings else None
    top, bottom = analyze_mod.top_and_bottom(result, 5)

    tested = [f for f in result.feature_results if f.status == analyze_mod.TESTED]
    tested.sort(key=lambda f: -f.spread)

    return {
        "version": __version__,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "channel": result.channel,
        "summary": result.summary,
        "params": result.params,
        "findings": result.findings,
        "headline": headline,
        "untested": result.untested_levers,
        "underpowered": result.underpowered_levers,
        "tested_features": tested,
        "top_videos": top,
        "bottom_videos": bottom,
        "plan": plan,
        "demo_mode": demo_mode,
        "spotlight_svg": spotlight_chart(result, headline),
        "distribution_svg": distribution_chart(result),
        "palette": PALETTE,
        "alpha": config.ALPHA,
    }


def render_html(
    result: AnalysisResult,
    plan: ExperimentPlan | None = None,
    demo_mode: bool = False,
) -> str:
    """Render the full report to an HTML string."""
    template = _env().get_template("report.html.j2")
    return template.render(**build_context(result, plan, demo_mode))


def write_report(
    result: AnalysisResult,
    out_path: Path,
    plan: ExperimentPlan | None = None,
    demo_mode: bool = False,
) -> Path:
    """Render and write the report, creating parent directories as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(result, plan, demo_mode), encoding="utf-8")
    return out_path


def to_json(result: AnalysisResult, plan: ExperimentPlan | None = None) -> str:
    """
    Machine-readable dump of the analysis.

    Exists so Hindsight can sit inside somebody else's pipeline: the findings
    and the designed experiment are exactly the inputs a generator would need
    to act on the result automatically.
    """
    def encode(obj: Any) -> Any:
        if is_dataclass(obj) and not isinstance(obj, type):
            return {k: encode(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [encode(v) for v in obj]
        return obj

    payload = {
        "hindsight_version": __version__,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "channel": dict(result.channel),
        "parameters": result.params,
        "summary": {k: v for k, v in result.summary.items()},
        "findings": [encode(f) for f in result.findings],
        "untested_levers": [
            {"feature": fr.feature.key, "label": fr.feature.label,
             "lever": fr.feature.lever, "note": fr.note}
            for fr in result.untested_levers
        ],
        "underpowered_levers": [
            {"feature": fr.feature.key, "label": fr.feature.label, "note": fr.note}
            for fr in result.underpowered_levers
        ],
        "next_experiment": plan.to_dict() if plan else None,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
