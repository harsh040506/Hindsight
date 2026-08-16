"""
Command-line interface.

Design goals, in order:

  1. `hindsight demo` must work on a clean checkout with no credentials, no
     API key, and no configuration. It is the first thing anyone runs.
  2. Every command that can fail for an environmental reason (missing token,
     exhausted quota, empty catalog) must fail with a message that says what
     to do next, not a traceback.
  3. Output is readable in a terminal without a wide window, and every number
     shown is one the report can also show.
"""

from __future__ import annotations

import logging
import sys
import webbrowser
from pathlib import Path

import click

from . import __version__, analyze as analyze_mod, config, db, demo as demo_mod
from . import experiment as exp_mod, report as report_mod
from .auth import AuthError, open_public_session, open_session
from .ingest import IngestError, ingest_channel

# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

# Box-drawing characters and status glyphs are not encodable on every console.
# Windows in particular falls back to cp1252 whenever stdout is redirected --
# piping the output to a file or another program -- and printing "─" there
# raises UnicodeEncodeError *after* the real work has finished, which turns a
# successful run into a traceback. Both problems are handled: the streams are
# switched to UTF-8 where possible, and anything still unencodable degrades to
# ASCII rather than crashing.

_GLYPHS_UNICODE = {
    "rule": "─", "ok": "✓", "warn": "!", "up": "▲", "down": "▼",
    "dot": "·", "pick": "◆", "arrow": "→", "approx": "≈", "gte": "≥",
}
_GLYPHS_ASCII = {
    "rule": "-", "ok": "OK", "warn": "!", "up": "+", "down": "-",
    "dot": "-", "pick": "*", "arrow": "->", "approx": "~", "gte": ">=",
}

G = dict(_GLYPHS_ASCII)


def _init_output() -> None:
    """Prefer UTF-8 output; fall back to ASCII glyphs if it is unavailable."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass  # not a reconfigurable stream (pytest capture, a pipe wrapper)

    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "".join(_GLYPHS_UNICODE.values()).encode(encoding)
    except (LookupError, UnicodeEncodeError):  # pragma: no cover - platform
        G.update(_GLYPHS_ASCII)
    else:
        G.update(_GLYPHS_UNICODE)


def _hdr(text: str) -> None:
    click.echo()
    click.secho(text, bold=True)
    click.secho(G["rule"] * min(len(text), 68), dim=True)


def _kv(key: str, value: str) -> None:
    click.echo(f"  {click.style(key + ':', dim=True)} {value}")


def _ok(text: str) -> None:
    click.secho(f"  {G['ok']} {text}", fg="green")


def _warn(text: str) -> None:
    click.secho(f"  {G['warn']} {text}", fg="yellow")


def _die(text: str) -> None:
    click.secho(f"\nError: {text}", fg="red", err=True)
    sys.exit(1)


def _print_findings(result: analyze_mod.AnalysisResult) -> None:
    s = result.summary
    _hdr("Catalog")
    _kv("videos analysed", f"{s['eligible']:,} of {s['total']:,}")
    _kv("lifetime views", f"{int(s['total_metric']):,}")
    _kv("median / best", f"{int(s['median_metric']):,} / {int(s['max_metric']):,}")
    for reason, n in s["excluded"].items():
        click.echo(f"  {click.style('excluded:', dim=True)} {n} — {reason}")

    _hdr(f"Findings ({len(result.findings)} survived correction)")
    if not result.findings:
        click.echo("  Nothing survived false-discovery correction.")
        click.echo("  That is a real result: no lever you varied shows a")
        click.echo("  measurable effect yet. See the untested levers below.")
    for f in result.findings:
        arrow = G["up"] if f.lift > 0 else G["down"]
        colour = "cyan" if f.lift > 0 else "red"
        click.echo()
        click.secho(f"  {arrow} {f.lift:+.1f} pts", fg=colour, bold=True, nl=False)
        click.echo(f"  {f.feature_label} = {f.bucket}")
        click.echo(
            f"      n={f.n}  95% CI [{f.ci_low:+.1f}, {f.ci_high:+.1f}]  "
            f"q={f.q_value:.4f}  median {int(f.median_views):,} views"
        )
        if f.replicates:
            _ok("replicates in both halves of the catalog")
        else:
            _warn("does not replicate across eras — treat as a hypothesis")

    if result.untested_levers:
        _hdr(f"Never tested ({len(result.untested_levers)})")
        for fr in result.untested_levers:
            click.secho(f"  {G['pick']} {fr.feature.label}", fg="yellow", bold=True)
            click.echo(f"      {fr.note}")

    if result.underpowered_levers:
        _hdr(f"Varied too thinly to measure ({len(result.underpowered_levers)})")
        for fr in result.underpowered_levers:
            click.echo(f"  {G['dot']} {fr.feature.label}")


def _print_plan(plan: exp_mod.ExperimentPlan) -> None:
    _hdr("Next experiment")
    _kv("lever", f"{plan.feature_label} — {plan.lever.lower()}")
    _kv("why", plan.priority)
    click.echo()
    for arm in plan.arms:
        click.secho(f"  {arm.name}", bold=True, nl=False)
        click.echo(f" {G['arrow']} {arm.value}  ({arm.videos} videos)")
        click.echo(f"      {click.style(arm.note, dim=True)}")
    click.echo()
    _kv("total", f"{plan.total_videos} videos {G['approx']} "
                 f"{plan.estimated_days:.0f} days at {plan.uploads_per_day}/day")
    _kv("resolves", f"effects {G['gte']} {plan.min_detectable_effect} percentile "
                    f"points at {plan.target_power:.0%} power")
    _kv("assignment", "alternate arms upload by upload")
    for h in plan.hold_constant:
        _kv("hold constant", f"{h['label']} = {h['value']}")
    for w in plan.warnings:
        _warn(w)
    _kv("read out with", plan.readout_command)


def _load_catalog(channel: str | None) -> tuple[list, dict]:
    """Load a channel's videos from the local catalog, or explain why not."""
    try:
        with db.session() as conn:
            ch = db.resolve_channel(conn, channel)
            videos = db.load_videos_with_latest_stats(conn, ch["channel_id"])
    except ValueError as exc:
        _die(str(exc))
    if not videos:
        _die(
            f"No public videos stored for {ch['slug']}. Run `hindsight ingest` "
            f"first, or check that the channel has published videos."
        )
    return videos, dict(ch)


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", is_flag=True, help="Show debug logging.")
@click.version_option(__version__, prog_name="hindsight")
def main(verbose: bool) -> None:
    """
    Hindsight — find out which of your uploads' choices actually worked,
    then design the experiment that settles the next question.

    Start with `hindsight demo` — it needs no credentials at all.
    """
    _init_output()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config.ensure_dirs()


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------


@main.command()
@click.option("--tz", default=0.0, type=float,
              help="Hours offset from UTC for publish-time features.")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Where to write the HTML report.")
@click.option("--open/--no-open", "open_browser", default=True,
              help="Open the report when it is written.")
def demo(tz: float, out: Path | None, open_browser: bool) -> None:
    """
    Run the whole pipeline on the bundled real catalog. No setup required.

    Uses a genuine 949-video channel snapshot shipped with the package:
    real publish times, view counts, durations and tags, with identifying
    fields replaced. Everything downstream of ingestion is the same code that
    runs on live data.

    Pass --tz 5.5 to reproduce the figures quoted in the README; the bundled
    channel publishes on UTC+5:30 and hours are reported in the timezone you
    give.
    """
    click.secho("Hindsight demo — bundled catalog, no credentials needed",
                bold=True)
    try:
        videos, channel = demo_mod.load_demo()
    except FileNotFoundError as exc:
        _die(str(exc))

    public = [v for v in videos if v.get("privacy") == "public"]
    click.echo(f"Loaded {len(public):,} public videos from the bundled dataset.")
    click.secho("Scoring cohorts and testing features…", dim=True)

    result = analyze_mod.analyze(public, channel, timezone_offset_h=tz)
    _print_findings(result)

    plan = None
    try:
        plan = exp_mod.design_experiment(result)
        _print_plan(plan)
    except ValueError as exc:
        _warn(str(exc))

    target = out or (config.OUT_DIR / "hindsight-demo.html")
    report_mod.write_report(result, target, plan=plan, demo_mode=True)
    _hdr("Report")
    _ok(f"written to {target}")
    if open_browser:
        webbrowser.open(target.resolve().as_uri())


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


@main.group()
def auth() -> None:
    """Inspect or create YouTube credentials."""


@auth.command("list")
@click.option("--dir", "search_dir", type=click.Path(path_type=Path), default=None,
              help="Directory to scan for *_token.json files.")
def auth_list(search_dir: Path | None) -> None:
    """Find reusable OAuth token files on this machine."""
    tokens = config.discover_tokens(search_dir)
    if not tokens:
        _warn(
            "No token files found. Either run `hindsight auth login`, or use "
            "API-key mode: `hindsight ingest --api-key KEY --channel @handle`."
        )
        return
    _hdr(f"Found {len(tokens)} token file(s)")
    for t in tokens:
        _kv(t.slug, str(t.path))


@auth.command("login")
@click.option("--client-secrets", type=click.Path(path_type=Path), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def auth_login(client_secrets: Path | None, out: Path | None) -> None:
    """Run the OAuth flow and save a new token file."""
    from .auth import login

    try:
        path = login(client_secrets, out)
    except AuthError as exc:
        _die(str(exc))
    _ok(f"token written to {path}")


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


@main.command()
@click.option("--token", type=click.Path(path_type=Path), default=None,
              help="OAuth token file. Auto-discovered if omitted.")
@click.option("--api-key", envvar="YOUTUBE_API_KEY", default=None,
              help="API key for reading a public channel (no OAuth needed).")
@click.option("--channel", default=None,
              help="Channel id, @handle or URL. Required with --api-key.")
@click.option("--limit", type=int, default=None,
              help="Only fetch the newest N videos (for a quick check).")
def ingest(token: Path | None, api_key: str | None, channel: str | None,
           limit: int | None) -> None:
    """
    Pull a channel's catalog into the local database.

    Two modes:

      OAuth     hindsight ingest --token path/to/token.json
      API key   hindsight ingest --api-key KEY --channel @somehandle

    Re-running is cheap and builds the stat history that `hindsight verdict`
    reads, so a daily run is a reasonable habit.
    """
    try:
        if api_key:
            if not channel:
                _die("--channel is required with --api-key (id, @handle or URL).")
            session = open_public_session(api_key, channel)
        else:
            if token is None:
                found = config.discover_tokens()
                if not found:
                    _die(
                        "No OAuth token found and no --api-key given.\n"
                        "  Try:  hindsight demo                    (no setup)\n"
                        "        hindsight ingest --api-key KEY --channel @handle\n"
                        "        hindsight auth login              (full OAuth)"
                    )
                if len(found) > 1 and channel is None:
                    _hdr("Multiple tokens found — pass --token to choose")
                    for t in found:
                        _kv(t.slug, str(t.path))
                    sys.exit(1)
                ref = next((t for t in found if t.slug == channel), found[0])
                token = ref.path
            session = open_session(Path(token))
    except AuthError as exc:
        _die(str(exc))

    click.echo(f"Ingesting {session.slug}…")
    with click.progressbar(length=100, label="  fetching") as bar:
        state = {"done": 0}

        def progress(done: int, total: int | None) -> None:
            pct = int(done / total * 100) if total else 0
            bar.update(max(0, min(100, pct) - state["done"]))
            state["done"] = min(100, pct)

        try:
            result = ingest_channel(session, limit=limit, progress=progress)
        except IngestError as exc:
            click.echo()
            _die(str(exc))

    _hdr("Ingested")
    _kv("channel", f"{result.channel_title} ({result.channel_id})")
    _kv("videos", f"{result.videos_seen:,}")
    _kv("api calls", str(result.api_calls))
    _kv("snapshot", result.observed_utc)
    for w in result.warnings:
        _warn(w)
    if session.is_public_mode:
        click.echo()
        click.secho("  Public mode: only public videos and public statistics "
                    "were read.", dim=True)
    if not session.is_public_mode and not session.has_analytics:
        click.echo()
        click.secho(f"  Retention/CTR unavailable — {session.analytics_error}.",
                    dim=True)
        click.secho("  Everything in this tool works without it; see "
                    "docs/SETUP.md to enable.", dim=True)

    _ok("now run: hindsight analyze")


# --------------------------------------------------------------------------
# analyze / report
# --------------------------------------------------------------------------


def _analysis_options(f):
    f = click.option("--channel", default=None, help="Channel slug or id.")(f)
    f = click.option("--tz", default=0.0, type=float,
                     help="Hours offset from UTC for publish-time features.")(f)
    f = click.option("--cohort", default=config.COHORT_HALF_WIDTH, type=int,
                     help="Cohort half-width in videos.")(f)
    f = click.option("--min-age", default=config.MIN_AGE_DAYS, type=int,
                     help="Exclude videos younger than this many days.")(f)
    f = click.option("--min-bucket", default=config.MIN_BUCKET_N, type=int,
                     help="Minimum videos per bucket before testing it.")(f)
    return f


@main.command()
@_analysis_options
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def analyze(channel, tz, cohort, min_age, min_bucket, as_json) -> None:
    """Score the catalog and report what actually moved views."""
    videos, ch = _load_catalog(channel)
    result = analyze_mod.analyze(
        videos, ch, half_width=cohort, min_age_days=min_age,
        min_bucket_n=min_bucket, timezone_offset_h=tz,
    )
    if as_json:
        click.echo(report_mod.to_json(result))
        return
    _print_findings(result)
    click.echo()
    _ok("write the full report with: hindsight report")


@main.command()
@_analysis_options
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option("--open/--no-open", "open_browser", default=True)
@click.option("--with-experiment/--no-experiment", default=True,
              help="Include a designed next experiment in the report.")
def report(channel, tz, cohort, min_age, min_bucket, out, open_browser,
           with_experiment) -> None:
    """Render the full analysis as a standalone HTML page."""
    videos, ch = _load_catalog(channel)
    result = analyze_mod.analyze(
        videos, ch, half_width=cohort, min_age_days=min_age,
        min_bucket_n=min_bucket, timezone_offset_h=tz,
    )

    plan = None
    if with_experiment:
        try:
            plan = exp_mod.design_experiment(result)
        except ValueError:
            plan = None

    slug = ch.get("slug") or ch["channel_id"]
    target = out or (config.OUT_DIR / f"hindsight-{slug}.html")
    report_mod.write_report(result, target, plan=plan)
    _ok(f"report written to {target}")
    if open_browser:
        webbrowser.open(target.resolve().as_uri())


# --------------------------------------------------------------------------
# design / verdict / experiments
# --------------------------------------------------------------------------


@main.command()
@_analysis_options
@click.option("--feature", default=None, help="Force a specific lever.")
@click.option("--min-effect", type=float, default=None,
              help="Smallest effect to detect, in percentile points.")
@click.option("--within-days", type=float, default=config.DEFAULT_BUDGET_DAYS,
              help="Time budget the test should read out within.")
@click.option("--save/--no-save", default=True, help="Record it for `verdict`.")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Also write the plan as JSON.")
def design(channel, tz, cohort, min_age, min_bucket, feature, min_effect,
           within_days, save, out) -> None:
    """Design the next A/B test from what the analysis found."""
    videos, ch = _load_catalog(channel)
    result = analyze_mod.analyze(
        videos, ch, half_width=cohort, min_age_days=min_age,
        min_bucket_n=min_bucket, timezone_offset_h=tz,
    )
    try:
        plan = exp_mod.design_experiment(
            result, feature_key=feature, min_effect=min_effect,
            within_days=within_days,
        )
    except ValueError as exc:
        _die(str(exc))

    _print_plan(plan)

    if save:
        with db.session() as conn:
            exp_mod.save_plan(conn, plan)
        _ok(f"saved as {plan.experiment_id}")

    target = out or (config.OUT_DIR / f"{plan.experiment_id}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plan.to_json(), encoding="utf-8")
    _ok(f"manifest written to {target}")


@main.command()
@click.argument("experiment_id")
@_analysis_options
def verdict(experiment_id, channel, tz, cohort, min_age, min_bucket) -> None:
    """Read out a running experiment against freshly ingested data."""
    with db.session() as conn:
        row = db.get_experiment(conn, experiment_id)
        if row is None:
            _die(f"No experiment {experiment_id!r}. See `hindsight experiments`.")
        record = dict(row)

    videos, ch = _load_catalog(channel or record["channel_id"])
    result = analyze_mod.analyze(
        videos, ch, half_width=cohort, min_age_days=min_age,
        min_bucket_n=min_bucket, timezone_offset_h=tz,
    )
    v = exp_mod.read_verdict(result, record)

    _hdr(f"Verdict — {v.experiment_id}")
    _kv("status", v.status)
    click.echo()
    click.echo(f"  {v.headline}")
    click.echo()
    for arm in v.arms:
        click.echo(
            f"  {arm.name:<12} {arm.value:<28} n={arm.n:<4} "
            f"mean pct={arm.mean_percentile:5.1f}  median {int(arm.median_views):,} views"
        )
    click.echo()
    click.echo(f"  {v.recommendation}")

    if v.status == "conclusive":
        with db.session() as conn:
            db.conclude_experiment(conn, experiment_id, v.to_dict())
        _ok("experiment marked concluded")


@main.command("experiments")
@click.option("--channel", default=None)
def list_experiments(channel: str | None) -> None:
    """List designed and concluded experiments."""
    with db.session() as conn:
        cid = None
        if channel:
            try:
                cid = db.resolve_channel(conn, channel)["channel_id"]
            except ValueError as exc:
                _die(str(exc))
        rows = db.list_experiments(conn, cid)

    if not rows:
        click.echo("No experiments yet. Create one with `hindsight design`.")
        return
    _hdr(f"{len(rows)} experiment(s)")
    for r in rows:
        click.echo(
            f"  {r['experiment_id']:<34} {r['status']:<11} "
            f"{r['feature']:<20} created {r['created_utc'][:10]}"
        )


# --------------------------------------------------------------------------
# export-demo (maintenance)
# --------------------------------------------------------------------------


@main.command("export-demo", hidden=True)
@click.option("--channel", default=None)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def export_demo(channel: str | None, out: Path | None) -> None:
    """Regenerate the bundled demo dataset from a live catalog."""
    with db.session() as conn:
        try:
            ch = dict(db.resolve_channel(conn, channel))
        except ValueError as exc:
            _die(str(exc))
        videos = db.load_videos_with_latest_stats(
            conn, ch["channel_id"], include_private=True
        )

    payload = demo_mod.anonymize(videos, ch)
    path = demo_mod.write_dataset(payload, out)
    _ok(f"wrote {len(payload['videos']):,} anonymised videos to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
