"""
Tests for CLI output encoding and command wiring.

The encoding tests exist because of a real failure: the CLI printed
box-drawing characters and status glyphs that Windows cannot encode under
cp1252, which is the encoding Python selects whenever stdout is redirected.
Ingest and report both completed their work and then died printing their own
success message. A crash after the work is done is still a crash, and piping
output to a file is an ordinary thing to do.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from hindsight import analyze as an, cli, experiment as ex
from tests.test_analyze import CHANNEL, build


@pytest.fixture(scope="module")
def result():
    """A small real analysis, shared by the encoding tests."""
    return an.analyze(build(n=200), CHANNEL, iters=200)


class TestGlyphTables:
    def test_both_tables_define_the_same_keys(self):
        """A key present in only one table crashes exactly where it matters."""
        assert set(cli._GLYPHS_UNICODE) == set(cli._GLYPHS_ASCII)

    def test_ascii_fallbacks_are_actually_ascii(self):
        for key, glyph in cli._GLYPHS_ASCII.items():
            glyph.encode("ascii")   # must not raise

    def test_ascii_fallbacks_survive_cp1252(self):
        for glyph in cli._GLYPHS_ASCII.values():
            glyph.encode("cp1252")

    def test_every_glyph_used_is_defined(self):
        """Guards against a G['typo'] reaching a user as a KeyError."""
        import re
        from pathlib import Path

        source = Path(cli.__file__).read_text(encoding="utf-8")
        used = set(re.findall(r"G\[[\"'](\w+)[\"']\]", source))
        assert used, "expected the CLI to use the glyph table"
        assert used <= set(cli._GLYPHS_ASCII)

    def test_init_output_is_safe_to_call(self):
        cli._init_output()
        assert set(cli.G) == set(cli._GLYPHS_ASCII)


class TestOutputIsEncodable:
    """
    Render real output with the ASCII fallbacks and confirm every byte
    survives cp1252. This is the regression test for the original crash.
    """

    @pytest.fixture
    def ascii_glyphs(self, monkeypatch):
        monkeypatch.setitem(cli.__dict__, "G", dict(cli._GLYPHS_ASCII))

    def test_findings_output_encodes(self, ascii_glyphs, result, capsys):
        cli._print_findings(result)
        captured = capsys.readouterr().out
        assert captured
        captured.encode("cp1252")

    def test_plan_output_encodes(self, ascii_glyphs, result, capsys):
        plan = ex.design_experiment(result)
        cli._print_plan(plan)
        captured = capsys.readouterr().out
        assert captured
        captured.encode("cp1252")

    def test_helpers_encode(self, ascii_glyphs, capsys):
        cli._hdr("A heading")
        cli._ok("it worked")
        cli._warn("careful")
        cli._kv("key", "value")
        capsys.readouterr().out.encode("cp1252")


class TestCommandWiring:
    def test_help_lists_the_main_commands(self):
        out = CliRunner().invoke(cli.main, ["--help"]).output
        for command in ("demo", "ingest", "analyze", "report", "design", "verdict"):
            assert command in out

    def test_demo_is_documented_as_needing_no_setup(self):
        out = CliRunner().invoke(cli.main, ["demo", "--help"]).output
        assert "No setup required" in out or "no setup" in out.lower()

    @pytest.mark.parametrize("command", [
        "demo", "ingest", "analyze", "report", "design", "verdict", "experiments",
    ])
    def test_every_command_has_help(self, command):
        res = CliRunner().invoke(cli.main, [command, "--help"])
        assert res.exit_code == 0
        assert res.output.strip()

    def test_ingest_requires_channel_with_api_key(self):
        res = CliRunner().invoke(cli.main, ["ingest", "--api-key", "K"])
        assert res.exit_code != 0
        assert "--channel is required" in res.output

    def test_version_flag(self):
        from hindsight import __version__
        res = CliRunner().invoke(cli.main, ["--version"])
        assert __version__ in res.output
