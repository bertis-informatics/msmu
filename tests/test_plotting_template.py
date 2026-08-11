import subprocess
import sys
from pathlib import Path

import plotly.io as pio
import pytest

from msmu._plotting._template import (
    add_msmu_pastel_template,
    add_msmu_template,
    set_templates,
    set_default_template,
)


@pytest.fixture(autouse=True)
def restore_plotly_default_template():
    original_default = pio.templates.default
    original_msmu_templates = {
        name: pio.templates[name]
        for name in ("msmu", "msmu_pastel")
        if name in pio.templates
    }
    yield
    pio.templates.default = "plotly"
    for name in ("msmu", "msmu_pastel"):
        if name in original_msmu_templates:
            pio.templates[name] = original_msmu_templates[name]
        elif name in pio.templates:
            del pio.templates[name]
    pio.templates.default = original_default


def test_import_msmu_preserves_plotly_template_state():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import plotly.io as pio

pio.templates.default = "plotly_dark"
templates_before = set(pio.templates)
default_before = pio.templates.default

import msmu  # noqa: F401

assert pio.templates.default == default_before, (default_before, pio.templates.default)
assert set(pio.templates) == templates_before
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_plotting_exports_set_templates():
    from msmu import pl

    assert "set_templates" in pl.__all__
    assert pl.set_templates is set_templates


def test_add_msmu_template_registers():
    add_msmu_template()
    assert "msmu" in pio.templates
    assert "colorway" in pio.templates["msmu"].layout


def test_add_msmu_pastel_template_overrides_colorway():
    add_msmu_template()
    add_msmu_pastel_template()
    assert "msmu_pastel" in pio.templates
    assert pio.templates["msmu_pastel"].layout.colorway[0] != pio.templates["msmu"].layout.colorway[0]


def test_set_default_template():
    add_msmu_template()
    set_default_template("msmu")
    assert pio.templates.default == "msmu"


def test_set_templates_registers_and_sets_default_template():
    set_templates()
    assert "msmu" in pio.templates
    assert "msmu_pastel" in pio.templates
    assert pio.templates.default == "msmu"
