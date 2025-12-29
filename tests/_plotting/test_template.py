import plotly.io as pio
from msmu._plotting._template import set_templates


def test_set_templates():
    # Save original default
    original_default = pio.templates.default

    try:
        set_templates()
        assert "msmu" in pio.templates
        assert "msmu_pastel" in pio.templates
        assert pio.templates.default == "msmu"
    finally:
        # Restore original default to avoid side effects
        pio.templates.default = original_default
