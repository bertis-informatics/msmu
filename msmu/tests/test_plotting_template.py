import plotly.io as pio

from msmu._plotting._template import add_msmu_pastel_template, add_msmu_template, set_default_template


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
