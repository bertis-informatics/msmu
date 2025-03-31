import plotly.graph_objects as go
import plotly.io as pio


DEFAULT_TEMPLATE = "msmu"


def _set_templates():
    _add_msmu_template()
    _add_msmu_pastel_template()
    _set_default_template(DEFAULT_TEMPLATE)


def _add_msmu_template():
    pio.templates["msmu"] = go.layout.Template(
        layout={
            "autotypenumbers": "strict",
            "colorway": [
                "#4E79A7",
                "#F28E2B",
                "#E15759",
                "#76B7B2",
                "#59A14F",
                "#EDC948",
                "#B07AA1",
                "#FF9DA7",
                "#9C755F",
                "#BAB0AC",
            ],
            "font": {"color": "rgb(36,36,36)"},
            "hoverlabel": {"align": "left"},
            "hovermode": "closest",
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
            "mapbox": {"style": "light"},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "title": {"x": 0, "xanchor": "left", "xref": "container", "pad": {"l": 16}},
            "xaxis": {
                "automargin": True,
                "gridcolor": None,
                "linecolor": "rgb(36,36,36)",
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
                "tickcolor": "rgb(36,36,36)",
                "title": {"standoff": 10},
                "zeroline": False,
                "zerolinecolor": "rgb(36,36,36)",
                "zerolinewidth": 1,
            },
            "yaxis": {
                "automargin": True,
                "gridcolor": None,
                "linecolor": "rgb(36,36,36)",
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
                "tickcolor": "rgb(36,36,36)",
                "title": {"standoff": 10},
                "zeroline": False,
                "zerolinecolor": "rgb(36,36,36)",
                "zerolinewidth": 1,
            },
        }
    )


def _add_msmu_pastel_template():
    pio.templates["msmu_pastel"] = pio.templates["msmu"]
    pio.templates["msmu_pastel"].layout.colorway = [
        "#A6CEE3",
        "#FDBF6F",
        "#FB9A99",
        "#B2DF8A",
        "#CAB2D6",
        "#FFFF99",
        "#FFCC99",
        "#CCEBC5",
        "#F0E442",
        "#D9D9D9",
    ]


def _set_default_template(template_name: str):
    pio.templates.default = template_name
