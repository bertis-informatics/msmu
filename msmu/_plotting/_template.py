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
            "annotationdefaults": {"arrowhead": 0, "arrowwidth": 1},
            "autotypenumbers": "strict",
            "coloraxis": {"colorbar": {"outlinewidth": 1, "tickcolor": "rgb(36,36,36)", "ticks": "outside"}},
            "colorscale": {
                "diverging": [
                    [0, "#8e0152"],
                    [0.1, "#c51b7d"],
                    [0.2, "#de77ae"],
                    [0.3, "#f1b6da"],
                    [0.4, "#fde0ef"],
                    [0.5, "#f7f7f7"],
                    [0.6, "#e6f5d0"],
                    [0.7, "#b8e186"],
                    [0.8, "#7fbc41"],
                    [0.9, "#4d9221"],
                    [1, "#276419"],
                ],
                "sequential": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
                "sequentialminus": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
            },
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
            "title": {"x": 0.05},
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
