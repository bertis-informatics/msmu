"""Statistics-oriented plotting functions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._ptypes import PlotScatter
from ._utils import apply_msmu_template


def plot_volcano(
    results: pd.DataFrame,
    *,
    ctrl: str | None,
    expr: str | None = None,
    log2fc_threshold: float,
    pval_threshold: float = 0.05,
    label_top: int | None = None,
) -> go.Figure:
    """Plot differential-expression results using raw p-values.

    Parameters:
        results: DataFrame with `features`, `log2fc`, and `p_value` columns. See [`run_de`][msmu.tl.run_de] for differential-expression analysis.
        ctrl: Control label shown in the title and downregulated annotation; does not select/filter rows.
        expr: Experimental label shown in the title and upregulated annotation; does not select/filter rows.
        log2fc_threshold: Absolute log2 fold-change cutoff. UP/DOWN require strictly greater/smaller fold changes.
        pval_threshold: Raw p-value cutoff (strictly less than); `q_value` is not used.
        label_top: Maximum number of feature labels in each significant direction, ranked by fold change; `None` disables labels.

    Returns:
        Plotly `Figure`; the input DataFrame is copied and remains unchanged.

    Notes:
        Supply finite log2 fold changes and positive p-values for finite plotted coordinates. The function does not perform multiple-testing correction.

    Examples:
        ```python
        import msmu as mm
        import pandas as pd
        results = pd.DataFrame({"features": ["P1", "P2"], "log2fc": [2., -2.], "p_value": [.001, .01]})
        fig = mm.pl.plot_volcano(results, ctrl="control", expr="treated", log2fc_threshold=1.)
        fig.show()
        ```
    """
    df = results.copy()
    df["logp"] = -np.log10(df["p_value"])
    up_cond = df["log2fc"] > log2fc_threshold
    down_cond = df["log2fc"] < -log2fc_threshold
    sig_cond = df["p_value"] < pval_threshold

    df.loc[:, "de"] = "nonDE"
    df.loc[up_cond & sig_cond, "de"] = "UP"
    df.loc[down_cond & sig_cond, "de"] = "DOWN"

    up_count = len(df.loc[df["de"] == "UP"])
    down_count = len(df.loc[df["de"] == "DOWN"])

    plot = PlotScatter(
        data=df,
        x="log2fc",
        y="logp",
        name="de",
        meta="features",
        text="p_value",
        hovertemplate="<b>%{meta}</b><br>Log<sub>2</sub>FC: %{x}<br>p-value: %{text}",
    )

    fig = plot.figure(mode="markers")

    fig.update_xaxes(title="log<sub>2</sub>FC")
    fig.update_yaxes(title="-log<sub>10</sub>p")

    fig.update_traces(marker=dict(color="#E15759"), selector=dict(name="UP"))
    fig.update_traces(marker=dict(color="#4E79A7"), selector=dict(name="DOWN"))
    fig.update_traces(marker=dict(color="#BAB0AC"), selector=dict(name="nonDE"))

    fig.update_traces(marker=dict(size=4))

    fig.update_layout(
        title=f"{ctrl} vs. {expr}",
        width=600,
        height=500,
    )

    fig.add_hline(
        y=-np.log10(pval_threshold),
        line=dict(color="grey", dash="dot", width=1),
    )
    fig.add_vline(
        x=log2fc_threshold,
        line=dict(color="grey", dash="dot", width=1),
    )
    fig.add_vline(
        x=-log2fc_threshold,
        line=dict(color="grey", dash="dot", width=1),
    )

    fig.add_annotation(
        x=float(df["log2fc"].min()),
        y=float(df["logp"].min()),
        text=f"{ctrl} ({down_count})",
        showarrow=False,
    )
    fig.add_annotation(
        x=float(df["log2fc"].max()),
        y=float(df["logp"].min()),
        text=f"{expr} ({up_count})",
        showarrow=False,
    )

    if label_top is not None:
        up_top = df.loc[df["de"] == "UP", :].sort_values("log2fc").tail(label_top)
        down_top = df.loc[df["de"] == "DOWN", :].sort_values("log2fc").head(label_top)

        concated_tops = pd.concat([up_top, down_top])

        for _, row in concated_tops.iterrows():
            fig.add_annotation(
                x=row["log2fc"],
                y=row["logp"],
                text=row["features"],
                arrowhead=0,
                arrowwidth=1,
            )

    # plot_volcano takes a results frame (no PlotContext), so it does not pass through
    # finalize_figure; apply the msmu house style explicitly.
    return apply_msmu_template(fig)
