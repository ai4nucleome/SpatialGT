"""Plotly-based spatial scatter plots with selection support."""

from __future__ import annotations
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def spatial_scatter(
    coords: np.ndarray,
    color_values: Optional[Sequence] = None,
    color_label: str = "Cell Type",
    selected_indices: Optional[List[int]] = None,
    highlight_color: str = "red",
    point_size: int = 4,
    title: str = "",
    height: int = 700,
) -> go.Figure:
    """
    Create an interactive spatial scatter plot.
    coords: (N, 2) array.
    """
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    df["index"] = range(len(df))

    if color_values is not None:
        df[color_label] = list(color_values)
        fig = px.scatter(
            df, x="x", y="y", color=color_label,
            hover_data=["index"],
            title=title,
            height=height,
        )
    else:
        fig = px.scatter(
            df, x="x", y="y",
            hover_data=["index"],
            title=title,
            height=height,
        )

    fig.update_traces(marker=dict(size=point_size))

    if selected_indices:
        sel_df = df.iloc[selected_indices]
        fig.add_trace(go.Scatter(
            x=sel_df["x"], y=sel_df["y"],
            mode="markers",
            marker=dict(size=point_size + 3, color=highlight_color, symbol="circle-open", line=dict(width=2)),
            name="Selected",
            hoverinfo="skip",
        ))

    fig.update_layout(
        dragmode="lasso",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_yaxes(autorange="reversed")

    return fig


def gene_expression_map(
    coords: np.ndarray,
    expression: np.ndarray,
    gene_name: str,
    point_size: int = 4,
    height: int = 600,
    colorscale: str = "Viridis",
) -> go.Figure:
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "expression": expression,
    })
    fig = px.scatter(
        df, x="x", y="y", color="expression",
        color_continuous_scale=colorscale,
        title=f"{gene_name} Expression",
        height=height,
    )
    fig.update_traces(marker=dict(size=point_size))
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_yaxes(autorange="reversed")
    return fig
