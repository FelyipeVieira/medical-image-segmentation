"""Training and metric plots."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def training_curves(history: pd.DataFrame) -> go.Figure:
    """Plot long-form epoch metrics as interactive curves."""
    return px.line(history, x="epoch", y="value", color="metric")
