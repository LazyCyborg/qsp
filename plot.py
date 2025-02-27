import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_similarity_matrix(similarity_matrix, segments):
    """Create a heatmap of the similarity matrix"""
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x="Segment", y="Segment", color="Similarity"),
        x=[f"Seg {i + 1}" for i in range(len(segments))],
        y=[f"Seg {i + 1}" for i in range(len(segments))],
        color_continuous_scale="RdBu_r",
        zmin=0, zmax=1
    )

    fig.update_layout(
        title="Segment Similarity Matrix",
        width=700,
        height=600,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Add segment text as hover information
    hover_text = [[
                      f"Segment {i + 1} vs Segment {j + 1}<br>Similarity: {similarity_matrix[i][j]:.2f}<br>Text 1: {segments[i][:50]}...<br>Text 2: {segments[j][:50]}..."
                      for j in range(len(segments))] for i in range(len(segments))]

    fig.update_traces(text=hover_text, hoverinfo="text")

    return fig


def plot_sentiment_timeline(df):
    """Create a timeline plot of sentiment scores"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add VADER compound sentiment
    fig.add_trace(
        go.Scatter(
            x=df["segment_id"],
            y=df["vader_compound"],
            mode="lines+markers",
            name="VADER Compound",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8)
        ),
        secondary_y=False
    )

    # Add colored background for sentiment regions
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=len(df) - 0.5,
        y0=0.05, y1=1,
        line=dict(width=0),
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below"
    )

    fig.add_shape(
        type="rect",
        x0=-0.5, x1=len(df) - 0.5,
        y0=-0.05, y1=0.05,
        line=dict(width=0),
        fillcolor="rgba(255, 255, 0, 0.1)",
        layer="below"
    )

    fig.add_shape(
        type="rect",
        x0=-0.5, x1=len(df) - 0.5,
        y0=-1, y1=-0.05,
        line=dict(width=0),
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below"
    )

    # Create a categorical scatter for RoBERTa sentiment
    roberta_colors = {
        "positive": "#2ca02c",  # Green
        "neutral": "#d9d9d9",  # Gray
        "negative": "#d62728"  # Red
    }

    roberta_y_pos = {"positive": 1, "neutral": 0, "negative": -1}

    fig.add_trace(
        go.Scatter(
            x=df["segment_id"],
            y=[roberta_y_pos[s] for s in df["roberta_sentiment"]],
            mode="markers",
            marker=dict(
                size=12,
                color=[roberta_colors[s] for s in df["roberta_sentiment"]],
                symbol="square",
                line=dict(width=1, color="black")
            ),
            name="RoBERTa Sentiment",
            text=df["roberta_sentiment"] + "<br>Confidence: " + df["roberta_confidence"].round(2).astype(str)
        ),
        secondary_y=True
    )

    # Update layout
    fig.update_layout(
        title="Sentiment Analysis Timeline",
        xaxis_title="Segment ID",
        yaxis_title="VADER Compound Score",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        width=900,
        height=500,
        hovermode="closest"
    )

    # Update y-axis for RoBERTa
    fig.update_yaxes(
        title_text="RoBERTa Sentiment",
        secondary_y=True,
        tickvals=[-1, 0, 1],
        ticktext=["Negative", "Neutral", "Positive"],
        range=[-1.5, 1.5]
    )

    # Update y-axis for VADER
    fig.update_yaxes(
        title_text="VADER Compound Score",
        secondary_y=False,
        range=[-1.1, 1.1]
    )

    return fig


def plot_similarity_scores(df):
    """Create a visualization of similarity scores"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add raw similarity scores
    fig.add_trace(
        go.Scatter(
            x=df["segment_id"],
            y=df["similarity_raw"],
            mode="lines+markers",
            name="Raw Similarity",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=8)
        ),
        secondary_y=False
    )

    # Add z-scores
    fig.add_trace(
        go.Scatter(
            x=df["segment_id"],
            y=df["similarity_z_score"],
            mode="lines+markers",
            name="Z-Score",
            line=dict(color="#2ca02c", width=2, dash="dash"),
            marker=dict(size=8)
        ),
        secondary_y=True
    )

    # Add reference lines for z-scores
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(df) - 0.5,
        y0=1.96, y1=1.96,
        line=dict(color="red", width=1, dash="dot"),
        secondary_y=True
    )

    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(df) - 0.5,
        y0=-1.96, y1=-1.96,
        line=dict(color="red", width=1, dash="dot"),
        secondary_y=True
    )

    # Update layout
    fig.update_layout(
        title="Segment Similarity Analysis",
        xaxis_title="Segment ID",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        width=900,
        height=500,
        hovermode="closest"
    )

    # Update y-axes
    fig.update_yaxes(
        title_text="Raw Similarity Score",
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Z-Score",
        secondary_y=True,
        range=[-3.5, 3.5]
    )

    # Add annotation for z-score significance
    fig.add_annotation(
        x=len(df) - 1,
        y=2.2,
        text="Significant Difference (95%)",
        showarrow=False,
        font=dict(color="red"),
        xanchor="right",
        secondary_y=True
    )

    return fig