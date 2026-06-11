import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_confidence_distribution(probs, true, pred):
    correct_mask = pred == true
    conf_correct = probs[correct_mask].max(axis=1)
    conf_incorrect = probs[~correct_mask].max(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=conf_correct,
        nbinsx=50,
        name=f"Correct ({correct_mask.sum()})",
        marker_color="green",
        opacity=0.6,
        histnorm="probability density"
    ))
    fig.add_trace(go.Histogram(
        x=conf_incorrect,
        nbinsx=50,
        name=f"Incorrect ({(~correct_mask).sum()})",
        marker_color="red",
        opacity=0.6,
        histnorm="probability density"
    ))

    fig.update_layout(
        title=dict(text="Confidence Distribution: Correct vs Misclassified", x=0.5),
        xaxis=dict(title="Prediction Confidence (max softmax)"),
        yaxis=dict(title="Density"),
        barmode="overlay",
        legend=dict(orientation="h", y=1.05),
        width=800, height=450,
        margin=dict(l=60, r=40, t=60, b=80)
    )

    return fig


def plot_accuracy_at_threshold(probs, true, pred):
    correct_mask = pred == true
    confidences = probs.max(axis=1)

    thresholds = np.linspace(0, 1, 101)
    accuracies = []
    coverages = []

    for thresh in thresholds:
        mask = confidences >= thresh
        if mask.sum() == 0:
            accuracies.append(0)
            coverages.append(0)
        else:
            accuracies.append((pred[mask] == true[mask]).mean())
            coverages.append(mask.mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=accuracies,
        mode="lines",
        name="Accuracy",
        line=dict(color="steelblue", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=coverages,
        mode="lines",
        name="Coverage",
        line=dict(color="darkorange", width=2, dash="dash"),
        yaxis="y2"
    ))

    fig.update_layout(
        title=dict(text="Accuracy / Coverage vs Confidence Threshold", x=0.5),
        xaxis=dict(title="Confidence Threshold"),
        yaxis=dict(title="Accuracy", range=[0, 1.05]),
        yaxis2=dict(title="Coverage", overlaying="y", side="right", range=[0, 1.05]),
        legend=dict(orientation="h", y=1.05),
        width=800, height=450,
        margin=dict(l=60, r=60, t=60, b=80)
    )

    return fig
