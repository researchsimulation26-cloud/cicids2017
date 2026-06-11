import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def plot_confusion_matrix(true, pred, class_names):
    cm = confusion_matrix(true, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f"{cm[i,j]}<br>({cm_norm[i,j]:.1%})",
                    font=dict(size=9, color="white" if cm_norm[i,j] > 0.5 else "black"),
                    showarrow=False
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        colorscale="Blues",
        texttemplate="",
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{customdata}<br>Rate: %{z:.1%}<extra></extra>",
        customdata=cm,
        colorbar=dict(title="Rate")
    ))

    fig.update_layout(
        title=dict(text="Confusion Matrix — CyberSAGE", x=0.5),
        xaxis=dict(title="Predicted Label", tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(title="True Label", tickfont=dict(size=9)),
        width=700, height=600,
        margin=dict(l=80, r=40, t=60, b=120)
    )

    return fig


def plot_per_class_metrics(true, pred, class_names):
    p, r, f, s = precision_recall_fscore_support(
        true, pred, labels=range(len(class_names)), zero_division=0
    )

    fig = go.Figure()
    x = np.arange(len(class_names))

    fig.add_trace(go.Bar(name="Precision", x=class_names, y=p, marker_color="steelblue"))
    fig.add_trace(go.Bar(name="Recall", x=class_names, y=r, marker_color="darkorange"))
    fig.add_trace(go.Bar(name="F1 Score", x=class_names, y=f, marker_color="seagreen"))

    fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                  annotation_text="0.7 threshold")

    fig.update_layout(
        title=dict(text="Per-Class Precision / Recall / F1", x=0.5),
        xaxis=dict(title="Class", tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(title="Score", range=[0, 1.1]),
        barmode="group",
        legend=dict(orientation="h", y=1.05),
        width=900, height=500,
        margin=dict(l=60, r=40, t=60, b=120)
    )

    return fig


def plot_support_vs_f1(true, pred, class_names):
    _, r, f, s = precision_recall_fscore_support(
        true, pred, labels=range(len(class_names)), zero_division=0
    )

    colors = []
    for f_i in f:
        if f_i >= 0.7:
            colors.append("green")
        elif f_i >= 0.4:
            colors.append("orange")
        else:
            colors.append("red")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s, y=f, mode="markers+text",
        marker=dict(size=12, color=colors, line=dict(width=1, color="black")),
        text=class_names,
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate="Class: %{text}<br>Support: %{x}<br>F1: %{y:.3f}<extra></extra>"
    ))

    fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                  annotation_text="F1 = 0.7")

    fig.update_layout(
        title=dict(text="Class Support vs F1 Score", x=0.5),
        xaxis=dict(title="Support (log scale)", type="log"),
        yaxis=dict(title="F1 Score", range=[-0.05, 1.05]),
        width=800, height=500,
        margin=dict(l=60, r=40, t=60, b=80)
    )

    return fig
