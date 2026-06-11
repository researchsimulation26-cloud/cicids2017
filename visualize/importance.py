import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from config import CLASS_NAMES, FEATURE_COLS


def plot_shap_global_importance(shap_values, feature_names, class_names, top_k=20):
    global_mean = np.abs(shap_values).mean(axis=(0, 1))
    sorted_idx = np.argsort(global_mean)[::-1][:top_k]

    fig = go.Figure(go.Bar(
        x=global_mean[sorted_idx][::-1],
        y=[feature_names[i] for i in sorted_idx][::-1],
        orientation="h",
        marker=dict(color="steelblue"),
        hovertemplate="Feature: %{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Global Feature Importance — SHAP (Top 20)", x=0.5),
        xaxis=dict(title="Mean |SHAP Value|"),
        yaxis=dict(title="", autorange="reversed"),
        height=600, width=800,
        margin=dict(l=150, r=40, t=60, b=80)
    )

    return fig


def plot_shap_per_class_heatmap(shap_values, feature_names, class_names, top_k=15):
    global_mean = np.abs(shap_values).mean(axis=(0, 1))
    top_features = np.argsort(global_mean)[::-1][:top_k]

    shap_per_class = np.array([np.abs(sv).mean(axis=0) for sv in shap_values])
    z_data = shap_per_class[:, top_features]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[feature_names[i] for i in top_features],
        y=class_names,
        colorscale="YlOrRd",
        hovertemplate="Class: %{y}<br>Feature: %{x}<br>Mean |SHAP|: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Mean |SHAP|")
    ))

    fig.update_layout(
        title=dict(text="Per-Class Feature Importance — SHAP", x=0.5),
        xaxis=dict(title="", tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(title="Attack Class", tickfont=dict(size=10)),
        height=500, width=900,
        margin=dict(l=80, r=40, t=60, b=120)
    )

    return fig


def plot_shap_waterfall(shap_values, sample_idx, feature_names, class_names, top_k=12):
    figs = []
    for cls_idx, cls_name in enumerate(class_names):
        sv = shap_values[cls_idx][sample_idx]
        sorted_idx = np.argsort(np.abs(sv))[::-1][:top_k]

        colors = ["#d73027" if v > 0 else "#4575b4" for v in sv[sorted_idx]]

        fig = go.Figure(go.Bar(
            x=sv[sorted_idx][::-1],
            y=[feature_names[i] for i in sorted_idx][::-1],
            orientation="h",
            marker=dict(color=colors[::-1]),
            hovertemplate="Feature: %{y}<br>Contribution: %{x:.4f}<extra></extra>"
        ))

        fig.add_vline(x=0, line_width=1, line_color="black")
        fig.update_layout(
            title=dict(text=f"SHAP Waterfall — {cls_name}", font=dict(size=12)),
            xaxis=dict(title="Contribution"),
            yaxis=dict(title="", autorange="reversed"),
            height=350, width=500,
            margin=dict(l=150, r=30, t=40, b=60),
            showlegend=False
        )
        figs.append(fig)

    return figs


def plot_gradient_saliency(saliency, feature_names, top_k=20):
    sorted_idx = np.argsort(saliency)[::-1][:top_k]

    fig = go.Figure(go.Bar(
        x=saliency[sorted_idx][::-1],
        y=[feature_names[i] for i in sorted_idx][::-1],
        orientation="h",
        marker=dict(color="teal"),
        hovertemplate="Feature: %{y}<br>Saliency: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Top 20 Features by Gradient Saliency", x=0.5),
        xaxis=dict(title="Mean |Gradient|"),
        yaxis=dict(title="", autorange="reversed"),
        height=600, width=800,
        margin=dict(l=180, r=40, t=60, b=80)
    )

    return fig


def plot_gnnexplainer_importance(feature_names, explanations_dict):
    n = len(explanations_dict)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=list(explanations_dict.keys()),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    colors = px.colors.qualitative.Set2
    for i, (cls_name, result) in enumerate(explanations_dict.items()):
        row = i // cols + 1
        col = i % cols + 1

        features = result["features"][::-1]
        scores = result["scores"][::-1]

        fig.add_trace(
            go.Bar(
                x=scores,
                y=features,
                orientation="h",
                marker=dict(color=colors[i % len(colors)]),
                hovertemplate="Feature: %{y}<br>Score: %{x:.4f}<extra></extra>",
                showlegend=False
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(text="GNNExplainer: Most Important Features Per Attack Type", x=0.5),
        height=250 * rows, width=350 * cols,
        margin=dict(l=120, r=30, t=60, b=60)
    )

    for i in range(n):
        row = i // cols + 1
        col = i % cols + 1
        fig.update_xaxes(title_text="Feature Mask Score", row=row, col=col)

    return fig
