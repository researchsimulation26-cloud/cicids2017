import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from model.cybersage import EmbeddingExtractor


def compute_embeddings(model, data, device):
    extractor = EmbeddingExtractor(model).to(device)
    extractor.eval()
    with torch.no_grad():
        embeddings = extractor(data.x, data.edge_index).cpu().numpy()
    return embeddings


def plot_tsne_embeddings(embeddings, labels, class_names, perplexity=40, n_samples=5000):
    rng = np.random.default_rng(42)
    n = min(n_samples, embeddings.shape[0])
    idx = rng.choice(embeddings.shape[0], n, replace=False)

    emb_sample = embeddings[idx]
    label_sample = labels[idx]

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    emb_2d = tsne.fit_transform(emb_sample)

    colors = px.colors.qualitative.T10 + px.colors.qualitative.Set2
    fig = go.Figure()

    for i, name in enumerate(class_names):
        mask = label_sample == i
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=emb_2d[mask, 0],
            y=emb_2d[mask, 1],
            mode="markers",
            marker=dict(size=4, color=colors[i % len(colors)], opacity=0.6),
            name=name,
            hovertemplate=f"Class: {name}<br>Node index: %{{customdata}}<extra></extra>",
            customdata=np.where(mask)[0]
        ))

    fig.update_layout(
        title=dict(text="t-SNE of GNN Node Embeddings", x=0.5),
        xaxis=dict(title="t-SNE 1", zeroline=False),
        yaxis=dict(title="t-SNE 2", zeroline=False),
        legend=dict(
            itemsizing="constant",
            font=dict(size=10)
        ),
        width=900, height=650,
        margin=dict(l=60, r=40, t=60, b=80),
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa"
    )

    return fig
