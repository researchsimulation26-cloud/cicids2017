import streamlit as st
import numpy as np

from app.utils import load_model_cached, load_graph_cached
from visualize.embeddings import compute_embeddings, plot_tsne_embeddings
from config import CLASS_NAMES


def render():
    st.title("Embedding Space")
    st.markdown("t-SNE visualization of the GNN's learned node embeddings.")

    model, device = load_model_cached()
    data = load_graph_cached()
    data = data.to(device)

    with st.spinner("Computing embeddings..."):
        embeddings = compute_embeddings(model, data, device)

    st.success(f"Embeddings computed: {embeddings.shape[0]} nodes x {embeddings.shape[1]} dimensions")

    col1, col2 = st.columns(2)
    with col1:
        perplexity = st.slider("t-SNE Perplexity", 5, 100, 40, 5)
    with col2:
        n_samples = st.slider("Number of Samples", 1000, 20000, 5000, 1000)

    classes_to_show = st.multiselect(
        "Classes to display",
        CLASS_NAMES,
        default=CLASS_NAMES[:6]
    )
    class_indices = [CLASS_NAMES.index(c) for c in classes_to_show]
    labels = data.y.cpu().numpy()

    mask = np.isin(labels, class_indices)
    embeddings_subset = embeddings[mask]
    labels_subset = labels[mask]

    with st.spinner("Running t-SNE..."):
        fig = plot_tsne_embeddings(
            embeddings_subset, labels_subset,
            CLASS_NAMES, perplexity=perplexity,
            n_samples=min(n_samples, len(embeddings_subset))
        )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "t-SNE projects high-dimensional embeddings to 2D. "
        "Well-separated clusters indicate the model has learned distinct "
        "representations for different attack types."
    )
