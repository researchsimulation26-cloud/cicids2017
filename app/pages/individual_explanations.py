import streamlit as st
import numpy as np

from app.utils import load_explain_cached
from visualize.importance import plot_shap_waterfall
from config import CLASS_NAMES, FEATURE_COLS


def render():
    st.title("Individual Explanation")
    st.markdown("SHAP waterfall plots for individual network flow predictions.")

    with st.spinner("Loading SHAP values..."):
        try:
            X_explain, y_explain, shap_values = load_explain_cached()
        except Exception as e:
            st.error(f"Could not load SHAP values: {e}")
            st.info("Upload X_explain.pkl, y_explain.pkl, and shap_values.pkl to your HF dataset.")
            return

    st.success(f"Loaded {len(X_explain):,} explained samples")

    col1, col2 = st.columns(2)
    with col1:
        target_class = st.selectbox("Select Attack Class", CLASS_NAMES)
    with col2:
        cls_idx = CLASS_NAMES.index(target_class)
        sample_indices = np.where(y_explain == cls_idx)[0]
        if len(sample_indices) == 0:
            st.warning(f"No samples found for {target_class}")
            sample_idx = 0
        else:
            sample_idx = st.selectbox(
                "Select Sample",
                sample_indices,
                format_func=lambda i: f"Sample {i} (true: {CLASS_NAMES[y_explain[i]]})"
            )

    top_k = st.slider("Features to show", 5, 20, 12, 1)

    figs = plot_shap_waterfall(
        shap_values, sample_idx,
        FEATURE_COLS, CLASS_NAMES,
        top_k=top_k
    )

    st.subheader(f"Top {top_k} Feature Contributions for Sample {sample_idx}")
    st.caption(f"True class: {CLASS_NAMES[y_explain[sample_idx]]}")

    for cls_name, fig in zip(CLASS_NAMES, figs):
        with st.expander(f"View as {cls_name}", expanded=(cls_name == target_class)):
            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Red bars indicate features pushing the prediction **toward** the class, "
        "blue bars push **away**. Larger absolute values mean higher impact."
    )
