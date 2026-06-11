import streamlit as st
import numpy as np

from app.utils import load_model_cached, load_graph_cached, load_explain_cached
from explain.saliency import compute_gradient_saliency
from visualize.importance import (
    plot_shap_global_importance,
    plot_shap_per_class_heatmap,
    plot_gradient_saliency
)
from config import CLASS_NAMES, FEATURE_COLS


def render():
    st.title("Feature Importance")
    st.markdown("Understand which features drive the model's decisions.")

    tab1, tab2, tab3 = st.tabs(["SHAP Global", "SHAP Per-Class Heatmap", "Gradient Saliency"])

    with tab1:
        st.subheader("Global SHAP Feature Importance")
        st.markdown("Mean absolute SHAP values across all samples and classes.")
        with st.spinner("Loading SHAP values..."):
            try:
                X_explain, y_explain, shap_values = load_explain_cached()
                fig = plot_shap_global_importance(shap_values, FEATURE_COLS, CLASS_NAMES)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load SHAP values: {e}")
                st.info("Upload shap_values.pkl to your Hugging Face dataset repo.")

    with tab2:
        st.subheader("Per-Class Feature Importance Heatmap")
        st.markdown("Mean |SHAP| per feature for each attack class.")
        with st.spinner("Loading SHAP values..."):
            try:
                X_explain, y_explain, shap_values = load_explain_cached()
                top_k = st.slider("Top K features", 5, 20, 15, 1)
                fig = plot_shap_per_class_heatmap(shap_values, FEATURE_COLS, CLASS_NAMES, top_k=top_k)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load SHAP values: {e}")

    with tab3:
        st.subheader("Gradient Saliency")
        st.markdown("Mean absolute gradient of the loss w.r.t. each input feature.")
        with st.spinner("Computing saliency map..."):
            model, device = load_model_cached()
            data = load_graph_cached()
            data = data.to(device)
            saliency = compute_gradient_saliency(model, data)
            fig = plot_gradient_saliency(saliency, FEATURE_COLS, top_k=20)
            st.plotly_chart(fig, use_container_width=True)
