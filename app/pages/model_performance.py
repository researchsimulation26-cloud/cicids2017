import streamlit as st
import numpy as np
import pandas as pd

from app.utils import load_data_cached, load_model_cached, load_graph_cached, get_test_results, clear_memory
from visualize.performance import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_support_vs_f1
)
from config import CLASS_NAMES


def render():
    st.title("Model Performance")
    st.markdown("Evaluation metrics for CyberSAGE on the CIC-IDS2017 test set.")

    try:
        model, device = load_model_cached()
        data = load_graph_cached()

        data = data.to(device)
        probs, pred, true = get_test_results(model, data, device)

        del data
        clear_memory()
    except Exception as e:
        st.error("Failed to load model or graph for Model Performance.")
        st.exception(e)
        return

    st.success(f"Test set: {len(true):,} samples")

    viz_option = st.radio(
        "Visualization",
        ["Confusion Matrix", "Per-Class Metrics", "Support vs F1"],
        horizontal=True
    )

    if viz_option == "Confusion Matrix":
        norm_mode = st.checkbox("Show normalized values", value=True)
        fig = plot_confusion_matrix(true, pred, CLASS_NAMES)
        st.plotly_chart(fig, width='stretch')

        with st.expander("Confusion Matrix Details"):
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true, pred)
            st.dataframe(
                pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES),
                width='stretch'
            )

    elif viz_option == "Per-Class Metrics":
        fig = plot_per_class_metrics(true, pred, CLASS_NAMES)
        st.plotly_chart(fig, width='stretch')

        from sklearn.metrics import classification_report
        report = classification_report(
            true, pred,
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True
        )
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df, width='stretch')

    elif viz_option == "Support vs F1":
        fig = plot_support_vs_f1(true, pred, CLASS_NAMES)
        st.plotly_chart(fig, width='stretch')

        st.info(
            "Classes with low support tend to have lower F1 scores. "
            "The red dashed line indicates F1 = 0.7 threshold."
        )
