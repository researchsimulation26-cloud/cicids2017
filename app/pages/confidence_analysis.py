import streamlit as st

from app.utils import load_model_cached, load_graph_cached, get_test_results
from visualize.confidence import plot_confidence_distribution, plot_accuracy_at_threshold


def render():
    st.title("Confidence Analysis")
    st.markdown("Analyze the model's prediction confidence across correct and incorrect predictions.")

    model, device = load_model_cached()
    data = load_graph_cached()
    data = data.to(device)

    probs, pred, true = get_test_results(model, data, device)
    st.success(f"Analyzing {len(true):,} test samples")

    tab1, tab2 = st.tabs(["Confidence Distribution", "Accuracy vs Threshold"])

    with tab1:
        fig = plot_confidence_distribution(probs, true, pred)
        st.plotly_chart(fig, use_container_width=True)

        correct = (pred == true).sum()
        incorrect = (pred != true).sum()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(true))
        with col2:
            st.metric("Correct", f"{correct} ({correct/len(true):.1%})")
        with col3:
            st.metric("Incorrect", f"{incorrect} ({incorrect/len(true):.1%})")

        st.info(
            "Ideally, correct predictions have high confidence (right-skewed) "
            "while incorrect ones have lower confidence (left-skewed). "
            "Overlap indicates uncertainty in the model's predictions."
        )

    with tab2:
        fig = plot_accuracy_at_threshold(probs, true, pred)
        st.plotly_chart(fig, use_container_width=True)

        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        mask = probs.max(axis=1) >= threshold
        if mask.sum() > 0:
            acc = (pred[mask] == true[mask]).mean()
            coverage = mask.mean()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy at threshold", f"{acc:.2%}")
            with col2:
                st.metric("Coverage", f"{coverage:.2%} ({mask.sum():,} samples)")
        else:
            st.warning("No samples meet this threshold.")
