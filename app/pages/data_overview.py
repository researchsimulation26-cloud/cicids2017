import streamlit as st
import plotly.express as px

from app.utils import load_data_cached, get_class_distribution
from config import FEATURE_COLS, CLASS_NAMES


def render():
    st.title("Data Overview")
    st.markdown("Explore the CIC-IDS2017 dataset used for training the CyberSAGE model.")

    df, le = load_data_cached()
    st.success(f"Loaded {len(df):,} samples with {len(FEATURE_COLS)} features and {len(CLASS_NAMES)} attack classes")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Feature Dimensions", len(FEATURE_COLS))
    with col3:
        st.metric("Attack Classes", len(CLASS_NAMES))

    st.subheader("Class Distribution")
    dist, labels = get_class_distribution(df)
    fig = px.bar(
        x=labels,
        y=dist.values,
        color=labels,
        color_discrete_sequence=px.colors.qualitative.T10,
        labels={"x": "Attack Class", "y": "Count"},
        title="Number of Samples per Class"
    )
    fig.update_layout(xaxis_tickangle=45, height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Statistics")
    col_config = {"Label": st.column_config.TextColumn("Label")}
    st.dataframe(
        df[FEATURE_COLS].describe().T.reset_index().rename(columns={"index": "Feature"}),
        use_container_width=True,
        height=400
    )

    st.subheader("Raw Data Sample")
    st.dataframe(
        df[FEATURE_COLS + ["Label"]].head(100),
        use_container_width=True,
        height=300
    )
