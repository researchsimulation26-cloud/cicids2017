import streamlit as st

st.set_page_config(
    page_title="CyberSAGE — GNN Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.utils import init_session, login_hf
from app.pages import (
    data_overview,
    model_performance,
    embeddings,
    feature_importance,
    confidence_analysis,
    individual_explanations
)

init_session()
login_hf()

PAGES = {
    "Data Overview": data_overview,
    "Model Performance": model_performance,
    "Embedding Space": embeddings,
    "Feature Importance": feature_importance,
    "Confidence Analysis": confidence_analysis,
    "Individual Explanations": individual_explanations,
}

st.sidebar.title("🛡️ CyberSAGE")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "GNN-based Network Intrusion Detection\n\n"
    "Model: **CyberSAGE**\n"
    "Dataset: **CIC-IDS2017**\n"
    "Framework: **PyTorch Geometric**"
)

selection = st.sidebar.radio("Navigate", list(PAGES.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How to use:**\n"
    "1. Start with **Data Overview**\n"
    "2. Review **Model Performance**\n"
    "3. Explore **Embedding Space**\n"
    "4. Understand **Feature Importance**\n"
    "5. Check **Confidence Analysis**\n"
    "6. Dive into **Individual Explanations**"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data sourced from Hugging Face Dataset Hub.\n"
    "Model hosted on Hugging Face Model Hub."
)

page = PAGES[selection]
page.render()
