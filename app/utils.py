import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import torch
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, login

from config import HF_DATASET_REPO, HF_MODEL_REPO, FEATURE_COLS, CLASS_NAMES
from data.loader import load_data, load_model, load_explain_data
from model.cybersage import CyberSAGE


def init_session():
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "explain_loaded" not in st.session_state:
        st.session_state.explain_loaded = False
    if "predictions" not in st.session_state:
        st.session_state.predictions = None


def login_hf():
    token = st.secrets.get("HF_TOKEN", None)
    if token:
        login(token, add_to_git_credential=False)


@st.cache_resource(ttl="1d")
def load_data_cached():
    login_hf()
    df, le = load_data(use_hf=True)
    return df, le


@st.cache_resource(ttl="1d")
def load_model_cached():
    login_hf()
    model, device = load_model(use_hf=True)
    return model, device


@st.cache_resource(ttl="1d")
def load_explain_cached():
    login_hf()
    X, y, shap_values = load_explain_data(use_hf=True)
    return X, y, shap_values


@st.cache_resource(ttl="1d")
def load_graph_cached():
    login_hf()
    from data.loader import load_graph
    return load_graph(use_hf=True)


def get_class_distribution(df, label_col="label_enc"):
    dist = df[label_col].value_counts().sort_index()
    labels = [CLASS_NAMES[i] for i in dist.index if i < len(CLASS_NAMES)]
    return dist, labels


def run_inference(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)
        pred = out.argmax(dim=1)
    return probs.cpu().numpy(), pred.cpu().numpy()


def get_test_results(model, data, device):
    test_mask = data.test_mask.cpu().numpy()
    probs, pred = run_inference(model, data, device)
    true = data.y.cpu().numpy()
    return (
        probs[test_mask],
        pred[test_mask],
        true[test_mask]
    )
