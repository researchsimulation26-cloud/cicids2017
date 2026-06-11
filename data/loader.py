import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path
from huggingface_hub import hf_hub_download

from config import HF_DATASET_REPO, HF_MODEL_REPO, FEATURE_COLS, CLASS_NAMES


def download_artifact(repo_id, filename, repo_type="dataset"):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def load_data(use_hf=True, local_dir=None, sample_size=None):
    if use_hf:
        parquet_path = download_artifact(HF_DATASET_REPO, "cicids2017_final.parquet")
    else:
        parquet_path = str(Path(local_dir) / "cicids2017_final.parquet")

    df = pd.read_parquet(parquet_path)
    df = df[df["Label"] != "Other_Attack"].copy()
    df = df.reset_index(drop=True)

    if "label_grouped_enc" in df.columns:
        le = joblib.load(download_artifact(HF_DATASET_REPO, "label_encoder_grouped.pkl"))
        label_col = le.transform(df["Label"])
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(CLASS_NAMES)
        label_col = le.transform(df["Label"])
        df = df[df["Label"].isin(CLASS_NAMES)].copy()

    df["label_enc"] = label_col

    if sample_size and len(df) > sample_size:
        from sklearn.utils import resample
        df = resample(df, n_samples=sample_size, random_state=42, stratify=df["label_enc"])

    return df, le


def load_scaler(use_hf=True, local_dir=None):
    if use_hf:
        path = download_artifact(HF_DATASET_REPO, "scaler.pkl")
    else:
        path = str(Path(local_dir) / "scaler.pkl")
    return joblib.load(path)


def load_model(use_hf=True, local_dir=None):
    if use_hf:
        path = download_artifact(HF_MODEL_REPO, "best_model_v2.pt", repo_type="model")
    else:
        path = str(Path(local_dir) / "best_model_v2.pt")

    import torch
    from model.cybersage import CyberSAGE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CyberSAGE(
        in_channels=40,
        hidden_channels=128,
        out_channels=12,
        dropout=0.4
    )
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def load_graph(use_hf=True, local_dir=None):
    import torch
    if use_hf:
        path = download_artifact(HF_DATASET_REPO, "cicids_graph.pt")
    else:
        path = str(Path(local_dir) / "cicids_graph.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def load_explain_data(use_hf=True, local_dir=None):
    import joblib
    if use_hf:
        X_path = download_artifact(HF_DATASET_REPO, "X_explain.pkl")
        y_path = download_artifact(HF_DATASET_REPO, "y_explain.pkl")
        shap_path = download_artifact(HF_DATASET_REPO, "shap_values.pkl")
    else:
        base = Path(local_dir)
        X_path = str(base / "X_explain.pkl")
        y_path = str(base / "y_explain.pkl")
        shap_path = str(base / "shap_values.pkl")

    X = joblib.load(X_path)
    y = joblib.load(y_path)
    shap_values = joblib.load(shap_path)
    return X, y, shap_values
