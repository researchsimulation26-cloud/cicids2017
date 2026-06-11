import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download

from config import HF_DATASET_REPO, HF_MODEL_REPO, FEATURE_COLS, CLASS_NAMES, SUBSAMPLE_SIZE


def download_artifact(repo_id, filename, repo_type="dataset"):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def load_data(use_hf=True, local_dir=None, sample_size=None):
    if sample_size is None:
        sample_size = SUBSAMPLE_SIZE

    if use_hf:
        parquet_path = download_artifact(HF_DATASET_REPO, "cicids2017_final.parquet")
    else:
        parquet_path = str(Path(local_dir) / "cicids2017_final.parquet")

    df = pd.read_parquet(parquet_path)

    label_col = "label_grouped" if "label_grouped" in df.columns else "Label"
    df = df[df[label_col].isin(CLASS_NAMES)].copy()
    df = df.reset_index(drop=True)

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    df["label_enc"] = le.transform(df[label_col])

    if len(df) > sample_size:
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


def _subsample_graph(data, max_nodes):
    if data.num_nodes <= max_nodes:
        return data
    import torch

    n = data.num_nodes
    rng = np.random.default_rng(42)
    chosen = rng.choice(n, max_nodes, replace=False)
    chosen = torch.tensor(chosen, dtype=torch.long)

    node_map = torch.full((n,), -1, dtype=torch.long)
    node_map[chosen] = torch.arange(max_nodes)

    data.x = data.x[chosen]
    data.y = data.y[chosen]
    data.train_mask = data.train_mask[chosen]
    data.val_mask = data.val_mask[chosen]
    data.test_mask = data.test_mask[chosen]

    edge_mask = node_map[data.edge_index[0]] != -1
    edge_mask &= node_map[data.edge_index[1]] != -1
    data.edge_index = node_map[data.edge_index[:, edge_mask]]

    return data


def load_graph(use_hf=True, local_dir=None, max_nodes=None):
    import torch
    if max_nodes is None:
        max_nodes = SUBSAMPLE_SIZE

    if use_hf:
        path = download_artifact(HF_DATASET_REPO, "cicids_graph.pt")
    else:
        path = str(Path(local_dir) / "cicids_graph.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)

    OTHER_CLASS = 8
    if data.num_classes > len(CLASS_NAMES):
        mask = data.y != OTHER_CLASS
        data.x = data.x[mask]
        data.y = data.y[mask]
        data.train_mask = data.train_mask[mask]
        data.val_mask = data.val_mask[mask]
        data.test_mask = data.test_mask[mask]

        remap = torch.full((data.num_classes,), -1, dtype=torch.long)
        new_idx = 0
        for i in range(data.num_classes):
            if i != OTHER_CLASS:
                remap[i] = new_idx
                new_idx += 1
        data.y = remap[data.y]

        old_nodes = torch.where(mask)[0]
        node_map = torch.full((mask.shape[0],), -1, dtype=torch.long)
        node_map[old_nodes] = torch.arange(old_nodes.shape[0])
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        data.edge_index = node_map[data.edge_index[:, edge_mask]]
        data.num_classes = new_idx

    data = _subsample_graph(data, max_nodes)

    return data


def load_explain_data(use_hf=True, local_dir=None):
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
    if shap_values.ndim == 3 and shap_values.shape[2] == len(CLASS_NAMES):
        shap_values = shap_values.transpose(2, 0, 1)
    return X, y, shap_values
