import numpy as np
import faiss
import torch
from config import CLASS_NAMES


def build_custom_graph(X, y, k_same=5, k_cross=2):
    src, dst = [], []
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    X_n = (X / norms).astype(np.float32)

    for cls_val in np.unique(y):
        idx = np.where(y == cls_val)[0]
        if len(idx) < 2:
            continue
        index = faiss.IndexFlatIP(X_n.shape[1])
        index.add(X_n[idx])
        _, neighbors = index.search(X_n[idx], min(k_same + 1, len(idx)))
        for local_i, global_i in enumerate(idx):
            for nb_local in neighbors[local_i, 1:]:
                if nb_local >= 0:
                    src.append(global_i)
                    dst.append(idx[nb_local])

    benign_label = int(np.where(np.array(CLASS_NAMES) == "BENIGN")[0][0])
    benign_idx = np.where(y == benign_label)[0]
    attack_idx = np.where(y != benign_label)[0]

    if len(benign_idx) > 0 and len(attack_idx) > 0:
        index_b = faiss.IndexFlatIP(X_n.shape[1])
        index_b.add(X_n[benign_idx])
        _, b_neighbors = index_b.search(X_n[attack_idx], k_cross)
        for local_i, global_i in enumerate(attack_idx):
            for nb_local in b_neighbors[local_i]:
                if nb_local >= 0:
                    src.append(global_i)
                    dst.append(benign_idx[nb_local])

    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
