import torch
import numpy as np


def compute_gradient_saliency(model, data, mask_name="train_mask"):
    model.train()
    x_input = data.x.clone().detach().requires_grad_(True)
    out = model(x_input, data.edge_index)
    mask = getattr(data, mask_name)
    loss = out[mask].sum()
    loss.backward()

    saliency = x_input.grad[mask].abs().mean(dim=0).cpu().numpy()
    return saliency


def get_top_salient_features(saliency, feature_names, top_k=20):
    sorted_idx = np.argsort(saliency)[::-1]
    return {
        "features": [feature_names[i] for i in sorted_idx[:top_k]],
        "values": saliency[sorted_idx[:top_k]]
    }
