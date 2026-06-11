import shap
import numpy as np
import torch


class NodeClassifierWrapper:
    def __init__(self, model, edge_index, device):
        self.model = model
        self.edge_index = edge_index
        self.device = device

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dummy_edge = torch.zeros((2, 0), dtype=torch.long).to(self.device)
            out = self.model(X_tensor, dummy_edge)
            probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs


def compute_shap_values(model, data, device, n_background=200, n_explain=500, nsamples=100):
    wrapper = NodeClassifierWrapper(model, data.edge_index, device)
    test_idx = torch.where(data.test_mask)[0].cpu().numpy()
    rng = np.random.default_rng(42)

    bg_idx = rng.choice(test_idx, min(n_background, len(test_idx)), replace=False)
    explain_idx = rng.choice(test_idx, min(n_explain, len(test_idx)), replace=False)

    X_bg = data.x[bg_idx].cpu().numpy()
    X_explain = data.x[explain_idx].cpu().numpy()
    y_explain = data.y[explain_idx].cpu().numpy()

    explainer_shap = shap.KernelExplainer(wrapper.predict_proba, X_bg)
    shap_values = explainer_shap.shap_values(X_explain, nsamples=nsamples)

    return shap_values, X_explain, y_explain


def get_top_features_per_class(shap_values, feature_names, class_names, top_k=10):
    results = {}
    for cls_idx in range(len(class_names)):
        sv = np.abs(shap_values[cls_idx])
        mean_importance = sv.mean(axis=0)
        top_idx = np.argsort(mean_importance)[::-1][:top_k]
        results[class_names[cls_idx]] = {
            "features": [feature_names[i] for i in top_idx],
            "values": mean_importance[top_idx]
        }
    return results
