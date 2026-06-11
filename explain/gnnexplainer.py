import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer


def setup_gnnexplainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        )
    )


def explain_node(explainer, data, node_id):
    return explainer(data.x, data.edge_index, index=node_id)


def get_feature_importance_from_explanation(explanation, feature_names, top_k=15):
    if explanation.node_mask.dim() == 2:
        raw_mask = explanation.node_mask[0].cpu().numpy()
    else:
        raw_mask = explanation.node_mask.cpu().numpy().flatten()

    if len(raw_mask) > len(feature_names):
        raw_mask = raw_mask[:len(feature_names)]

    top_idx = np.argsort(raw_mask)[::-1][:top_k]
    return {
        "features": [feature_names[i] for i in top_idx],
        "scores": raw_mask[top_idx]
    }


def find_correct_test_node(data, target_class, model=None, device=None):
    test_nodes = torch.where(data.test_mask)[0]
    if model is not None:
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred_test = out[test_nodes].argmax(dim=1)
        candidates = test_nodes[(data.y[test_nodes] == target_class) &
                                (pred_test == target_class)]
    else:
        candidates = test_nodes[data.y[test_nodes] == target_class]

    if len(candidates) == 0:
        return None
    return candidates[0].item()
