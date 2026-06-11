import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score


def get_dampened_weights(y, device):
    classes = np.unique(y)
    raw = compute_class_weight("balanced", classes=classes, y=y)
    dampened = np.sqrt(raw)
    dampened = dampened / dampened.mean()
    return torch.tensor(dampened, dtype=torch.float32).to(device)


def train_epoch(model, data, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, criterion, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask]).item()
    pred = out[mask].argmax(dim=1).cpu().numpy()
    true = data.y[mask].cpu().numpy()
    f1 = f1_score(true, pred, average="macro", zero_division=0)
    acc = (pred == true).mean()
    return loss, acc, f1


def train_model(model, data, criterion, optimizer, scheduler, epochs=400, early_stop_patience=50):
    best_val_f1 = 0.0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, data, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, data, criterion, data.val_mask)

        if scheduler:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            break

    return history, best_val_f1
