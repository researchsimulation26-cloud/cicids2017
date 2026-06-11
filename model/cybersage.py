import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm


class CyberSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.4):
        super().__init__()
        self.convs = nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_channels),
            BatchNorm(hidden_channels),
            BatchNorm(hidden_channels),
        ])
        self.skip = nn.Linear(in_channels, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_channels)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        s = self.skip(x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x + s)


class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        s = self.model.skip(x)
        for conv, bn in zip(self.model.convs, self.model.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=0.0, training=False)
        return x + s
