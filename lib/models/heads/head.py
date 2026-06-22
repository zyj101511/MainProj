import torch
from torch import nn


class TestHead(nn.Module):
    def __init__(self, hidden_dim, search_feature_size=2):
        super().__init__()
        self.search_feature_size = search_feature_size
        self.box = nn.Linear(hidden_dim, 4)
        self.score = nn.Linear(hidden_dim, 1)
        self.size = nn.Linear(hidden_dim, 2)
        self.offset = nn.Linear(hidden_dim, 2)

    def forward(self, feature: torch.Tensor, return_dist=False):
        # feature: [T, B, C, H, W]
        if feature.dim() != 5:
            raise RuntimeError(f'feature should be [T, B, C, H, W], got {feature.shape}')

        feat = feature.mean(dim=0)
        pooled = feat.mean(dim=(-1, -2))

        pred_box = self.box(pooled).sigmoid()
        score_map = self.score(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).sigmoid()
        size_map = self.size(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).sigmoid()
        offset_map = self.offset(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return pred_box, offset_map, size_map, score_map


def build_head(cfg, hidden_dim):
    search_size = cfg.DATA.SEARCH.WIDTH // cfg.MODEL.BACKBONE.STRIDE
    return TestHead(hidden_dim=hidden_dim, search_feature_size=search_size)
