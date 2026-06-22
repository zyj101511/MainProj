import torch
from torch import nn


class TestBackbone(nn.Module):
    def __init__(self, embed_dim=8, template_tokens=1, search_feature_size=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.template_tokens = template_tokens
        self.search_feature_size = search_feature_size
        self.search_tokens = search_feature_size ** 2
        self.proj = nn.Linear(3, embed_dim)

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        # template/search: [T, B, C, H, W]
        if template.dim() != 5 or search.dim() != 5:
            raise RuntimeError('template and search should be [T, B, C, H, W]')

        T, B = search.shape[:2]
        z = template.mean(dim=(-1, -2)).permute(0, 1, 3, 2)
        z = z[:, :, :1, :].expand(T, B, self.template_tokens, 3)

        x = search.mean(dim=(-1, -2)).permute(0, 1, 3, 2)
        x = x[:, :, :1, :].expand(T, B, self.search_tokens, 3)

        features = self.proj(torch.cat([z, x], dim=2))
        return features, {'attn': None}


def sdtrack_tiny():
    return TestBackbone()
