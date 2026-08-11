from collections import OrderedDict

import torch
from torch import nn


class PatchLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        patch_size: int = 4,
        embed_dim: int = 128,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=False,
        )
        self.classification_head = nn.Linear(hidden_size, num_classes)

    @staticmethod
    def patch_index_to_coord(grid_size: int = 16):
        return [(i // grid_size, i % grid_size) for i in range(grid_size * grid_size)]

    def image_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(x.shape[0], -1, self.patch_dim)

    def forward_features(self, x: torch.Tensor, return_intermediates: bool = False):
        intermediates = OrderedDict()
        patches = self.image_to_patches(x)
        embedded = self.proj(patches)
        outputs, (h_n, _) = self.lstm(embedded)
        feature = h_n[-1]
        if return_intermediates:
            intermediates["patches"] = patches
            intermediates["patch_embeddings"] = embedded
            intermediates["lstm_outputs"] = outputs
            intermediates["last_hidden"] = feature
            return feature, intermediates
        return feature

    def forward(self, x: torch.Tensor):
        return self.classification_head(self.forward_features(x))
