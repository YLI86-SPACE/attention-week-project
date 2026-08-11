from collections import OrderedDict
from typing import Literal, Optional, Tuple

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .rope import apply_rope_2d
from .window_attention import WindowMultiHeadSelfAttention


class MLP(nn.Module):
    def __init__(self, dim: int = 128, ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = dim * ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        mlp_ratio: int = 4,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        attention: Literal["full", "window"] = "full",
        window_size: int = 4,
    ) -> None:
        super().__init__()
        self.attention_kind = attention
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        if attention == "full":
            self.attn = MultiHeadSelfAttention(dim, heads, attn_dropout)
        elif attention == "window":
            self.attn = WindowMultiHeadSelfAttention(dim, heads, window_size, attn_dropout)
        else:
            raise ValueError("attention must be full or window")
        self.mlp = MLP(dim, mlp_ratio, mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int],
        rope_coords: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ):
        info = {}
        h = self.norm1(x)
        if self.attention_kind == "full":
            rope_fn = apply_rope_2d if rope_coords is not None else None
            if return_intermediates:
                attn_out, weights = self.attn(
                    h,
                    rope_coords=rope_coords,
                    rope_fn=rope_fn,
                    need_weights=True,
                )
                info["attention_weights"] = weights
            else:
                attn_out = self.attn(h, rope_coords=rope_coords, rope_fn=rope_fn)
        else:
            if return_intermediates:
                attn_out, weights = self.attn(h, grid_size, need_weights=True)
                info["window_attention_weights"] = weights
            else:
                attn_out = self.attn(h, grid_size)
        info["attention_output"] = attn_out
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        info["block_output"] = x
        if return_intermediates:
            return x, info
        return x


class MiniViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        position: Literal["abspos", "none", "rope2d"] = "abspos",
        attention: Literal["full", "window"] = "full",
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 128,
        heads: int = 4,
        blocks: int = 4,
        mlp_ratio: int = 4,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        self.position = position
        self.attention = attention
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_tokens = self.grid_size[0] * self.grid_size[1]
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, self.num_tokens, dim))
            if position == "abspos"
            else None
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    mlp_ratio,
                    attn_dropout,
                    mlp_dropout,
                    attention,
                    window_size,
                )
                for _ in range(blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(dim, eps=1e-5)
        self.classification_head = nn.Linear(dim, num_classes)
        self.regression_head = nn.Linear(dim, 3)
        self.register_buffer("coords", self._make_coords(), persistent=False)

    def _make_coords(self):
        h, w = self.grid_size
        coords = [(r, c) for r in range(h) for c in range(w)]
        return torch.tensor(coords, dtype=torch.long)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward_features(self, x: torch.Tensor, return_intermediates: bool = False):
        intermediates = OrderedDict()
        x = self.patch_tokens(x)
        intermediates["patch_tokens"] = x
        if self.pos_embed is not None:
            x = x + self.pos_embed
        rope_coords = self.coords if self.position == "rope2d" else None
        for i, block in enumerate(self.blocks):
            if return_intermediates:
                intermediates[f"block{i}_input"] = x
                x, info = block(x, self.grid_size, rope_coords, return_intermediates=True)
                intermediates[f"block{i}"] = info
            else:
                x = block(x, self.grid_size, rope_coords)
        tokens = self.final_norm(x)
        feature = tokens.mean(dim=1)
        if return_intermediates:
            intermediates["tokens_final"] = tokens
            intermediates["pooled_feature"] = feature
            return feature, intermediates
        return feature

    def forward(self, x: torch.Tensor):
        feature = self.forward_features(x)
        return self.classification_head(feature), self.regression_head(feature)
