from typing import Sequence
import torch
from torch import nn

class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        missing_context_mode: str,
        use_context_features: bool,
    ):
        super().__init__()
        self.use_context_features = use_context_features
        if self.use_context_features:
            if input_dim % 3 != 0:
                raise ValueError(f"Expected input_dim divisible by 3 for [prefix, mask, suffix], got {input_dim}.")
            self.segment_dim = input_dim // 3
        else:
            self.segment_dim = input_dim
        self.missing_context_mode = missing_context_mode
        if self.use_context_features and self.missing_context_mode == "learned":
            self.prefix_missing_embedding = nn.Parameter(torch.zeros(self.segment_dim))
            self.suffix_missing_embedding = nn.Parameter(torch.zeros(self.segment_dim))
        layers = [nn.LayerNorm(input_dim)]
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        prefix_missing: torch.Tensor | None = None,
        suffix_missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_context_features and self.missing_context_mode == "learned":
            if prefix_missing is None or suffix_missing is None:
                raise ValueError("Missing-context masks are required when missing_context_mode='learned'.")
            prefix_x, mask_x, suffix_x = torch.split(x, self.segment_dim, dim=-1)
            prefix_missing = prefix_missing.to(dtype=torch.bool, device=x.device)
            suffix_missing = suffix_missing.to(dtype=torch.bool, device=x.device)
            prefix_x = torch.where(
                prefix_missing.unsqueeze(-1),
                self.prefix_missing_embedding.unsqueeze(0),
                prefix_x,
            )
            suffix_x = torch.where(
                suffix_missing.unsqueeze(-1),
                self.suffix_missing_embedding.unsqueeze(0),
                suffix_x,
            )
            x = torch.cat([prefix_x, mask_x, suffix_x], dim=-1)
        return self.net(x).squeeze(-1)
