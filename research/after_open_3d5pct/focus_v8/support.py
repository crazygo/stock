"""Training-only support masks, frozen in each neural checkpoint."""
from __future__ import annotations
import numpy as np
import torch
from .core import FocusC

class SupportedC(FocusC):
    def __init__(self, route, recipe):
        super().__init__(route, recipe)
        self.register_buffer('day_support', torch.ones(126, dtype=torch.bool))
        self.register_buffer('group_support', torch.ones(6, dtype=torch.bool))

    def fit_support(self, arrays, rows, ids):
        if self.daily:
            self.day_support.copy_((arrays[2][ids, :, 10] > 0).any(0))
        if self.group:
            dates = rows.iloc[ids].session_date.to_numpy()
            for slot in range(6):
                seen = (arrays[3][ids, -1, slot, 9] > 0).numpy()
                self.group_support[slot] = len(np.unique(dates[seen])) >= 10

    def forward(self, x5, x60, xday, group_seq, prior):
        if self.daily:
            xday = xday.masked_fill(~self.day_support[None, :, None], 0)
        if self.group:
            group_seq = group_seq.masked_fill(~self.group_support[None, None, :, None], 0)
        return super().forward(x5, x60, xday, group_seq, prior)


def build_c(route, recipe):
    return SupportedC(route, recipe) if recipe.get('support') else FocusC(route, recipe)
