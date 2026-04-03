from __future__ import annotations

from copy import deepcopy

import torch.nn as nn
from mmengine.model import BaseModule, ModuleList

from mmdet.registry import MODELS


@MODELS.register_module()
class CBSwinTransformer(BaseModule):
    """A lightweight composite Swin backbone for project-side MMDet configs.

    This keeps the integration fully inside the project by composing multiple
    Swin backbones and fusing same-stage outputs with learnable 1x1 adapters.
    """

    def __init__(
        self,
        backbone_cfg: dict,
        num_backbones: int = 2,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if num_backbones < 2:
            raise ValueError("CBSwinTransformer requires num_backbones >= 2")

        backbone_cfg = deepcopy(backbone_cfg)
        backbone_cfg.setdefault("out_indices", out_indices)
        self.out_indices = tuple(out_indices)
        self.num_backbones = int(num_backbones)

        self.backbones = ModuleList()
        for _ in range(self.num_backbones):
            self.backbones.append(MODELS.build(deepcopy(backbone_cfg)))

        embed_dims = int(backbone_cfg.get("embed_dims", 96))
        self.out_channels = [embed_dims * (2**idx) for idx in self.out_indices]
        self.cb_linears = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=1) for channels in self.out_channels
        )

    def init_weights(self) -> None:
        for backbone in self.backbones:
            backbone.init_weights()
        for conv in self.cb_linears:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        branch_outs = [backbone(x) for backbone in self.backbones]
        fused_outs = list(branch_outs[-1])
        for stage_idx, conv in enumerate(self.cb_linears):
            for branch_idx in range(self.num_backbones - 1):
                fused_outs[stage_idx] = fused_outs[stage_idx] + conv(branch_outs[branch_idx][stage_idx])
        return tuple(fused_outs)
