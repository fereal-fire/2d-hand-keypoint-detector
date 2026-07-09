# Copyright (c) OpenMMLab. All rights reserved.
import torch
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from mmcv_custom.checkpoint import load_checkpoint
from dinov2.models import build_model
from omegaconf import OmegaConf

@BACKBONES.register_module()
class DINOv2(BaseBackbone):
    def __init__(
        self,
        strict=False,
        patch_padding="pad",
        arch="vit_base",
        img_size=224,
        patch_size=16,
        layerscale=1.0e-05,
        ffn_layer = "mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=0,
        interpolate_offset=0.1,
        interpolate_antialias=False,
        in_chans=3,
        drop_path_rate=0.3,
        drop_path_uniform=False,
        channel_adaptive=False,
        **kwargs,
    ):
        super().__init__()
        self.patch_padding = patch_padding
        self.arch = arch
        self.img_size = img_size
        self.patch_size = int(patch_size)
        self.num_register_tokens = int(num_register_tokens)
        self.strict = bool(strict)
        
        args = OmegaConf.create({"arch": self.arch, 
                                "patch_size": self.patch_size,
                                "layerscale": layerscale,
                                "ffn_layer": ffn_layer,
                                "block_chunks": block_chunks,
                                "qkv_bias": qkv_bias,
                                "proj_bias": proj_bias,
                                "ffn_bias": ffn_bias,
                                "num_register_tokens": self.num_register_tokens,
                                "interpolate_offset": interpolate_offset,
                                "interpolate_antialias": interpolate_antialias,
                                "in_chans": in_chans,
                                "drop_path_rate": drop_path_rate,
                                "drop_path_uniform": drop_path_uniform,
                                "channel_adaptive": channel_adaptive})
        self.dino, self.embed_dim = build_model(args, only_teacher=True, img_size = self.img_size)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dino.forward_features(x)

        patch_tokens = out.get("x_norm_patchtokens", None)
        if patch_tokens is None:
            x_norm = out["x_norm"] 
            patch_tokens = x_norm[:, 1 + self.num_register_tokens :, :]

        B, N, C = patch_tokens.shape
        H, W = x.shape[-2], x.shape[-1]
        Hp, Wp = H // self.patch_size, W // self.patch_size
        if Hp * Wp != N:
            raise ValueError(f"N={N} != Hp*Wp={Hp*Wp} for input {(H,W)} with patch={self.patch_size}")

        feat = patch_tokens.transpose(1, 2).reshape(B, C, Hp, Wp).contiguous()
        return feat