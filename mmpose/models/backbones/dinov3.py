# Copyright (c) OpenMMLab. All rights reserved.
import torch
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from mmcv_custom.checkpoint import load_checkpoint
from dinov3.models import build_model
from omegaconf import OmegaConf

@BACKBONES.register_module()
class DINOv3(BaseBackbone):
    def __init__(
        self,
        strict=False,
        patch_padding="pad",
        arch="vit_base",
        img_size=224,
        patch_size=16,
        pos_embed_rope_base = 100.0,
        pos_embed_rope_min_period = None,
        pos_embed_rope_max_period = None,
        pos_embed_rope_normalize_coords = 'separate',
        pos_embed_rope_shift_coords = None,
        pos_embed_rope_jitter_coords = None,
        pos_embed_rope_rescale_coords = None,
        layerscale=1.0e-05,
        ffn_layer = "mlp",
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        norm_layer='layernorm',
        n_storage_tokens=0,
        drop_path_rate=0.3,
        mask_k_bias=False,
        untie_cls_and_patch_norms=False,
        untie_global_and_local_cls_norm=False,
        fp8_enabled=False,
        **kwargs,
    ):
        super().__init__()
        self.patch_padding = patch_padding
        self.arch = arch
        self.img_size = img_size
        self.patch_size = int(patch_size)
        self.n_storage_tokens = int(n_storage_tokens)
        self.strict = bool(strict)
        
        args = OmegaConf.create({"arch": self.arch, 
                                "patch_size": self.patch_size,
                                "pos_embed_rope_base": pos_embed_rope_base,
                                "pos_embed_rope_min_period": pos_embed_rope_min_period,
                                "pos_embed_rope_max_period": pos_embed_rope_max_period ,
                                "pos_embed_rope_normalize_coords": pos_embed_rope_normalize_coords,
                                "pos_embed_rope_shift_coords": pos_embed_rope_shift_coords,
                                "pos_embed_rope_jitter_coords": pos_embed_rope_jitter_coords,
                                "pos_embed_rope_rescale_coords": pos_embed_rope_rescale_coords,
                                "layerscale": layerscale,
                                "ffn_layer": ffn_layer,
                                "qkv_bias": qkv_bias,
                                "proj_bias": proj_bias,
                                "ffn_bias": ffn_bias,
                                "norm_layer": norm_layer,
                                "n_storage_tokens": self.n_storage_tokens,
                                "drop_path_rate": drop_path_rate,
                                "mask_k_bias": mask_k_bias,
                                "untie_cls_and_patch_norms": untie_cls_and_patch_norms,
                                "untie_global_and_local_cls_norm": untie_global_and_local_cls_norm, 
                                "fp8_enabled": fp8_enabled})
        self.dino, self.embed_dim = build_model(args, only_teacher=True, img_size = self.img_size)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained, patch_padding=self.patch_padding)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dict = self.dino.forward_features(x)

        patch_tokens = out_dict.get("x_norm_patchtokens", None)
        
        if patch_tokens is None:
            x_norm = out_dict["x_norm"] 
            patch_tokens = x_norm[:, 1 + self.n_storage_tokens :, :]

        B, N, C = patch_tokens.shape
        H, W = x.shape[-2], x.shape[-1]
        Hp, Wp = H // self.patch_size, W // self.patch_size
        
        if Hp * Wp != N:
            raise ValueError(f"Token count mismatch: N={N} patches, but image {H}x{W} "
                             f"implies {Hp}x{Wp}={Hp*Wp} patches.")

        feat = patch_tokens.transpose(1, 2).reshape(B, C, Hp, Wp).contiguous()
        return feat