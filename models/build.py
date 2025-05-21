# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from .soft_mixture_of_experts.vit import SoftMoEViT, ViT, vit_small





def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm


    if model_type == 'soft_moe_128':
        print("soft moe")
        model = SoftMoEViT(num_classes=config.MODEL.NUM_CLASSES, num_experts=128, num_encoder_layers=8, nhead=6, d_model=384, last_n=2)
    elif model_type == 'soft_moe_32':
        print("soft moe")
        model = SoftMoEViT(num_classes=config.MODEL.NUM_CLASSES, num_experts=32, num_encoder_layers=8, nhead=6, d_model=384, last_n=2)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
