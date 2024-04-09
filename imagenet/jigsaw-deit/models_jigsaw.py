# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import einops
from munch import Munch


class JigsawVisionTransformer(VisionTransformer):
    def __init__(self, mask_ratio, use_jigsaw, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.use_jigsaw = use_jigsaw

        self.num_patches = self.patch_embed.num_patches

        if self.use_jigsaw:
            self.jigsaw = torch.nn.Sequential(*[torch.nn.Linear(self.embed_dim, self.embed_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(self.embed_dim, self.embed_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(self.embed_dim, self.num_patches)])
            self.rotations = torch.nn.Sequential(*[torch.nn.Linear(self.embed_dim, self.embed_dim),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(self.embed_dim, self.embed_dim),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(self.embed_dim, 4)])
            
            for m in self.rotations.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
            # self.target = torch.arange(self.num_patches)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # target = einops.repeat(self.target, 'L -> N L', N=N) 
        # target = target.to(x.device)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # N, len_keep
        ids_masked = ids_shuffle[:, len_keep:]  # N, len_masked
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        return x_masked, ids_keep, ids_masked

    def forward_jigsaw(self, x, eval=False):
        # masking: length -> length * mask_ratio
        x, keep, masked = self.random_masking(x, 0.0 if eval else self.mask_ratio)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x_jigsaw = self.jigsaw(x[:, 1:])
        x_rot = self.rotations(x[:, 1:])
        return x_jigsaw, x_rot, keep, masked

    def forward_cls(self, x): 
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

    def forward(self, x, x_shuffled, eval=False):
        x = self.patch_embed(x)
        pred_cls = self.forward_cls(x)
        outs = Munch(pred_cls=pred_cls)
        if self.use_jigsaw:
            x_shuffled = self.patch_embed(x_shuffled)
            pred_jigsaw, pred_rot, ids_used, ids_masked = self.forward_jigsaw(x_shuffled, eval=eval)

            # print("pred_jigsaw", pred_jigsaw.shape)
            # print("pred_rot", pred_rot.shape)

            jigsaw_out = torch.full((x.shape[0], self.num_patches, self.num_patches), -1.0, device=x.device, dtype=pred_jigsaw.dtype)
            rot_out = torch.full((x.shape[0], self.num_patches, 4), -1.0, device=x.device, dtype=pred_rot.dtype)

            # print("jigsaw_out", jigsaw_out.shape)
            # print("rot_out", rot_out.shape)
            # print("ids_used", ids_used.shape)


            # Use scatter_ to fill in the jigsaw_out and rot_out tensors
            jigsaw_out.scatter_(1, ids_used.unsqueeze(2).expand(-1, -1, self.num_patches), pred_jigsaw)
            rot_out.scatter_(1, ids_used.unsqueeze(2).expand(-1, -1, 4), pred_rot)

            outs.pred_jigsaw=jigsaw_out
            outs.pred_rot=rot_out
            outs.ids_masked=ids_masked
            outs.ids_used=ids_used

        return outs


@register_model
def jigsaw_tiny_patch16_224(mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio, use_jigsaw=use_jigsaw,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def jigsaw_small_patch16_224(mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio, use_jigsaw=use_jigsaw,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def jigsaw_base_patch16_224(mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio, use_jigsaw=use_jigsaw,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

if __name__ == '__main__':
    net = jigsaw_base_patch16_224(mask_ratio=0.5, use_jigsaw=True, pretrained=False)
    net = net.cuda()
    img = torch.cuda.FloatTensor(6, 3, 224, 224)
    with torch.no_grad():
        outs = net(img)