import torch
from torch import nn
import numpy as np
from typing import Callable, List, Optional
from timm.models.layers import DropPath, trunc_normal_
from functools import partial

class Attention(nn.Module):
    def __init__(self, dim, num_heads: int = 8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_mask=None):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(in_features=dim, out_features=dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x): 
        B, N, C = x.shape # B = Batch size, N = Number of Tokens, C = Channels or Feature Dimensions
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) # (B, N, C) -> (B, N, 3, num_heads, head_dim) 
        qkv = (qkv.permute(2, 0, 3, 1, 4)) # (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        q = qkv[0] # (B, num_heads, N, head_dim)
        k = qkv[1] # (B, num_heads, N, head_dim)
        v = qkv[2] # (B, num_heads, N, head_dim)

        attn = q @ k.transpose(-2, -1) * self.scale  # Attention scores: (B, num_heads, N, N)
        attn = attn.softmax(dim=-1) # Softmax across tokens (last dim, N)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2) # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim)
        x = x.reshape(B, N, C) # (B, N, num_heads, head_dim) -> (B, N, C)
        x = self.proj_drop(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, in_features, out_features = None, hidden_features = None, act_layer = nn.GELU, drop = 0.0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiheadAttention(nn.MultiheadAttention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return super().forward(x, x, x, attn_mask=attn_mask)[0]

class BlockWithMasking(nn.Module):
    def __init__(self, dim, attn_target: Callable, mlp_ratio: int = 4, act_layer: Callable = nn.GELU, 
                 norm_layer: Callable = nn.LayerNorm, ffn_dropout_rate: float = 0.0, drop_path: float = 0.0, 
                 layer_scale_type: str = None, layer_scale_init_value: float = 1):
        
        super().__init__()
        assert isinstance(attn_target, nn.Module), f"attn_target should be a Callable. Otherwise attn_target is shared across blocks!"
        
        self.attn = attn_target
        if drop_path > 0.0: self.drop_path = DropPath(drop_prob=drop_path)
        else: self.drop_path = nn.Identity()

        self.norm1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,

            act_layer = act_layer,
            drop = ffn_dropout_rate
        )

        self.norm2 = norm_layer(dim)

        self.layer_scale_type = layer_scale_type
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in ["per_channel", "scalar"], f"Found layer_scale_type to be {self.layer_scale_type}; should be either `per_channel`, `scalar`"
            if self.layer_scale_type == "per_channel":
                gamma_shape = [1, 1, dim]
            else: gamma_shape = [1, 1, 1]
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size = gamma_shape) * layer_scale_init_value,
                requires_grad=True
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size = gamma_shape) * layer_scale_init_value,
                requires_grad=True
            )
        else: 
            self.layer_scale_gamma1 = nn.Identity()
            self.layer_scale_gamma2 = nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask:torch.Tensor):
        x = self.norm1(x)
        if self.layer_scale_type is not None:
            if isinstance(self.attn, MultiheadAttention): # torch's MultiheadAttention processes (N, B, C)
                x = self.attn(x.permute(1,0,2), attn_mask)
                x = x.permute(1,0,2)
            else: 
                x = self.attn(x, attn_mask)

            x = x + self.drop_path(x)
            x = (x * self.layer_scale_gamma1)

            x = self.norm2(x)
            x = self.mlp(x)
            x = self.drop_path(x)
            x = x * self.layer_scale_gamma2
            return x
        
        if isinstance(self.attn, MultiheadAttention): # torch's MultiheadAttention processes (N, B, C)
            x = self.attn(x.permute(1,0,2), attn_mask)
            x = x.permute(1,0,2)
        else: 
            x = self.attn(x, attn_mask)

        x = x + self.drop_path(x)

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        return x

_LAYER_NORM = partial(nn.LayerNorm, eps = 1e-6)

class SimpleTransformer(nn.Module):
    def __init__(self, attn_target: Callable, embed_dim: int, num_blocks: int, block: Callable = BlockWithMasking,
                 pre_transformer_layer: Optional[Callable] = None, post_transformer_layer: Optional[Callable] = None,
                 drop_path_rate: float = 0.0, drop_path_type: str = "progressive", norm_layer: Callable = _LAYER_NORM,
                 mlp_ratio: int = 4, ffn_dropout_rate: float = 0.0, layer_scale_type: Optional[str] = None,
                 layer_scale_init_value: float = 1e-4, weight_init_style="jax"):
        
        super().__init__()
        
        self.pre_transformer_layer = pre_transformer_layer
        self.post_transformer_layer = post_transformer_layer
        self.weight_init_style = weight_init_style

        if drop_path_type == "progressive":
            drop_path_rate_values = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        elif drop_path_type == "uniform":
            drop_path_rate_values = [drop_path_rate for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown `drop_path_rate`, make sure it is either `progressive` or `uniform`")
        
        self.blocks = nn.Sequential(
            *[block(
                dim = embed_dim, 
                attn_target=attn_target, 
                mlp_ratio=mlp_ratio,
                ffn_dropout_rate = ffn_dropout_rate,
                drop_path=drop_path_rate_values[i],
                norm_layer = norm_layer,
                layer_scale_init_value=layer_scale_init_value,
                layer_scale_type=layer_scale_type
            ) for i in range(num_blocks)]
        )

        # self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init_style == "jax":                   # Based on MAE and official Jax ViT implementation
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.weight_init_style == "pytorch":
                trunc_normal_(m.weight, std=0.02)                 # PyTorch ViT uses trunc_normal_

            # if m.bias is not None:
            #     torch.nn.init.constant_(m.bias, 0)
            #     torch.nn.init.constant_(m.weight, 1.)
    
    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor = None, use_checkpoint: bool = False, 
                checkpoint_every_n: int = 1, checkpoint_block_ids: Optional[List[int]] = None):
        """
        Inputs:
            - tokens: shape B x N x C
                             B = Batch size, N = Number of Tokens, C = Channels or Feature Dimensions
            - attn_mask: (NxN)

        Outputs:
            - x: data of shape B x N x C
        """

        if self.pre_transformer_layer: tokens = self.pre_transformer_layer(tokens)

        if use_checkpoint and checkpoint_block_ids is None:
            checkpoint_block_ids = [i for i in range(len(self.blocks)) if i % checkpoint_every_n == 0]
        if checkpoint_block_ids is not None:
            checkpoint_block_ids = set(checkpoint_block_ids)

        for block_id, block in enumerate(self.blocks):
            if use_checkpoint and block_id in checkpoint_block_ids:
                torch.utils.checkpoint(
                    block, tokens, attn_mask, use_reentrant=False
                )
            else:
                tokens = block(tokens, attn_mask)
        
        if self.post_transformer_layer: tokens = self.post_transformer_layer(tokens)
        return tokens