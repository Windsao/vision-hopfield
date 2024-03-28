import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import math
from math import sqrt
from src.entmax import EntmaxAlpha
from src.softmax_1 import Softmax_1
from src.sparse_max import Sparsemax
from torch.jit import Final
import os

_EXPORTABLE = False


_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if 'TIMM_FUSED_ATTN' in os.environ:
    _USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])
else:
    _USE_FUSED_ATTN = 1  # 0 == off, 1 == on (for tested use), 2 == on (for experimental use)

def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, softmax = None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
        
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = softmax(attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


# class Hopfield(nn.Module):
#     fused_attn: Final[bool]

#     def __init__(
#             self,
#             dim: int,
#             num_heads: int = 8,
#             qkv_bias: bool = False,
#             attn_drop: float = 0.,
#             proj_drop: float = 0.,
#             qk_scale=None,
#             mode = 'softmax_1',
#             step_size: int = 1,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = qk_scale or self.head_dim ** -0.5
#         self.fused_attn = False
#         self.step_size = step_size

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
#         self.kernel_1 = nn.Linear(dim, dim)
#         # self.kernel_2 = nn.Linear(dim, dim)
        
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         if mode == 'sparsemax':
#             self.softmax = Sparsemax(dim=-1)
#         elif mode == 'entmax':
#             self.softmax = EntmaxAlpha(dim=-1)
#         elif mode == 'softmax':
#             self.softmax = nn.Softmax(dim=-1)
#         elif mode == 'softmax_1':
#             self.softmax = Softmax_1(dim=-1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         B, N, C = x.shape
#         q, k = self.q(self.kernel_1(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), self.k(self.kernel_1(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         # v = k
        
#         for _ in range(self.step_size):

#             if self.fused_attn:
#                 x = scaled_dot_product_attention(
#                     q, k, k,
#                     dropout_p=self.attn_drop.p if self.training else 0.,
#                     softmax= self.softmax
#                 )
#             else:
#                 q = q * self.scale
#                 attn = q @ k.transpose(-2, -1)
#                 attn = self.softmax(attn)
#                 attn = self.attn_drop(attn)
#                 x = attn @ k
                
#             q = x
            
#             #print("x shape", x.shape)
#         q = q.transpose(1, 2).reshape(B, N, C)
#         x = self.v(q)    
#         #x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

class HopfieldPooling(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_scale=None,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            num_pattern: int = 1,
            mode = 'softmax_1',
            step_size: int = 1
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        #self.fused_attn = use_fused_attn()
        self.fused_attn = False
        self.step_size = step_size

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if mode == 'sparsemax':
            self.softmax = Sparsemax(dim=-1)
        elif mode == 'entmax':
            self.softmax = EntmaxAlpha(dim=-1)
        elif mode == 'softmax':
            self.softmax = nn.Softmax(dim=-1)
        elif mode == 'softmax_1':
            self.softmax = Softmax_1(dim=-1)
        pooling_weight_size = dim

        self.query = nn.Parameter(
            torch.randn(
                size=(
                    *
                    (
                        (1,
                         num_pattern)),
                    dim if pooling_weight_size is None else pooling_weight_size), dtype=torch.float32),
            requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = self.query.shape
        B, N, C = x.shape
        H = self.n_heads
        queries = self.query.repeat((*((B, 1)), 1))
        
        q, k = self.q(queries).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # v = k

        
        
        for _ in range(self.step_size):

            if self.fused_attn:
                x = scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    softmax= self.softmax
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = self.softmax(attn)
                attn = self.attn_drop(attn)
                x = attn @ v
            q = x
            
            #print("x shape", x.shape)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.v(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HopfieldLayer(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_scale=None,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            mode = 'softmax_1',
            step_size: int = 1
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False
        self.step_size = step_size

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if mode == 'sparsemax':
            self.softmax = Sparsemax(dim=-1)
        elif mode == 'entmax':
            self.softmax = EntmaxAlpha(dim=-1)
        elif mode == 'softmax':
            self.softmax = nn.Softmax(dim=-1)
        elif mode == 'softmax_1':
            self.softmax = Softmax_1(dim=-1)
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, N, C = x.shape
        q, k = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = k

        q, k = self.q_norm(q), self.k_norm(k)
        
        for _ in range(self.step_size):

            if self.fused_attn:
                x = scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    softmax= self.softmax
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = self.softmax(attn)
                attn = self.attn_drop(attn)
                x = attn @ v
            q = x
            
            #print("x shape", x.shape)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.v(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Association(nn.Module):
    '''
    The Hopfield operation
    '''

    def __init__(self, scale=None, attention_dropout=0.0, mode='softmax', norm=False):
        super(Association, self).__init__()
        self.scale = scale
        self.norm = norm
        self.dropout = nn.Dropout(attention_dropout)
        if mode == 'sparsemax':
            self.softmax = Sparsemax(dim=-1)
        elif mode == 'softmax':
            self.softmax = nn.Softmax(dim=-1)
        elif mode == 'softmax1':
            self.softmax = Softmax_1(dim=-1)

    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.norm and H == 1:
            scores = F.normalize(scores)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, H, scores.size(-2), 1)
            scores = scores.masked_fill_(mask, float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()
    
class Hopfield(nn.Module):

    def __init__(
            self,
            dim,
            num_heads = 8,
            d_keys=None,
            d_values=None,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            qk_scale=None,
            mode='softmax',
            step_size: int = 1,
            ):
        super(Hopfield, self).__init__()
        d_model = dim
        n_heads = num_heads
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.inner_attention = Association(
            scale=qk_scale, attention_dropout=attn_drop, mode=mode)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.update_steps = step_size

    def forward(self, x, mask=None):
        R, Y = x[:, 0].unsqueeze(dim=1), x[:, 1:]
        
        B, L, _ = R.shape
        _, S, _ = Y.shape
        H = self.n_heads

        queries = self.query_projection(R).view(B, L, H, -1)
        keys = self.key_projection(Y)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for i in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)

        return self.out_projection(out)
    
    
    
