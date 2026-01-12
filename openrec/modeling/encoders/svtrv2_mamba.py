"""
SVTRv2 + DA-Mamba Encoder
=========================

DA-Mamba (Dynamic Adaptive Mamba)를 SVTRv2에 통합한 버전.
Global Attention을 DASSM (Dynamic Adaptive SSM)으로 교체.

빌드 필수:
    # 1. DCNv3 빌드
    cd reference/DAMamba/classification/models/ops_dcnv3
    python setup.py build_ext --inplace
    
    # 2. Selective Scan 빌드
    cd reference/DAMamba/classification/models/selective_scan  
    python setup.py build_ext --inplace

사용법:
    Encoder:
      name: SVTRv2Mamba
      mixer: [['Conv']*6, ['Conv','Conv','FMamba','Mamba','Mamba','Mamba'], ['Mamba']*6]
"""

import math
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_, constant_
from einops import repeat

from openrec.modeling.common import DropPath, Identity, Mlp

# DA-Mamba 의존성 경로 추가
DAMAMBA_PATH = os.path.join(os.path.dirname(__file__), '../../../reference/DAMamba/classification/models')
if DAMAMBA_PATH not in sys.path:
    sys.path.insert(0, DAMAMBA_PATH)

# 필수 의존성 import
from ops_dcnv3.functions import DCNv3Function
from utils import selective_scan_fn


# =============================================================================
# DA-Mamba 원본에서 가져온 유틸리티
# =============================================================================

class to_channels_first(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim, norm_layer, in_format='channels_last', out_format='channels_last', eps=1e-6):
    """DA-Mamba 원본의 norm layer 빌더"""
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    """DA-Mamba 원본의 activation 빌더"""
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class LayerNorm2d(nn.Module):
    """2D LayerNorm (DA-Mamba 원본)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        if self.training:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
        return x


class ResDWC(nn.Module):
    """Residual Depthwise Conv (DA-Mamba 원본)"""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)
        a = torch.zeros(kernel_size ** 2)
        a[kernel_size ** 2 // 2] = 1.  # center = 1
        self.conv_constant = nn.Parameter(a.reshape(1, 1, kernel_size, kernel_size))
        self.conv_constant.requires_grad = False

    def forward(self, x):
        return F.conv2d(x, self.conv.weight + self.conv_constant, self.conv.bias, 
                        stride=1, padding=self.kernel_size // 2, groups=self.dim)


class ConvFFN(nn.Module):
    """Conv-based FFN (DA-Mamba 원본)"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.conv = ResDWC(hidden_features, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# DA-Mamba 핵심 모듈 (원본 그대로)
# =============================================================================

class Dynamic_Adaptive_Scan(nn.Module):
    """
    Dynamic Adaptive Scan (DA-Mamba 원본)
    DCNv3 기반 적응적 스캔
    """
    def __init__(
            self,
            channels=64,
            kernel_size=1,
            dw_kernel_size=3,
            stride=1,
            pad=0,
            dilation=1,
            group=1,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
    ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        # DW Conv + Norm + Act
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=dw_kernel_size, stride=1, 
                      padding=(dw_kernel_size - 1) // 2, groups=channels),
            build_norm_layer(channels, norm_layer, 'channels_first', 'channels_last'),
            build_act_layer(act_layer)
        )
        
        # Offset prediction
        self.offset = nn.Linear(channels, group * (kernel_size * kernel_size - self.remove_center) * 2)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group))

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)

    def forward(self, input, x):
        """
        Args:
            input: (B, C, H, W) - offset 계산용
            x: (B, H, W, C) - DCNv3 적용 대상
        Returns:
            (B, C, H, W)
        """
        N, _, H, W = input.shape
        x_proj = x
        
        # Offset 계산
        x1 = self.dw_conv(input)  # (B, H, W, C) - channels_last
        offset = self.offset(x1)  # (B, H, W, group*k*k*2)
        
        # DCNv3 적용
        mask = torch.ones(N, H, W, self.group, device=x.device, dtype=x.dtype)
        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center
        )

        # Center feature scale (optional)
        if self.center_feature_scale:
            center_feature_scale = F.linear(
                x1, 
                weight=self.center_feature_scale_proj_weight,
                bias=self.center_feature_scale_proj_bias
            ).sigmoid()
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        return x.permute(0, 3, 1, 2).contiguous()


class DASSM(nn.Module):
    """
    Dynamic Adaptive State Space Model (DA-Mamba 원본)
    
    expand=1 기본: d_inner = d_model (채널 확장 없음)
    """
    def __init__(
        self,
        d_model,
        head_dim=16,
        d_state=1,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection
        self.in_proj = nn.Conv2d(self.d_model, self.d_inner, 1, bias=bias, **factory_kwargs)

        # Local conv
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # SSM projections
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, 
                                      dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        # A, D 파라미터
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        # Selective scan
        self.selective_scan = selective_scan_fn

        # Output
        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, 1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Dynamic Adaptive Scan
        num_group = d_model // head_dim
        self.da_scan = Dynamic_Adaptive_Scan(channels=self.d_inner, group=num_group)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, 
                dt_max=0.1, dt_init_floor=1e-4, bias=True, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        D = torch.ones(d_inner, device=device)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def ssm(self, x: torch.Tensor):
        """State Space Model forward"""
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)

        # B, C, dt 계산
        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        # Selective scan
        h = self.selective_scan(
            xs, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        h = h.reshape(B, C, H * W)

        # Output
        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H, W = x.shape
        input = x  # expand=1이면 C == d_inner
        
        x = self.in_proj(x)
        x = self.act(self.conv2d(x))

        x = self.da_scan(input, x.permute(0, 2, 3, 1).contiguous())
        y = self.ssm(x)
        y = y.reshape(B, C, H, W)

        y = self.out_norm(y)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class DASSMBlock(nn.Module):
    """
    DASSM Block - Global Attention Block 대체용 (DA-Mamba 원본 Block 기반)
    """
    def __init__(
        self,
        dim,
        head_dim=16,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        **kwargs
    ):
        super().__init__()
        
        # Position embedding
        self.pos_embed = ResDWC(dim, 3)
        
        # Token mixer
        self.norm1 = LayerNorm2d(dim)
        self.token_mixer = DASSM(dim, head_dim=head_dim, dropout=drop)
        
        # FFN
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = ConvFFN(dim, int(mlp_ratio * dim), act_layer=act_layer, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        x = self.pos_embed(x)
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# SVTRv2 기존 컴포넌트
# =============================================================================

class ConvBNLayer(nn.Module):
    """Conv2d + BatchNorm + Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=0, bias=False, groups=1, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Attention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Attention Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, eps=1e-6):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class ConvBlock(nn.Module):
    """Conv 기반 Local Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1e-6, num_conv=2, kernel_size=3):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = nn.Sequential(*[
            nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=num_heads)
            for _ in range(num_conv)
        ]) if num_conv > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        C, H, W = x.shape[1:]
        x = x + self.drop_path(self.mixer(x))
        x = self.norm1(x.flatten(2).transpose(1, 2))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x


class FlattenTranspose(nn.Module):
    """4D -> 3D: (B, C, H, W) -> (B, H*W, C)"""
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


class FlattenBlockRe2D(Block):
    """4D 입력을 3D로 변환 후 Attention, 다시 4D"""
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


# =============================================================================
# Subsample
# =============================================================================

class SubSample1D(nn.Module):
    """3D 입력 다운샘플링"""
    def __init__(self, in_channels, out_channels, stride=[2, 1]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [H, W]


class SubSample2DTo2D(nn.Module):
    """2D -> 2D 다운샘플링"""
    def __init__(self, in_channels, out_channels, stride=[2, 1]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x, sz):
        x = self.conv(x)
        x = self.norm(x)
        H, W = x.shape[2:]
        return x, [H, W]


class IdentitySize(nn.Module):
    def forward(self, x, sz):
        return x, sz


# =============================================================================
# SVTRStage
# =============================================================================

class SVTRStage(nn.Module):
    """
    SVTRv2 Stage with Mamba support
    
    Mixer 타입:
    - 'Conv': ConvBlock (local)
    - 'Global': Attention Block (3D)
    - 'FGlobal': Flatten + Attention Block
    - 'Mamba': DASSMBlock (2D 유지)
    - 'FMamba': Conv->Mamba 전환점
    """
    def __init__(self, dim=64, out_dim=256, depth=3, mixer=['Local'] * 3,
                 kernel_sizes=[3] * 3, sub_k=[2, 1], num_heads=2, head_dim=16,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path=[0.1] * 3, norm_layer=nn.LayerNorm,
                 act=nn.GELU, eps=1e-6, num_conv=[2] * 3, downsample=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.mixer_types = mixer
        
        self.blocks = nn.Sequential()
        for i in range(depth):
            if mixer[i] == 'Conv':
                self.blocks.append(ConvBlock(
                    dim=dim, kernel_size=kernel_sizes[i], num_heads=num_heads,
                    mlp_ratio=mlp_ratio, drop=drop_rate, act_layer=act,
                    drop_path=drop_path[i], norm_layer=norm_layer, eps=eps,
                    num_conv=num_conv[i]))
            
            elif mixer[i] in ['Mamba', 'FMamba']:
                self.blocks.append(DASSMBlock(
                    dim=dim, head_dim=head_dim, mlp_ratio=mlp_ratio,
                    drop=drop_rate, drop_path=drop_path[i], act_layer=act))
            
            elif mixer[i] == 'Global':
                self.blocks.append(Block(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    act_layer=act, attn_drop=attn_drop_rate, drop_path=drop_path[i],
                    norm_layer=norm_layer, eps=eps))
            
            elif mixer[i] == 'FGlobal':
                self.blocks.append(FlattenTranspose())
                self.blocks.append(Block(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    act_layer=act, attn_drop=attn_drop_rate, drop_path=drop_path[i],
                    norm_layer=norm_layer, eps=eps))
            
            elif mixer[i] == 'FGlobalRe2D':
                self.blocks.append(FlattenBlockRe2D(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    act_layer=act, attn_drop=attn_drop_rate, drop_path=drop_path[i],
                    norm_layer=norm_layer, eps=eps))

        # Downsample
        if downsample:
            last_mixer = mixer[-1]
            if last_mixer in ['Conv', 'FGlobalRe2D', 'Mamba', 'FMamba']:
                self.downsample = SubSample2DTo2D(dim, out_dim, stride=sub_k)
            else:
                self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward(self, x, sz):
        for blk in self.blocks:
            x = blk(x)
        x, sz = self.downsample(x, sz)
        return x, sz


# =============================================================================
# Patch Embedding
# =============================================================================

class ADDPosEmbed(nn.Module):
    """학습 가능한 Position Embedding"""
    def __init__(self, feat_max_size=[8, 32], embed_dim=768):
        super().__init__()
        pos_embed = torch.zeros([1, feat_max_size[0] * feat_max_size[1], embed_dim], dtype=torch.float32)
        trunc_normal_(pos_embed, mean=0, std=0.02)
        self.pos_embed = nn.Parameter(
            pos_embed.transpose(1, 2).reshape(1, embed_dim, feat_max_size[0], feat_max_size[1]),
            requires_grad=True)

    def forward(self, x):
        sz = x.shape[2:]
        return x + self.pos_embed[:, :, :sz[0], :sz[1]]


class POPatchEmbed(nn.Module):
    """Patch-Overlapping Patch Embedding"""
    def __init__(self, in_channels=3, feat_max_size=[8, 32], embed_dim=768,
                 use_pos_embed=False, flatten=False, bias=False):
        super().__init__()
        self.patch_embed = nn.Sequential(
            ConvBNLayer(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1, act=nn.GELU, bias=bias),
            ConvBNLayer(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, act=nn.GELU, bias=bias),
        )
        if use_pos_embed:
            self.patch_embed.append(ADDPosEmbed(feat_max_size, embed_dim))
        if flatten:
            self.patch_embed.append(FlattenTranspose())

    def forward(self, x):
        sz = x.shape[2:]
        x = self.patch_embed(x)
        return x, [sz[0] // 4, sz[1] // 4]


class LastStage(nn.Module):
    """최종 출력 stage"""
    def __init__(self, in_channels, out_channels, last_drop, out_char_num=0):
        super().__init__()
        self.last_conv = nn.Linear(in_channels, out_channels, bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop)

    def forward(self, x, sz):
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
        else:
            x = x.reshape(-1, sz[0], sz[1], x.shape[-1])
        x = x.mean(1)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        return x, [1, sz[1]]


class Feat2D(nn.Module):
    """3D -> 2D 변환"""
    def forward(self, x, sz):
        if x.dim() == 3:
            C = x.shape[-1]
            x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        return x, sz


# =============================================================================
# 메인 모델: SVTRv2Mamba
# =============================================================================

class SVTRv2Mamba(nn.Module):
    """
    SVTRv2 + DA-Mamba Encoder
    """
    def __init__(self,
                 max_sz=[32, 128],
                 in_channels=3,
                 out_channels=192,
                 depths=[3, 6, 3],
                 dims=[64, 128, 256],
                 mixer=[['Conv'] * 3, ['Conv'] * 3 + ['Global'] * 3, ['Global'] * 3],
                 use_pos_embed=True,
                 sub_k=[[1, 1], [2, 1], [1, 1]],
                 num_heads=[2, 4, 8],
                 head_dim=16,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 last_drop=0.1,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 last_stage=False,
                 feat2d=False,
                 eps=1e-6,
                 num_convs=[[2] * 3, [2] * 3 + [3] * 3, [3] * 3],
                 kernel_sizes=[[3] * 3, [3] * 3 + [3] * 3, [3] * 3],
                 pope_bias=False,
                 **kwargs):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        # Patch Embedding
        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        first_mixer = mixer[0][0] if mixer and mixer[0] else 'Conv'
        self.pope = POPatchEmbed(
            in_channels=in_channels,
            feat_max_size=feat_max_size,
            embed_dim=dims[0],
            use_pos_embed=use_pos_embed,
            flatten=(first_mixer not in ['Conv', 'Mamba', 'FMamba']),
            bias=pope_bias
        )

        # Stochastic depth
        dpr = np.linspace(0, drop_path_rate, sum(depths))

        # Stages
        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = SVTRStage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                kernel_sizes=kernel_sizes[i_stage] if len(kernel_sizes[i_stage]) == len(mixer[i_stage]) 
                             else [3] * len(mixer[i_stage]),
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
                num_conv=num_convs[i_stage] if len(num_convs[i_stage]) == len(mixer[i_stage]) 
                         else [2] * len(mixer[i_stage]),
            )
            self.stages.append(stage)

        # Output
        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.stages.append(LastStage(self.num_features, out_channels, last_drop))
        if feat2d:
            self.stages.append(Feat2D())
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_embed', 'downsample', 'pos_embed', 'A_logs', 'Ds'}

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(0, 1)
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x
