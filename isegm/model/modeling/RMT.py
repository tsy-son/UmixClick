from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from typing import List
# from ..builder import BACKBONES
# from mmcv_custom import load_checkpoint
# from mmseg.utils import get_root_logger
from typing import Tuple
import sys
import os

from isegm.model.is_model import ISModel
from isegm.model.modeling.cbam import CBAMBlock


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x
    

class RetNetRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None] #(n H*W H*W)
        return mask
    
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] #(l l)
        mask = mask.abs() #(l l)
        mask = mask * self.decay[:, None, None] #(n l l)
        return mask
    
    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0]*slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0]*slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1]) #(n l l)
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos
    
class VisionRetentionChunk(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)


        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        
        qr_w = qr.transpose(1, 2) #(b h n w d1)
        kr_w = kr.transpose(1, 2) #(b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)
        qk_mat_w = qk_mat_w + mask_w  #(b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1) #(b h n w w)
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = qr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)
        qk_mat_h = qk_mat_h + mask_h  #(b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)
        
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output
    
class VisionRetentionAll(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        
        assert h*w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = theta_shift(q, sin, cos) #(b n h w d1)
        kr = theta_shift(k, sin, cos) #(b n h w d1)

        qr = qr.flatten(2, 3) #(b n l d1)
        kr = kr.flatten(2, 3) #(b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d2)
        vr = vr.flatten(2, 3) #(b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2) #(b n l l)
        qk_mat = qk_mat + mask  #(b n l l)
        qk_mat = torch.softmax(qk_mat, -1) #(b n l l)
        output = torch.matmul(qk_mat, vr) #(b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
    
class RetBlock(nn.Module):

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:
            self.retention = VisionRetentionAll(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)

    def forward(
            self,
            x: torch.Tensor, 
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  #(b c h w)
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) #(b oh ow oc)
        return x
    
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging=None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            RetBlock(flag, embed_dim, num_heads, ffn_dim, 
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, d = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x=x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
            else:
                x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x
    
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous() #(b h w c)
        x = self.norm(x) #(b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1) #(b h w c)
        return x


class RMT(nn.Module):

    def __init__(self, in_chans=3, out_indices=(0, 1, 2, 3),
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 patch_norm=True, use_checkpoint=False, chunkwise_recurrents=[True, True, False, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6, norm_eval=True):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.norm_eval = norm_eval

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None)


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(nn.LayerNorm(embed_dims[i]))

    def forward_features(self, x, additional_features):
        x = self.patch_embed(x)
        if additional_features is not None:
            x += additional_features
        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x)
            if i in self.out_indices:
                x_out = self.extra_norms[i](x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def forward(self, x, additional_features):
        x = self.forward_features(x, additional_features)
        return x


class rmt_b4(RMT):
    def __init__(self, **kwargs):
        super(rmt_b4, self).__init__(
            embed_dims=[80, 160, 320, 512],
            depths=[4, 8, 25, 8],
            num_heads=[5, 5, 10, 16],
            init_values=[2, 2, 2, 2],
            heads_ranges=[5, 5, 6, 6],
            mlp_ratios=[4, 4, 3, 3],
            drop_path_rate=0.4,
            chunkwise_recurrents=[True, True, True, False],
            layerscales=[False, False, True, True],
            layer_init_values=1e-6,
            out_indices=(0, 1, 2, 3))

class upSamp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upSamp, self).__init__()
        self.upLiner = nn.Linear(in_channels, out_channels*4, bias=False)
        self.out_channels = out_channels

    def forward(self, feature):
        feature = feature.permute(0, 2, 3, 1).contiguous() # b c h w -> b h w c
        feature = self.upLiner(feature)
        feature = rearrange(feature, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=self.out_channels).permute(0, 3, 1, 2).contiguous()

        return feature

nonlinearity = partial(F.relu, inplace=True)

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.cbam = CBAMBlock(channel * 4)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel * 4, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
        )
        self.relu3x3 = nonlinearity

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = torch.cat((dilate1_out, dilate2_out, dilate3_out, dilate4_out), dim=1)
        out = self.cbam(out)
        out = self.conv3x3(out)
        out = self.relu3x3(out)
        # out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels//4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class MADecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(MADecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4)
        )
        self.relu1 = nonlinearity
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4)
        )
        self.relu3x3 = nonlinearity
        self.conv3x7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, (3, 7), padding=(1, 3)),
            nn.Conv2d(in_channels // 4, in_channels // 4, (7, 3), padding=(3, 1)),
            nn.BatchNorm2d(in_channels // 4)
        )
        self.relu3x7 = nonlinearity
        self.conv3x11 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, (5, 11), padding=(2, 5)),
            nn.Conv2d(in_channels // 4, in_channels // 4, (11, 5), padding=(5, 2)),
            nn.BatchNorm2d(in_channels // 4)
        )
        self.relu3x11 = nonlinearity
        self.cbam = CBAMBlock(in_channels * 2)
        self.conv3x3cat = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.relu3x3cat = nonlinearity
        self.deconv2 = nn.Sequential(
            upSamp(in_channels, n_filters),
            # nn.ConvTranspose2d(in_channels, n_filters, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(n_filters)
        )
        self.reludeconv2 = nonlinearity



    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)

        x2 = self.conv3x3(x)
        x2 = self.relu3x3(x2)

        x3 = self.conv3x7(x)
        x3 = self.relu3x7(x3)

        x4 = self.conv3x11(x)
        x4 = self.relu3x11(x4)

        x = torch.cat((x, x1, x2, x3, x4), dim=1)
        x = self.cbam(x)
        x = self.conv3x3cat(x)
        x = self.relu3x3cat(x)

        x = self.deconv2(x)
        x = self.reludeconv2(x)

        return x

class MSCABlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(MSCABlock, self).__init__()
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, 1, 2),
            nn.BatchNorm2d(in_channels)
        )
        self.relu5x5 = nonlinearity
        self.conv1x7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 7), padding=(0, 3)),
            nn.Conv2d(in_channels, in_channels, (7, 1), padding=(3, 0)),
            nn.BatchNorm2d(in_channels)
        )
        self.relu1x7 = nonlinearity
        self.conv1x11 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 11), padding=(0, 5)),
            nn.Conv2d(in_channels, in_channels, (11, 1), padding=(5, 0)),
            nn.BatchNorm2d(in_channels)
        )
        self.relu1x11 = nonlinearity
        self.conv1x21 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 21), padding=(0, 10)),
            nn.Conv2d(in_channels, in_channels, (21, 1), padding=(10, 0)),
            nn.BatchNorm2d(in_channels)
        )
        self.relu1x21 = nonlinearity

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        self.relu1x1 = nonlinearity
        self.deconv2 = nn.Sequential(
            upSamp(in_channels, n_filters),
            # nn.ConvTranspose2d(in_channels, n_filters, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(n_filters)
        )
        self.reludeconv2 = nonlinearity

    def forward(self, x):
        x1 = self.conv5x5(x)
        x1 = self.relu5x5(x1)

        x2 = self.conv1x7(x1)
        x2 = self.relu1x7(x2)

        x3 = self.conv1x11(x1)
        x3 = self.relu1x11(x3)

        x4 = self.conv1x21(x1)
        x4 = self.relu1x21(x4)

        x = x1+x2+x3+x4
        x = self.conv1x1(x)
        x = self.relu1x1(x)

        x = self.deconv2(x)
        x = self.reludeconv2(x)

        return x

class DLink_Dblock(nn.Module):
    def __init__(self, channel):
        super(DLink_Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class RMTModel(ISModel):
    def __init__(self, num_classes=1, num_channels=3):
        super(RMTModel, self).__init__()
        filters = [80, 160, 320, 512]
        self.backbone = rmt_b4()

        # fix_params(self.backbone)

        self.decoder4 = MADecoderBlock(filters[3], filters[2])
        self.decoder3 = MADecoderBlock(filters[2], filters[1])
        self.decoder2 = MADecoderBlock(filters[1], filters[0])
        self.decoder1 = MADecoderBlock(filters[0], filters[0])
        # self.decoder4 = MSCABlock(filters[3], filters[2])
        # self.decoder3 = MSCABlock(filters[2], filters[1])
        # self.decoder2 = MSCABlock(filters[1], filters[0])
        # self.decoder1 = MSCABlock(filters[0], filters[0])
        #
        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = upSamp(filters[0], 32)
        # self.finaldeconv = MADecoderBlock(filters[0], 32)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        # self.eem = EEM()
        self.dblock = Dblock(filters[3])
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        #
        self._init_weights()  # load pretrain  加载预训练时打开

    def forward(self, x, coord_features):
        # img = x[:, :3, :, :]
        # cannygray = x[:, 3:, :, :]
        # border = self.eem(cannygray)
        features = self.backbone(x, coord_features)
        e1, e2, e3, e4 = features
        e4 = self.dropout1(e4)
        e4 = self.dblock(e4)


        d4 = self.decoder4(e4) + e3
        d4 = self.dropout2(d4)
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out

    # #
    def _init_weights(self):
        # pretrained_dict = torch.load(r'D:\code\pretrain\mit_b4.pth')
        pretrained_dict = torch.load(r'E:\code\SimpleClick-1.0\weights\pretrained\RMT-B.pth')
        # pretrained_dict = torch.load(r'E:\download\ChromeDownload\upernet_swin_base_patch4_window7_512x512.pth')
        # pretrained_dict = torch.load(r'D:\Code\Data\DeepGlobe Road dataset\results\umit\model000000063.model')
        model_dict1 = self.backbone.state_dict()
        # model_dict2 = self.backbone2.state_dict()
        pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict1}
        # pretrained_dict1 = {k[9:]: v for k, v in pretrained_dict['state_dict'].items() if k[9:] in model_dict1}
        # pretrained_dict1 = {k[16:]: v for k, v in pretrained_dict.items() if k[16:] in model_dict1 and 'backbone' in k} ## zi ji xun lian de mo xing
        model_dict1.update(pretrained_dict1)
        # model_dict2.update(pretrained_dict2)
        self.backbone.load_state_dict(model_dict1)
        # self.backbone2.load_state_dict(model_dict2)
        print("successfully loaded!!!!")