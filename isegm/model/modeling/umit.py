import warnings

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn import Sequential, Conv2d, UpsamplingBilinear2d
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import cv2
from einops import rearrange
# from models.cbam import CBAMBlock
from .cbam import CBAMBlock
from ..is_model import ISModel


# from mmcv.cnn import build_norm_layer
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # (B,32*32,C)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                            4)  # (B,32*32, C,1,64) -> (C,B,1,32*32, 64)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class FCT(nn.Module):
    def __init__(self, dim=64, decode_dim=1024, hw=128 * 128):
        super(FCT, self).__init__()
        self.dim_o = dim
        a = dim
        dim, decode_dim = hw, hw
        hw = a
        self.decode_dim = decode_dim
        self.weight_q = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k = nn.Linear(dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(hw // 2 + 1, hw // 2 + 1) * 0.02)
        self.proj = nn.Linear(hw, hw)
        self.ac_bn_2 = torch.nn.Sequential(torch.nn.ReLU(), nn.BatchNorm2d(self.dim_o))
        # self.writer = open('../../nan_check.txt', 'w')

    def forward(self, x):  # 【B，C，N】
        raw = x
        B, C, H, W = x.shape
        N = H * W
        x = x.reshape(B, C, N)  # .transpose(-2, -1) #[B, N, C]
        q = self.weight_q(x).transpose(-2, -1)  # [B，N，C] 1,16384,64
        k = self.weight_k(x).transpose(-2, -1)  # [B，N，C] 1,16384,64
        q = torch.fft.rfft2(q, dim=(-2, -1), norm='ortho')  # 1,16384,33
        k = torch.fft.rfft2(k, dim=(-2, -1), norm='ortho')  # 1,16384,33

        '''
        [B,N,C//2+1]
        '''
        q_r, q_i = q.real.transpose(-2, -1), q.imag.transpose(-2, -1)  # 1, 33,16384
        attn_r = q_r @ k.real  # [N,N] 1,33,33
        attn_i = q_i @ k.imag  # [N,N] 1,33,33
        attn_r = self.weight_alpha * attn_r  # 1025,1025  * 1,33,33
        attn_i = self.weight_alpha * attn_i
        # aa = torch.softmax(attn_r,dim=-1)
        x_r = torch.softmax(attn_r, dim=-1) @ q_i  # [B, N, C] 无softmax 95.7
        x_i = torch.softmax(attn_i, dim=-1) @ q_r  # [B, N, C]
        x = torch.view_as_complex(torch.stack([x_r, x_i], dim=-1)).transpose(-2, -1)
        x = torch.fft.irfft2(x, dim=(-2, -1), norm='ortho')
        x = self.proj(x)
        x = x.reshape(B, C, H, W)
        x = self.ac_bn_2(x)
        return raw + x


def logmax(X, axis=-1):
    X_log = torch.log(X)
    return X_log / X_log.sum(axis, keepdim=True)


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # self.fct1 = FCT(dim=64, hw=128*128)
        # self.fct2 = FCT(dim=128, hw=64*64)
        # self.fct3 = FCT(dim=320, hw=32*32)
        # self.fct4 = FCT(dim=512, hw=16*16)

    def forward_features(self, x, additional_features):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        if additional_features is not None:
            x += additional_features
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (1,64,128,128)
        # x = self.fct1(x)
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = self.fct2(x)
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = self.fct3(x)
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = self.fct4(x)
        outs.append(x)

        return outs

    def forward(self, x, additional_features):
        x = self.forward_features(x, additional_features)

        #        x = self.head(x[3])

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



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

class EEM(nn.Module):
    def __init__(self):
        super(EEM, self).__init__()
        embed_dims = [8,16,32,64]
        num_heads = [1, 1, 1]
        mlp_ratios = [4, 4, 4]
        qkv_bias = True
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        sr_ratios = [16, 16, 16]
        drop_rate = 0.0
        attn_drop_rate = 0.
        qk_scale = None
        self.layer1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,sr_ratio=sr_ratios[0])])
        self.norm1 = norm_layer(embed_dims[0])
        self.Linear1 = nn.Linear(embed_dims[0], embed_dims[1], bias=False)

        self.layer2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer, sr_ratio=sr_ratios[1])])
        self.norm2 = norm_layer(embed_dims[1])
        self.Linear2 = nn.Linear(embed_dims[1], embed_dims[2], bias=False)

        self.layer3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer, sr_ratio=sr_ratios[2])])
        self.norm3 = norm_layer(embed_dims[2])
        self.Linear3 = nn.Linear(embed_dims[2], embed_dims[3], bias=False)

    def forward(self, cannygray):
        cannygray = rearrange(cannygray, 'b c (p1 h) (p2 w)-> b (p1 p2 c) h w', p1=2, p2=2)
        B, C, H, W = cannygray.shape
        cannygray = cannygray.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.layer1):
            cannygray = blk(cannygray, H, W)
        cannygray = self.Linear1(self.norm1(cannygray).reshape(B, H, W, -1)).permute(0, 3, 1, 2)

        cannygray = cannygray.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.layer2):
            cannygray = blk(cannygray, H, W)
        cannygray = self.Linear2(self.norm2(cannygray).reshape(B, H, W, -1)).permute(0, 3, 1, 2)

        cannygray = cannygray.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.layer3):
            cannygray = blk(cannygray, H, W)
        cannygray = self.Linear3(self.norm3(cannygray).reshape(B, H, W, -1)).permute(0, 3, 1, 2)

        return cannygray

# 冻结主干网络
def fix_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

# @torch.jit.script
class MitModel(ISModel):
    def __init__(self, num_classes=1, num_channels=3):
        super(MitModel, self).__init__()
        filters = [64, 128, 320, 512]
        self.backbone = mit_b4()

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
        pretrained_dict = torch.load(r'D:\code\pretrain\mit_b4.pth')
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
