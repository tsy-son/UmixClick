import torch
from torch import nn
import math


class Channel_Attention_Module_Conv(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(Channel_Attention_Module_Conv, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x)
        max_x = self.max_pooling(x)
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(avg_out + max_out)
        return x * v


class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size=(k, k), stride=(1, 1), padding=((k - 1) // 2, (k - 1) // 2),
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim=1, keepdim=True)
        max_x, _ = self.max_pooling(x, dim=1, keepdim=True)
        v = self.conv(torch.cat((max_x, avg_x), dim=1))
        v = self.sigmoid(v)
        return x * v


class CBAMBlock(nn.Module):
    def __init__(self, channels: int = None, spatial_attention_kernel_size=5):
        super(CBAMBlock, self).__init__()
        self.channel_attention_block = Channel_Attention_Module_Conv(channels=channels, gamma=2, b=1)
        self.spatial_attention_block = Spatial_Attention_Module(k=spatial_attention_kernel_size)

    def forward(self, x):
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x

# if __name__ == "__main__":
#     feature_maps = torch.randn((8, 54, 32, 32))
#     model = CBAMBlock("FC", 5, channels=54, ratio=9)
#     model(feature_maps)
#     model = CBAMBlock("Conv", 5, channels=54, gamma=2, b=1)
#     model(feature_maps)
