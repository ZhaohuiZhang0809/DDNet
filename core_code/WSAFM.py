import torch.nn.functional as F
import torch.fft
import torch
from einops import rearrange
from thop import profile
from torch import nn
from torchinfo import summary
from models.common import DWConv
from models.common import Dwt2d, Iwt2d
from models.AFEM import FreqMamba
from models.LASSM_Encoder import VSSBlock as SpaceMamba

class FreqConvModule(nn.Module):
    """Frequency Domain Convolution Module"""
    def __init__(self, in_channels, out_channels):
        super(FreqConvModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        # Convert input tensor to frequency domain
        x_fft = torch.fft.fft2(x, dim=(2, 3))  # 2D Fourier transform on input tensor
        x_fft = self.conv1(x_fft.real)
        x_fft = self.relu(x_fft.real)
        x_fft = self.conv2(x_fft.real)
        x_ifft = torch.fft.ifft2(x_fft, dim=(2, 3)).real

        return x_ifft

class GlobalConvModule(nn.Module):
    """Global Convolution Module"""
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x

class MFFM(nn.Module):
    """Multi-scale Frequency Fusion Module"""
    def __init__(self, in_channels, out_channels):
        super(MFFM, self).__init__()
        self.Multi_Conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=3, padding=1),
        )
        self.Freq_Conv_1 = FreqConvModule(in_channels, in_channels // 2)

        self.Multi_Conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=3, padding=1),
        )
        self.Freq_Conv_2 = FreqConvModule(in_channels, in_channels // 2)

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, sx, fx):
        x1 = self.Multi_Conv_1(sx) + self.Freq_Conv_1(sx)
        x2 = self.Multi_Conv_2(fx) + self.Freq_Conv_2(fx)

        x = torch.cat([x1, x2], dim=1)
        x = self.norm(x)
        x = self.relu(x)
        output = self.proj(x)

        return output

class VotingGate(nn.Module):
    """Voting Gate for feature fusion"""
    def __init__(self, in_channels, mlp_ratio=4):
        super(VotingGate, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.voting_gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // int(mlp_ratio), bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // int(mlp_ratio), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, fuse, sx):
        x = torch.cat([fuse, sx], dim=1)
        out = self.voting_gate(self.gap(x).transpose(dim0=1, dim1=3)).transpose(dim0=1, dim1=3) * sx
        return out

class WSAFM(nn.Module):
    """Wavelet and Spatial Attention Fusion Module"""
    def __init__(self, dim=96, mlp_ratio=4.0, d_state=32, force_fp32=True):
        super(WSAFM, self).__init__()
        self.FreqMamba = FreqMamba(dim=dim * 4, mlp_ratio=mlp_ratio, d_state=d_state, force_fp32=force_fp32)
        self.dwt = Dwt2d()
        self.iwt = Iwt2d()
        self.MFFM_1 = MFFM(in_channels=dim, out_channels=dim)
        self.MFFM_2 = MFFM(in_channels=dim * 3, out_channels=3 * dim)
        self.VG = VotingGate(dim)
        self.shortcut = nn.Sequential()

    def forward(self, sx, wx):
        # Spatial branch processing
        sx_l, lh, hl, hh = self.dwt(sx)
        sx_h = torch.cat((lh, hl, hh), dim=1)

        # Frequency branch processing
        d_wx = self.dwt(wx, separate=False)
        wx_l, wx_h = self.FreqMamba(d_wx)

        # Multi-scale fusion
        lx = self.MFFM_1(sx_l, wx_l)
        hx = self.MFFM_2(sx_h, wx_h)

        # Inverse wavelet transform
        out = self.iwt(torch.cat((lx, hx), dim=1))

        # Voting gate fusion
        out = self.VG(out, sx)
        # out = rearrange(out, 'b c h w -> b h w c')
        # out = self.SpaceMamba(out)
        # out = rearrange(out, 'b h w c -> b c h w')

        return out
