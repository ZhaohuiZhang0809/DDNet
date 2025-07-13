import torch
import torch.nn.functional as F
from thop import profile
from torch import nn
from torchinfo import summary

from models.configs.CrossMamba import CrossMamba
from models.common import Laplace
from models.configs.FreqMamba import FreqMamba


class AdaptiveChannelSelection(nn.Module):
    def __init__(self, dim=96, ratio=4):
        super(AdaptiveChannelSelection, self).__init__()
        self.Laplace = Laplace(dim=dim)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Define Gaussian kernel parameters
        self.gaussian_kernel = self._get_gaussian_kernel(3, 1.0)
        self.gaussian_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self._init_gaussian_weights()

        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim * 2 // ratio, bias=False),  # Increased input dim for concatenated features
            nn.ReLU(),
            nn.Linear(dim * 2 // ratio, dim, bias=False),  # Output remains dim to match original
            nn.Softmax(dim=-1)
        )

    def _get_gaussian_kernel(self, kernel_size=3, sigma=1.0):
        """Create 2D Gaussian kernel"""
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float)
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        return kernel / kernel.sum()

    def _init_gaussian_weights(self):
        """Initialize convolution weights with Gaussian kernel"""
        with torch.no_grad():
            self.gaussian_conv.weight.data = self.gaussian_kernel.repeat(self.gaussian_conv.out_channels, 1, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Laplace branch
        x_h = self.Laplace(x)
        # Gaussian branch
        x_l = self.gaussian_conv(x)

        c_x = torch.cat([x_l, x_h], dim=1)
        c_w = self.gap(c_x).view(b, c * 2)
        c_w = self.fc(c_w).view(b, c, 1, 1)
        x = x * c_w + x

        return x


class AFEM(nn.Module):
    def __init__(self, dim=96,  mlp_ratio=4.0, d_state=32, force_fp32=True):
        super(AFEM, self).__init__()
        self.FreMamba = FreqMamba(dim=dim, mlp_ratio=mlp_ratio, d_state=d_state, force_fp32=force_fp32)
        self.CMM_1 = CrossMamba(dim1=dim // 4 * 3, dim2=dim // 4, mlp_ratio=mlp_ratio, d_state=d_state, force_fp32=force_fp32)
        self.CSM = AdaptiveChannelSelection(dim=dim // 4 * 3)
        self.CMM_2 = CrossMamba(dim1=dim // 4, dim2=dim // 4 * 3, mlp_ratio=mlp_ratio, d_state=d_state, force_fp32=force_fp32)

    def forward(self, x):
        lx, hx = self.FreMamba(x)
        hx = self.CMM_1(hx, lx) + hx
        hx = self.CSM(hx)
        lx = self.CMM_2(lx, hx) + lx
        out = torch.cat((lx, hx), dim=1)

        return out
