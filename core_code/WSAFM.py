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
    """
    频域卷积模块。
    """
    def __init__(self, in_channels, out_channels):
        super(FreqConvModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        # 将输入张量转换到频域
        x_fft = torch.fft.fft2(x, dim=(2, 3))  # 对输入张量进行二维傅里叶变换
        x_fft = self.conv1(x_fft.real)
        x_fft = self.relu(x_fft.real)
        x_fft = self.conv2(x_fft.real)
        x_ifft = torch.fft.ifft2(x_fft, dim=(2, 3)).real

        return x_ifft

# 定义全局卷积模块
class GlobalConvModule(nn.Module):
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
    r""" Multi-scale Frequency Fusion Module

    Args:
    """
    def __init__(self, in_channels, out_channels):
        super(MFFM, self).__init__()
        # self.DWConv = DWConv(in_channels * 2, in_channels)
        # self.GCM = GlobalConvModule(in_channels, out_channels, (7, 7))      # local 3M proj 2M fre 4M GCM 14M

        self.Multi_Conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=3, padding=1),
        )
        # self.local_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.Freq_Conv_1 = FreqConvModule(in_channels, in_channels // 2)

        self.Multi_Conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=3, padding=1),
        )
        # self.local_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
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
    def __init__(self, in_channels, mlp_ratio=4):
        super(VotingGate, self).__init__()
        # self.voting_gate = nn.Sequential(
        #     # DWConv(in_channels * 2, in_channels),
        #     # nn.BatchNorm2d(in_channels),
        #     # nn.ReLU(inplace=True),
        #     # DWConv(in_channels, in_channels // 2),
        #     # nn.BatchNorm2d(in_channels // 2),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels * 2, in_channels // 4, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(in_channels // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.voting_gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // int(mlp_ratio), bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // int(mlp_ratio), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, fuse, sx):
        x = torch.cat([fuse, sx], dim=1)
        # out = self.voting_gate(x) * sx
        out =  self.voting_gate(self.gap(x).transpose(dim0=1, dim1=3)).transpose(dim0=1, dim1=3) * sx

        return out


# class MaskAvgPool2d(nn.Module):
#     def __init__(self,):
#         super(MaskAvgPool2d, self).__init__()
#
#     def forward(self, x, mask):
#         """
#         :param x: 输入张量，形状为 (batch_size, channels, height, width)
#         :param mask: 掩码，形状为 (batch_size, 1, height, width)，
#                      其中值为 1 表示有效区域，0 表示无效区域
#         :return: 输出张量，经过 Masked Average Pooling 处理
#         """
#         masked_x = x * mask
#
#         prototype = F.adaptive_avg_pool2d(masked_x, (1, 1))
#         return prototype


# class DPE(nn.Module):
#     r""" Dynamically enhance the target prototype
#
#     Args:
#     """
#     def __init__(self, dim, scaler=20):
#         super(DPE, self).__init__()
#
#         self.GAP = nn.AdaptiveAvgPool2d((1, 1))
#         self.MLP = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.ReLU(),
#             nn.Linear(dim * 2, dim)
#         )
#
#         self.FC = nn.Sequential(
#             nn.Linear(dim * 2, dim // 4),
#             nn.ReLU(),
#             nn.Linear(dim // 4, dim)
#         )
#
#         self.scaler = scaler
#
#         self.MAP = MaskAvgPool2d()
#
#     def Sim(self, fts, prototype, thresh):
#         """
#         Calculate the distance between features and prototypes
#
#         Args:
#             fts: input features
#                 expect shape: N x C x H x W
#             prototype: prototype of one semantic class
#                 expect shape: 1 x C
#         """
#         sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
#         coarse_mask = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
#
#         return coarse_mask
#
#     def forward(self, s_x, q_x):
#         b, c, h, w = s_x.size()
#         q_x_ = rearrange(q_x, 'b (n c) h w -> b n c h w', n=1)
#
#         s_p = self.GAP(s_x).view(b, 1, c)
#         thresh = self.MLP(rearrange(q_x, 'b c h w -> b h w c'))
#         thresh = rearrange(thresh, 'b h w c -> b c h w')
#         c_mask = self.Sim(q_x_, s_p, thresh)
#
#         q_p = self.MAP(q_x, c_mask)
#
#         ch_w = self.FC(torch.cat((s_p.view(b, c), q_p.view(b, c)), dim=1)).view(b, c, 1, 1)
#         out = s_x * ch_w + q_x * ch_w
#
#         return out



class WSAFM(nn.Module):
    def __init__(self, dim=96, mlp_ratio=4.0, d_state=32, force_fp32=True):
        super(WSAFM, self).__init__()
        self.FreqMamba = FreqMamba(dim=dim * 4, mlp_ratio=mlp_ratio, d_state=d_state, force_fp32=force_fp32)
        # self.SpaceMamba = SpaceMamba(hidden_dim=dim, mlp_ratio=mlp_ratio, ssm_d_state=d_state, ssm_init="v0",
        #     forward_type="v1", ssm_ratio = 1.0)

        self.dwt = Dwt2d()
        self.iwt = Iwt2d()
        # 双输入通道注意力
        self.MFFM_1 = MFFM(in_channels=dim, out_channels=dim)
        self.MFFM_2 = MFFM(in_channels=dim * 3, out_channels=3 * dim)

        self.VG = VotingGate(dim)

        self.shortcut = nn.Sequential()

    def forward(self, sx, wx):
        # sx = rearrange(sx, 'b c h w -> b h w c')
        # sx = self.SpaceMamba(sx)
        # sx = rearrange(sx, 'b h w c -> b c h w')
        sx_l, lh, hl, hh = self.dwt(sx)
        sx_h = torch.cat((lh, hl, hh), dim=1)

        d_wx = self.dwt(wx, separate=False)
        wx_l, wx_h = self.FreqMamba(d_wx)

        lx, hx = wx_l + sx_l, wx_h + sx_h

        # lx = self.MFFM_1(sx_l, wx_l)
        # hx = self.MFFM_2(sx_h, wx_h)

        # lx, hx  = self.FreqMamba(torch.cat((lx, hx), dim=1))

        out = self.iwt(torch.cat((lx, hx), dim=1))

        out = self.VG(out, sx)

        # out = rearrange(out, 'b c h w -> b h w c')
        # out = self.SpaceMamba(out)
        # out = rearrange(out, 'b h w c -> b c h w')

        return out




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = WSAFM(dim = 768).to(device)

    # 打印网络结构和参数
    summary(net, [(2, 768, 10, 10), (2, 768, 10, 10)])

    inputs1 = torch.randn(2, 768, 10, 10).cuda()
    inputs2 = torch.randn(2, 768, 10, 10).cuda()
    flops, params = profile(net, (inputs1, inputs2))

    print("FLOPs=, params=", flops, params)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    import time
    import numpy as np


    def calculate_fps(model, input_size, batch_size=1, num_iterations=100):
        t_all = []
        # 模型设置为评估模式
        model.eval()
        # 模拟输入数据
        input_data1 = torch.randn(batch_size, *input_size).to(device)  # 如果有GPU的话
        input_data2 = torch.randn(batch_size, *input_size).to(device)
        # 运行推理多次
        with torch.no_grad():
            for _ in range(num_iterations):
                # 启动计时器
                start_time = time.time()
                output = model(input_data1, input_data2)
                # 计算总时间
                total_time = time.time() - start_time
                t_all.append(total_time)

        print('average time:', np.mean(t_all) / 1)
        print('average fps:', 1 / np.mean(t_all))

        print('fastest time:', min(t_all) / 1)
        print('fastest fps:', 1 / min(t_all))

        print('slowest time:', max(t_all) / 1)
        print('slowest fps:', 1 / max(t_all))


    net = WSAFM(dim=768).to(device)
    calculate_fps(net, input_size=(768, 10, 10), batch_size=2, num_iterations=10)