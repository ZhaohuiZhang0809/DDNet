import torch
from einops import rearrange
from thop import profile
from torch import nn
import math

from torchinfo import summary

from comparison_models.common import DoubleConv
from models.configs.MambaScanner import mamba_init, CrossScanner


class CrossSS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model1=96,
            d_model2=96,
            mlp_ratio=4.0,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # ======================
            dropout=0.0,
            # ======================
            seq=False,
            force_fp32=True,
            **kwargs,
    ):
        super(CrossSS2D, self).__init__()
        r"""
        Arg:
            d_model: 模型的输出维度（默认为96）。
            d_state: 状态维度（默认为16）。
            ssm_ratio: 状态维度与模型维度的比率（默认为2.0）。
            dt_rank: 动态时间参数的维度，默认为“auto”，会根据 d_model 计算
        """

        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4     # 扫描方向
        self.k_group = k_group

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model1)
        dt_rank = math.ceil(d_model1 / 16) if dt_rank == "auto" else dt_rank

        # in proj ============================
        self.in_proj1 = nn.Linear(d_model1, d_inner * 2, bias=bias)
        # in proj ============================
        self.in_proj2 = nn.Linear(d_model2, d_inner, bias=bias)

        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        init_dt_A_D = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=k_group,
        )

        # batch, length, force_fp32, seq, k_group, inner, rank
        self.mambaScanner = CrossScanner(seq=seq, force_fp32=force_fp32, init_dt_A_D=init_dt_A_D,
                                         x_proj_weight=self.x_proj_weight)

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model1, bias=bias)
        # self.out_proj = nn.Conv2d(d_inner * 2, d_model1, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()


    def forward(self, q_x, kv_x):
        q_x = self.in_proj1(q_x)
        q_x, z = q_x.chunk(2, dim=-1)  # (b, h, w, d)
        # z = self.act(z)
        q_x = q_x.permute(0, 3, 1, 2).contiguous()
        q_x = self.conv2d(q_x)  # (b, d, h, w)
        q_x = self.act(q_x)

        kv_x = self.in_proj2(kv_x)
        kv_x = kv_x.permute(0, 3, 1, 2).contiguous()
        kv_x = self.conv2d(kv_x)  # (b, d, h, w)
        kv_x = self.act(kv_x)

        B, D, H, W = q_x.shape
        L = H * W

        """ 遍历路径 """
        x_hwwh_q = torch.stack([q_x.view(B, -1, L), torch.transpose(q_x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # 拼接 x_hwwh 和 其翻转
        q_x_ = torch.cat([x_hwwh_q, torch.flip(x_hwwh_q, dims=[-1])], dim=1)  # (b, k, d, l)

        x_hwwh_kv = torch.stack([kv_x.view(B, -1, L), torch.transpose(kv_x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # 拼接 x_hwwh 和 其翻转
        kv_x = torch.cat([x_hwwh_kv, torch.flip(x_hwwh_kv, dims=[-1])], dim=1)  # (b, k, d, l)

        out_y = self.mambaScanner(q_x_, kv_x)

        """ 四种遍历路径叠加 (Mamba之后) """
        # token位置还原
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # 四种状态叠加
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        # y = y * q_x.view(B, -1, L)

        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)

        # 正则输出
        y = self.out_norm(y).view(B, H, W, -1)
        # z是一个门控（SiLU激活分支）
        # y = y * z
        y = y + z
        # y = torch.cat([y, z],dim=-1)
        # y = rearrange(y, "b h w c -> b c h w")

        out = self.dropout(self.out_proj(y))
        out = rearrange(out, 'b h w c -> b c h w')

        return out



class CrossMamba(nn.Module):
    def __init__(self, dim1 = 96, dim2 = 96, mlp_ratio=4.0, d_state=16, force_fp32=True):
        super(CrossMamba, self).__init__()
        self.CrossSS2D = CrossSS2D(d_model1=dim1, d_model2=dim2, mlp_ratio=mlp_ratio, d_state=d_state, force_fp32=force_fp32)

    def forward(self, q_x, kv_x):
        q_x = rearrange(q_x, "b c h w -> b h w c")
        kv_x = rearrange(kv_x, "b c h w -> b h w c")
        out = self.CrossSS2D(q_x, kv_x)

        return out




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CrossMamba(dim1=72, dim2=24).to(device)

    # 打印网络结构和参数
    summary(net, [(2, 72, 80, 80), (2, 24, 80, 80)])

    inputs1 = torch.randn(2, 72, 80, 80).cuda()
    inputs2 = torch.randn(2, 24, 80, 80).cuda()
    flops, params = profile(net, (inputs1, inputs2))
    print("FLOPs=, params=", flops, params)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))