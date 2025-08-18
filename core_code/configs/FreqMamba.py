import torch
from einops import rearrange
import math
from thop import profile
from torch import nn
from torchinfo import summary

from models.configs.MambaScanner import mamba_init, MambaScanner
from models.common import Dwt2d, DoubleConv


class FreqSS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
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
        super(FreqSS2D, self).__init__()
        r""" V-Mamba-v0 Framework
        Args:
            d_model: Output dimension of the model (default: 96)
            d_state: State dimension (default: 16)
            ssm_ratio: Ratio of state dimension to model dimension (default: 2.0)
            dt_rank: Dimension of dynamic time parameters, defaults to "auto" which calculates based on d_model
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
        k_group = 4     # Scanning direction
        self.k_group = k_group
        self.dim = d_model

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # self.selective_scan = selective_scan_fn  # Selective scan (accelerated)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
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
        self.mambaScanner = MambaScanner(seq=seq, force_fp32=force_fp32, init_dt_A_D=init_dt_A_D,
                                         x_proj_weight=self.x_proj_weight)

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        # self.out_proj = nn.Conv2d(d_inner * 2, d_model, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()


        # adaptive parameter
        # self.gamma1 = nn.Parameter(torch.ones((d_model)), requires_grad=True)
        # self.gamma2 = nn.Parameter(torch.ones((d_model)), requires_grad=True)


    def forward(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
        # z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        B, D, H, W = x.shape
        L = H * W

        """ Traversal path from ll -> hh """
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        # Concatenate x_hwwh with its flipped version
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        out_y = self.mambaScanner(xs)

        # Restore token positions
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # Superposition of four states
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        # Restore shape for output
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)

        # Normalized output
        y = self.out_norm(y).view(B, H, W, -1)
        # z acts as a gating mechanism (SiLU activation branch)
        # y = y * z
        y = y + z
        # y = torch.cat([y, z],dim=-1)
        # y = rearrange(y, "b h w c -> b c h w")

        out = self.dropout(self.out_proj(y))
        out = rearrange(out, 'b h w c -> b c h w')

        return out


class FreqMamba(nn.Module):
    def __init__(self, dim=96,  mlp_ratio=4.0, d_state=16, force_fp32=True):
        super(FreqMamba, self).__init__()
        self.Dwt2d = Dwt2d()
        self.FreqSS2D = FreqSS2D(d_model=dim // 4, mlp_ratio=mlp_ratio, d_state=d_state, force_fp32=force_fp32)

    def forward(self, x):
        ll, lh, hl, hh = torch.chunk(x, 4, dim=1)                       # b,c,h,w
        # Concatenate ll, lh, hl, hh
        wx = torch.cat([ll, lh], dim=3)
        wx = torch.cat([wx, torch.cat([hl, hh], dim=3)], dim=2)  # b,c,2h,2w
        wx = rearrange(wx, "b c h w -> b h w c")
        wx = self.FreqSS2D(wx)

        # Crop and restore
        top, bottom = torch.chunk(wx, 2, dim=2)
        lx, wx_lh = torch.chunk(top, 2, dim=3)
        wx_hl, wx_hh = torch.chunk(bottom, 2, dim=3)

        # lx = lx + ll
        # wx_lh = wx_lh + lh
        # wx_hl = wx_hl + hh
        # wx_hh = wx_hh + hh

        hx = torch.cat([wx_lh, wx_hl, wx_hh], dim=1)

        return lx, hx


