
from typing import Any
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

from models.WindowsMaskingRoute import WindowsMaskingRoute
from models.common import *
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from functools import partial
import torch
from einops import rearrange
import math
from thop import profile
from torch import nn
from torchinfo import summary

from models.configs.MambaScanner import mamba_init, MambaScanner
from models.common import window_partition


# SS2Dv1, local windows
class SS2Dv1:
    def __initv1__(
            self,
            # basic dims ===========
            d_model=96,
            windows_size=4,
            topk=4,
            mlp_ratio=4.0,
            d_state=16,
            ssm_ratio=1.0,
            dt_rank="auto",
            # ======================
            dropout=0.0,
            # ======================
            seq=False,
            force_fp32=True,
            **kwargs,
    ):
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
        self.dim = d_model
        self.windows_size = windows_size

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # self.selective_scan = selective_scan_fn  # 选择性扫描（加速）

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

        self.forward = self.forwardv1
        # batch, length, force_fp32, seq, k_group, inner, rank
        self.mambaScanner = MambaScanner(seq=seq, force_fp32=force_fp32, init_dt_A_D=init_dt_A_D, x_proj_weight=self.x_proj_weight)
        # self.WMR = WindowsMaskingRoute(in_channels=d_model)

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # adaptive parameter
        self.gamma1 = nn.Parameter(torch.ones((d_model)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((d_model)), requires_grad=True)
        self.gamma3 = nn.Parameter(torch.ones((d_model)), requires_grad=True)
        self.gamma4 = nn.Parameter(torch.ones((d_model)), requires_grad=True)

        # channel transfer
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // int(mlp_ratio), bias=False),
            nn.GELU(),
            nn.Linear(d_model // int(mlp_ratio), d_model, bias=False),
            nn.Sigmoid()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))


    def forwardv1(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)

        x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        B, D, H, W = x.shape
        L = H * W

        # locality = torch.zeros((B, 1, H, W))
        # # 填充矩阵中的1
        # locality[:, :, ::2, ::3] = 1
        #
        # one_masks, zero_masks = self.WMR(x, locality)
        # x = torch.cat((one_masks, zero_masks), dim=0)
        # x = rearrange(x, '(B C H W) -> B C H W', B=B, C=D, W=W)

        """ local遍历路径 """
        # partition
        xs = window_partition(x, self.windows_size)                               # (b, local, local, c)
        trans_xs = torch.transpose(x, dim0=2, dim1=3)
        trans_xs = window_partition(trans_xs, self.windows_size)                  # (b, local, local, c)

        # xs = rearrange(xs, 'b c n m h w -> b c (n m h w)')
        # trans_xs = rearrange(trans_xs, 'b c n m h w -> b c (n m h w)')

        # # 堆叠输入张量 x 的两个视角（原始和转置）, [b, 2, d, l]

        xs = rearrange(xs, '(b n) c h w -> b c (n h w)', b=B)
        trans_xs = rearrange(trans_xs, '(b n) c h w -> b c (n h w)', b=B)
        x_hwwh = torch.stack([xs, trans_xs], dim=1).view(B, 2, -1, L)
        # 拼接 x_hwwh 和 其翻转
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        # 选择性扫描
        out_y = self.mambaScanner(xs)

        # """ 四种遍历路径叠加 (Mamba之后) """
        # token位置还原
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # 四种状态叠加
        y = self.gamma1.view(1, self.dim, 1) * out_y[:, 0] + self.gamma2.view(1, self.dim, 1) * inv_y[:, 0] + self.gamma3.view(1, self.dim, 1) * wh_y + self.gamma4.view(1, self.dim, 1) * invwh_y

        # 还原形状
        y = rearrange(y, 'b c (n m h w) -> b n m h w c', n=H // self.windows_size, h=self.windows_size, w=self.windows_size)
        y = y.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, L, -1)

        # global channel
        gc = self.gap(x).view(B, D)
        gc = self.fc(gc).view(B, 1, D)
        y = y * gc

        # 正则输出
        y = self.out_norm(y).view(B, H, W, -1)
        # z是一个门控（SiLU激活分支）
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out




# support: v0, v0seq
class SS2Dv0:
    def __initv0__(
            self,
            # basic dims ===========
            d_model=96,
            topk=4,
            mlp_ratio=4.0,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # ======================
            dropout=0.0,
            # ======================
            seq=False,
            force_fp32=True,
            windows_size=4,
            **kwargs,
    ):
        r""" V-Mamba-v0 框架
        Arg:
            d_model: 模型的输出维度（默认为96）。
            d_state: 状态维度（默认为16）。
            ssm_ratio: 状态维度与模型维度的比率（默认为2.0）。
            dt_rank: 动态时间参数的维度，默认为“auto”，会根据 d_model 计算
        """

        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        # act_layer = nn.ReLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 2     # 扫描方向
        self.k_group = k_group
        self.dim = d_model
        self.windows_size = windows_size

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # self.selective_scan = selective_scan_fn  # 选择性扫描（加速）

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

        self.WMR = WindowsMaskingRoute(in_channels=d_model, window_size=windows_size)

        # batch, length, force_fp32, seq, k_group, inner, rank
        self.mambaScanner = MambaScanner(seq=seq, force_fp32=force_fp32, init_dt_A_D=init_dt_A_D, x_proj_weight=self.x_proj_weight)


        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        # self.out_proj = nn.Conv2d(d_inner * 2, d_model, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.kv = nn.Linear(d_inner, d_inner * 2)
        # partition windows
        self.unfold = nn.Unfold(kernel_size=windows_size, stride=windows_size)

        # adaptive parameter
        # self.gamma1 = nn.Parameter(torch.ones((d_model)), requires_grad=True)
        # self.gamma2 = nn.Parameter(torch.ones((d_model)), requires_grad=True)
        # self.gamma3 = nn.Parameter(torch.ones((d_model)), requires_grad=True)
        # self.gamma4 = nn.Parameter(torch.ones((d_model)), requires_grad=True)

        # channel transfer
        self.gap = nn.AdaptiveAvgPool1d((1))

        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model // int(mlp_ratio), bias=False),
            nn.ReLU(),
            nn.Linear(d_model // int(mlp_ratio), d_model, bias=False),
            nn.Sigmoid()
        )


    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
        # z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)  # (b, d, h, w)
        # x = self.act(x)

        B, D, H, W = x.shape

        """ 掩码矩阵初始化 """
        locality = torch.zeros((B, 1, H, W)).to(x.device)
        locality[:, :, ::self.windows_size * 4, ::self.windows_size * 4] = 1    # 固定步幅标记

        one_masks, zero_masks = self.WMR(x, locality.requires_grad_(True))   # one_masks:(B, C, nW*window_size*window_size)

        """ Ouroboros衔尾操作 """
        first_elements = one_masks[:, 0, :].unsqueeze(dim=1)
        xs = torch.cat((one_masks, first_elements), dim=1)

        B, W_S, N = xs.shape
        xs = rearrange(xs, 'b w (c n) -> b c (w n)', c=D, w=W_S).unsqueeze(dim=1)

        """ 两种不同的遍历路径 """
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)  # (B, 2, C, nW*window_size*window_size)

        # 选择性扫描
        out_y = self.mambaScanner(xs)

        # 去除多余的token
        out_y = rearrange(out_y, "b k c (w n) -> b k w (c n)", w=W_S)[:,:,:W_S-1,:]
        out_y = rearrange(out_y, "b k w (c n) -> b k c (w n)", c=D)

        # """ 两种遍历路径叠加 (Mamba之后) """
        # token位置还原
        inv_y = torch.flip(out_y[:, 1:2], dims=[-1])
        # 四种状态叠加, 添加 投影权重
        y = inv_y[:, 0] + out_y[:, 0]           # (B, C, nW*window_size*window_size)

        # y = (self.gamma1.view(1, self.dim, 1) * out_y_[:, 0] + self.gamma2.view(1, self.dim, 1) * inv_y[:, 0]
        #      + self.gamma3.view(1, self.dim, 1) * wh_y + self.gamma4.view(1, self.dim, 1) * invwh_y

        # 还原形状，方便输出
        y = y.transpose(dim0=1, dim1=2).contiguous()

        # # # Local Aggregate
        # spatial
        Win_Q = self.unfold(x)
        B, L, N = Win_Q.shape
        Win_Q = rearrange(Win_Q, "b (c w) n -> (b n) w c", c=D)          # (B, C*window_size*window_size, nW_1) -> (B*nW_1, window_size*window_size, C)
        SSM_K, SSM_V = rearrange(self.kv(y).unsqueeze(dim=1).expand(-1, N, -1, -1), "b nw_1 l c -> (b nw_1) l c").transpose(dim0=1,dim1=2).chunk(
                                    2, dim=1)                                   # (B, nW_2*window_size*window_size, C) -> (B*nW_1, nW_2*window_size*window_size, C)

        # Local Aggregate attention
        A_M = Win_Q @ SSM_K                                                     # (B*nW_1, window_size*window_size, C) @ (B*nW_1, nW_2*window_size*window_size, C) -> (B*nW_1, W^2, nW_2*W^2)
        A_M = A_M / math.sqrt(W_S**2)
        A_M = torch.softmax(A_M, dim=-1)
        LA_Att = A_M @ SSM_V.transpose(dim0=1, dim1=2)                          # (B*nW_1, W^2, C)
        LA_Att = rearrange(LA_Att, '(b n) w c -> b (n w) c', b=B, n=N)

        # channel
        zero_masks = rearrange(zero_masks, 'b w (c n) -> b c (w n)', c=D)
        foreground_feature = self.gap(y.transpose(dim0=1, dim1=2))
        background_feature = self.gap(zero_masks)
        channel_feature = torch.cat([foreground_feature, background_feature], dim=1).transpose(dim0=1, dim1=2)
        channel_weight = self.fc(channel_feature)

        LA_Att = LA_Att * channel_weight

        # 正则输出
        y = self.out_norm(LA_Att).view(B, H, W, -1)
        y = y + z
        # y = torch.cat([y, z],dim=-1)
        # y = rearrange(y, "b h w c -> b c h w")
        # z是一个门控（SiLU激活分支）
        # y = y * z
        out = self.dropout(self.out_proj(y))
        # out = rearrange(out, "b c h w -> b h w c")
        return out


class SS2D(nn.Module, SS2Dv0, SS2Dv1):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            windows_size=2,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            mlp_ratio=4.0,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        r""" 初始化 SS2D
        Arg:
            d_model, d_state: 模型的维度和状态维度，影响特征表示的大小
            ssm_ratio: 状态空间模型的比例，可能影响模型的复杂度
            dt_rank: 时间步长的秩，控制时间序列的处理方式
            act_layer: 激活函数，默认为 SiLU（Sigmoid Linear Unit）
            d_conv: 卷积层的维度，值小于 2 时表示不使用卷积
            conv_bias: 是否使用卷积偏置项
            dropout: dropout 概率，用于防止过拟合
            bias: 是否在模型中使用偏置项
            dt_min, dt_max: 时间步长的最小和最大值
            dt_init: 时间步长的初始化方式，可以为 "random" 等
            dt_scale, dt_init_floor: 影响时间步长的缩放和下限
            initialize: 指定初始化方法
            forward_type: 决定前向传播的实现方式，支持多种类型
            channel_first: 指示是否使用通道优先的格式
            **kwargs: 允许传递额外参数，方便扩展
        """
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, windows_size=windows_size, mlp_ratio=mlp_ratio, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )

        # 调用不同的初始化函数
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        else:
            self.__initv1__(**kwargs)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            windows_size: int = 4,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            _SS2D: type = SS2D,
            **kwargs,
    ):
        r""" VMamba整体架构
        Arg:
            维度变换参数:
                hidden_dim: 输入和输出的特征维度
                drop_path: 用于随机丢弃路径的概率，防止过拟合
                norm_layer: 归一化层，默认为 LayerNorm
                channel_first: 数据格式，指示是否采用通道优先
            SSM相关参数:
                ssm_d_state: 状态空间模型的状态维度
                ssm_ratio: 决定是否使用 SSM 的比例
                ssm_dt_rank: 时间步长的秩
                ssm_act_layer: SSM 的激活函数，默认为 SiLU
                ssm_conv: 卷积层的大小
                ssm_conv_bias: 是否使用卷积偏置
                ssm_drop_rate: SSM 中的 dropout 概率
                ssm_init: SSM 的初始化方式
                forward_type: 决定前向传播的实现方式
            MLP相关参数:
                mlp_ratio: MLP 隐藏层与输入层的维度比率
                mlp_act_layer: MLP 的激活函数，默认为 GELU
                mlp_drop_rate: MLP 中的 dropout 概率
                gmlp: 是否使用 GMLP 结构
            其他参数:
                use_checkpoint: 是否使用梯度检查点以节省内存
                post_norm: 是否在添加残差连接后进行归一化
                _SS2D: 状态空间模型的类类型
        """

        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        ''' SSM模块, 初始化设置为 V0 版本 '''
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _SS2D(
                d_model=hidden_dim,
                windows_size=windows_size,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                mlp_ratio=mlp_ratio,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        ''' MLP '''
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


# 主函数
class LASSM(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            windows_size=2,
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version: str = "v2",  # "v1", "v2", "v3"
            patchembed_version: str = "v1",  # "v1", "v2"
            use_checkpoint=False,
            # =========================
            posembed=False,
            imgsize=224,
            _SS2D=SS2D,
            # =========================
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.input_resolution = [(10, 10), (20, 20), (40, 40), (80, 80)]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)  # 根据patchembed版本选择嵌入化函数
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                windows_size=windows_size,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,                  # 有三个版本的下采样
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features),  # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                          channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            windows_size=2,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # ===========================
            _SS2D=SS2D,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                windows_size=windows_size,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type='v0' if d % 2 == 0 and forward_type == ['v0', 'v1'] else ('v1' if forward_type == ['v0', 'v1'] else forward_type),
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        # x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b h w c')
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        ''' 堆叠模块 '''
        x_ = []
        for layer in self.layers:
            x_.append(rearrange(x, 'b h w c ->  b c h w'))
            x = layer(x)

        x = rearrange(x, 'b h w c ->  b c h w')
        return x, x_



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LASSM(
        depths=[2, 2, 2, 2], dims=96, drop_path_rate=0.3,
        patch_size=2, in_chans=96, num_classes=1, windows_size=5,
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v2", forward_type="v0",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v1", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=320,
    ) # 窗口大小必须为 20 的因数，因为最底层特征大小为 20*20

    net.cuda().train()

    summary(net, (2, 96, 80, 80))

    inputs = torch.randn(2, 96, 80, 80).cuda()
    flops, params = profile(net, (inputs,))
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
        input_data = torch.randn(batch_size, *input_size).to(device)  # 如果有GPU的话

        # 运行推理多次
        with torch.no_grad():
            for _ in range(num_iterations):
                # 启动计时器
                start_time = time.time()
                output = model(input_data)
                # 计算总时间
                total_time = time.time() - start_time
                t_all.append(total_time)

        print('average time:', np.mean(t_all) / 1)
        print('average fps:', 1 / np.mean(t_all))

        print('fastest time:', min(t_all) / 1)
        print('fastest fps:', 1 / min(t_all))

        print('slowest time:', max(t_all) / 1)
        print('slowest fps:', 1 / max(t_all))


    calculate_fps(net, input_size=(96, 80, 80), batch_size=2, num_iterations=10)



