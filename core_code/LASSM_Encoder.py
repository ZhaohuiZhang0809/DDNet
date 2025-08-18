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


# SS2Dv1
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
            d_model: The output dimension of the model (default is 96).
            d_state: The state dimension (default is 16).
            ssm_ratio: The ratio of the state dimension to the model dimension (default is 2.0).
            dt_rank: The dimension of the dynamic time parameter, which is calculated based on d_model if set to "auto"
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
        self.windows_size = windows_size

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # self.selective_scan = selective_scan_fn  # Selective scanning (acceleration)

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
        # # Fill in the 1s in the matrix
        # locality[:, :, ::2, ::3] = 1
        #
        # one_masks, zero_masks = self.WMR(x, locality)
        # x = torch.cat((one_masks, zero_masks), dim=0)
        # x = rearrange(x, '(B C H W) -> B C H W', B=B, C=D, W=W)

        """ Local traversal path """
        # partition
        xs = window_partition(x, self.windows_size)                               # (b, local, local, c)
        trans_xs = torch.transpose(x, dim0=2, dim1=3)
        trans_xs = window_partition(trans_xs, self.windows_size)                  # (b, local, local, c)

        # xs = rearrange(xs, 'b c n m h w -> b c (n m h w)')
        # trans_xs = rearrange(trans_xs, 'b c n m h w -> b c (n m h w)')

        # # Stack the two views of the input tensor x (original and transposed), [b, 2, d, l]

        xs = rearrange(xs, '(b n) c h w -> b c (n h w)', b=B)
        trans_xs = rearrange(trans_xs, '(b n) c h w -> b c (n h w)', b=B)
        x_hwwh = torch.stack([xs, trans_xs], dim=1).view(B, 2, -1, L)
        # Concatenate x_hwwh and its reverse
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        # Selective scanning
        out_y = self.mambaScanner(xs)

        # """ Overlay of four traversal paths (after Mamba) """
        # Restore token position
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # Overlay of four states
        y = self.gamma1.view(1, self.dim, 1) * out_y[:, 0] + self.gamma2.view(1, self.dim, 1) * inv_y[:, 0] + self.gamma3.view(1, self.dim, 1) * wh_y + self.gamma4.view(1, self.dim, 1) * invwh_y

        # Restore shape
        y = rearrange(y, 'b c (n m h w) -> b n m h w c', n=H // self.windows_size, h=self.windows_size, w=self.windows_size)
        y = y.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, L, -1)

        # global channel
        gc = self.gap(x).view(B, D)
        gc = self.fc(gc).view(B, 1, D)
        y = y * gc

        # Regular output
        y = self.out_norm(y).view(B, H, W, -1)
        # z是一个门控（SiLU激活分支）
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out




# support: v0, v0seq, LASSM
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
        r"""
            Arg:
            d_model: The output dimension of the model (default is 96).
            d_state: The state dimension (default is 16).
            ssm_ratio: The ratio of the state dimension to the model dimension (default is 2.0).
            dt_rank: The dimension of the dynamic time parameter, which is calculated based on d_model if set to "auto"
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
        k_group = 2     # Scanning direction
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

        # self.selective_scan = selective_scan_fn  # Selective scanning (acceleration)

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

        """ Initialization of the mask matrix """
        locality = torch.zeros((B, 1, H, W)).to(x.device)
        locality[:, :, ::self.windows_size * 4, ::self.windows_size * 4] = 1    # Fixed stride marking

        one_masks, zero_masks = self.WMR(x, locality.requires_grad_(True))   # one_masks:(B, C, nW*window_size*window_size)
        
        one_masks = torch.autograd.Variable(one_masks, requires_grad=True)
        zero_masks = torch.autograd.Variable(zero_masks, requires_grad=True)

        """ Ouroboros operation """
        first_elements = one_masks[:, 0, :].unsqueeze(dim=1)
        xs = torch.cat((one_masks, first_elements), dim=1)

        B, W_S, N = xs.shape
        xs = rearrange(xs, 'b w (c n) -> b c (w n)', c=D, w=W_S).unsqueeze(dim=1)

        """ Two different traversal paths """
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)  # (B, 2, C, nW*window_size*window_size)

        # Selective scanning
        out_y = self.mambaScanner(xs)

        # Remove redundant tokens
        out_y = rearrange(out_y, "b k c (w n) -> b k w (c n)", w=W_S)[:,:,:W_S-1,:]
        out_y = rearrange(out_y, "b k w (c n) -> b k c (w n)", c=D)

        # """ Overlay of two traversal paths (after Mamba) """
        # Restore token position
        inv_y = torch.flip(out_y[:, 1:2], dims=[-1])
        # Overlay of two states, add projection weights
        y = inv_y[:, 0] + out_y[:, 0]           # (B, C, nW*window_size*window_size)

        # y = (self.gamma1.view(1, self.dim, 1) * out_y_[:, 0] + self.gamma2.view(1, self.dim, 1) * inv_y[:, 0]
        #      + self.gamma3.view(1, self.dim, 1) * wh_y + self.gamma4.view(1, self.dim, 1) * invwh_y

        # Restore shape for output
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
    
        # Regular output
        y = self.out_norm(LA_Att).view(B, H, W, -1)
        y = y + z
        # y = torch.cat([y, z],dim=-1)
        # y = rearrange(y, "b h w c -> b c h w")
        # z is a gate (SiLU activated branch)
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
        r""" Initialize SS2D
        Arg:
            d_model, d_state: The model dimension and state dimension, affecting the size of feature representation
            ssm_ratio: The ratio of the state space model, which may affect the complexity of the model
            dt_rank: The rank of the dynamic time step, controlling the processing of time series
            act_layer: The activation function, defaulting to SiLU (Sigmoid Linear Unit)
            d_conv: The dimension of the convolution layer, with values less than 2 indicating no convolution
            conv_bias: Whether to use convolution bias
            dropout: Dropout probability to prevent overfitting
            bias: Whether to use bias in the model
            dt_min, dt_max: The minimum and maximum values of the dynamic time step
            dt_init: The initialization method of the dynamic time step, which can be "random", etc.
            dt_scale, dt_init_floor: Affecting the scaling and lower limit of the dynamic time step
            initialize: Specifying the initialization method
            forward_type: Determining the implementation of the forward propagation, supporting multiple types
            channel_first: Indicating whether to use channel-first format
            **kwargs: Allowing additional parameters to be passed for easy expansion
        """
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, windows_size=windows_size, mlp_ratio=mlp_ratio, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )

        # Call different initialization functions
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
        r""" VMamba overall architecture
        Arg:
            Dimension transformation parameters:
                hidden_dim: The feature dimension of input and output
                drop_path: The probability used for random path dropping to prevent overfitting
                norm_layer: The normalization layer, defaulting to LayerNorm
                channel_first: Data format, indicating whether to use channel-first
            SSM-related parameters:
                ssm_d_state: The state dimension of the state space model
                ssm_ratio: The ratio determining whether to use SSM
                ssm_dt_rank: The rank of the dynamic time step
                ssm_act_layer: The activation function of SSM, defaulting to SiLU
                ssm_conv: The size of the convolution layer
                ssm_conv_bias: Whether to use convolution bias
                ssm_drop_rate: The dropout probability in SSM
                ssm_init: The initialization method of SSM
                forward_type: Determining the implementation of the forward propagation
            MLP-related parameters:
                mlp_ratio: The ratio of the hidden layer to the input layer in MLP
                mlp_act_layer: The activation function of MLP, defaulting to GELU
                mlp_drop_rate: The dropout probability in MLP
                gmlp: Whether to use GMLP structure
            Other parameters:
                use_checkpoint: Whether to use gradient checkpointing to save memory
                post_norm: Whether to normalize after adding the residual connection
                _SS2D: The class type of the state space model
        """

        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        ''' SSM module, initialized as V0 version '''
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


# main function
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
                downsample=downsample,                  
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
        ''' stack '''
        x_list = []
        for layer in self.layers:
            x_.append(rearrange(x, 'b h w c ->  b c h w'))
            x = layer(x)

        x = rearrange(x, 'b h w c ->  b c h w')
        return x, x_list



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
    ) # 320,windows_size=5 or 10 or ...

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
        # Set the model to evaluation mode
        model.eval()
        # Create dummy input data
        input_data = torch.randn(batch_size, *input_size).to(device)  # If GPU is available
        
        # Run inference multiple times
        with torch.no_grad():
            for _ in range(num_iterations):
                # Start the timer
                start_time = time.time()
                output = model(input_data)
                # Calculate the total time
                total_time = time.time() - start_time
                t_all.append(total_time)
        
        print('average time:', np.mean(t_all) / 1)
        print('average fps:', 1 / np.mean(t_all))
        
        print('fastest time:', min(t_all) / 1)
        print('fastest fps:', 1 / min(t_all))
        
        print('slowest time:', max(t_all) / 1)
        print('slowest fps:', 1 / max(t_all))
        
        
    calculate_fps(net, input_size=(96, 80, 80), batch_size=2, num_iterations=10)




