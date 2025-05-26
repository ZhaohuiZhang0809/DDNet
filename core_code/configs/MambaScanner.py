import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch import nn
import math


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        r""" 初始化 delta 投影参数
        Args:
            dt_rank：输入特征的维度
            d_inner：输出特征的维度
            dt_scale：初始化标准差的缩放因子，默认值为 1.0
            dt_init：初始化方法，可以是 "constant" 或 "random"
            dt_min 和 dt_max：初始化偏置时的最小和最大值
            dt_init_floor：偏置的下限，防止过小的值
        """

        # 创建线性层
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # 根据选择的初始化方法，使用常量或随机均匀分布来设置线性层权重
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化偏置
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        # 计算偏置的逆 Softplus; Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        # 更新偏置值
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        r""" 初始化一个参数张量 A_log (矩阵A)
        Arg:
            d_state：状态维度，表示生成的张量的长度
            d_inner：内层维度，表示输出张量的行数
            copies：用于重复张量的数量，默认为 -1 表示不重复
            device：指定张量存储的设备（如 CPU 或 GPU）
            merge：布尔值，决定是否将多个副本合并为一个张量
        """

        # 创建张量 A, (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        # 离散化, 计算 A 的对数
        A_log = torch.log(A)  # Keep A_log in fp32
        # 是否对 A_log 进行扩展; (d_inner, d_state) -> (copies, d_inner, d_state) -> (copies * d_inner, d_state)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        # 将 A_log 转换为可训练参数
        A_log = nn.Parameter(A_log)
        # 将 A_log 设置为无权重衰减
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        r""" 初始化一个参数张量 D (矩阵D)
        Arg:
            d_inner：内层维度，表示生成的张量的大小
            copies：用于重复张量的数量，默认为 -1，表示不重复
            device：指定张量存储的设备（如 CPU 或 GPU）
            merge：布尔值，决定是否将多个副本合并为一个张量
        """
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        r""" 初始化 delta投影, A, D矩阵
        Arg:
            d_state：状态维度，可能用于描述模型的状态空间
            dt_rank：降维或投影的秩
            d_inner：内层维度，通常用于网络的内部表示
            dt_scale、dt_init、dt_min、dt_max、dt_init_floor：这些参数用于控制投影的初始化
            k_group：组的数量，默认为 4，用于重复初始化
        """

        # dt proj ============================ delta 投影
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        # delta 投影权重和偏置
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class MambaScanner(nn.Module):
    def __init__(self, force_fp32, seq, init_dt_A_D, x_proj_weight):
        super().__init__()
        r''' 公用的Mamba Scanner
        Arg:
            force_fp32：是否使用fp32进行计算
            seq：是否分开扫描
            inner：hidden state dim
            rank：投影维度
        '''
        self.force_fp32 = force_fp32
        self.seq = seq

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = init_dt_A_D
        self.x_proj_weight = x_proj_weight
        self.selective_scan = selective_scan_fn  # 选择性扫描（加速）

    def forward(self, xs):
        B, K, D, L = xs.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape

        force_fp32 = self.force_fp32
        seq = self.seq

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        """ x --投影-> delta, B, C 矩阵 """
        # 由投影后的x分别得到 delta, B, C 矩阵, '(B, L, D) -> (B, L, N)'
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        # 将 dts（delta） 通过权重矩阵 self.dt_projs_weight 进行投影, '(B, L, N) -> (B, L, D)'
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)               # 保证 delta, B, C 在内存中的连续（加速计算）
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)

        As = -self.A_logs.float().exp()  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                """ 选择性扫描 """
                yi = self.selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)  # 在 selective_scan 函数中进行离散化操作
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y



class CrossScanner(nn.Module):
    def __init__(self, force_fp32, seq, init_dt_A_D, x_proj_weight):
        super().__init__()
        r''' 公用的Mamba Scanner
        Arg:
            force_fp32：是否使用fp32进行计算
            seq：是否分开扫描
            inner：hidden state dim
            rank：投影维度
        '''
        self.force_fp32 = force_fp32
        self.seq = seq

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = init_dt_A_D
        self.x_proj_weight = x_proj_weight
        self.selective_scan = selective_scan_fn  # 选择性扫描（加速）

    def forward(self, q_x, kv_x):
        B, K, D, L = kv_x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape

        force_fp32 = self.force_fp32
        seq = self.seq

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", kv_x, self.x_proj_weight)
        x_c = torch.einsum("b k d l, k c d -> b k c l", q_x, self.x_proj_weight)

        """ x --投影-> delta, B, C 矩阵 """
        # 由投影后的x分别得到 delta, B, C 矩阵, '(B, L, D) -> (B, L, N)'
        dts, Bs, _ = torch.split(x_dbl, [R, N, N], dim=2)
        _, _, Cs = torch.split(x_c, [R, N, N], dim=2)

        # 将 dts（delta） 通过权重矩阵 self.dt_projs_weight 进行投影, '(B, L, N) -> (B, L, D)'
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        kv_x = kv_x.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)               # 保证 delta, B, C 在内存中的连续（加速计算）
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)

        As = -self.A_logs.float().exp()  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(kv_x.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:
            kv_x, dts, Bs, Cs = to_fp32(kv_x, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                """ 选择性扫描 """
                yi = self.selective_scan(
                    kv_x.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)  # 在 selective_scan 函数中进行离散化操作
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = self.selective_scan(
                kv_x, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y