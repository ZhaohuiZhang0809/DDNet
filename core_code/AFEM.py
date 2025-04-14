import torch
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

        self.fc = nn.Sequential(
            nn.Linear(dim, dim // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(dim // ratio, dim, bias=False),  # 从 c/r -> c
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.Laplace(x)
        c_w = self.gap(x_h).view(b, c)
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




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = AFEM(dim = 384).to(device)

    # 打印网络结构和参数
    summary(net, (2, 384, 20, 20))

    inputs = torch.randn(2, 384, 20, 20).cuda()
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


    calculate_fps(net, input_size=(384, 20, 20), batch_size=2, num_iterations=3)
