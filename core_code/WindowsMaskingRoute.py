import torch
import torch.nn as nn
from einops import rearrange


## 掩码生成过程：1. 初始化：首先给定一个固定的0掩码矩阵,随后在0掩码矩阵上以窗口大小作为水平方向和垂直方向步幅进行标记（固定的窗口标记）.
##            2. 标记偏移：在得到固定标记后，使用可形变的卷积为每一个生成可学习的偏差，移动标记点坐标
##            3. 窗口掩码：随后,在得到图像矩阵和掩码矩阵后,进行窗口划分。根据掩码矩阵对图像矩阵进行取值。若掩码窗口张量中, 某一窗口包含1则保留对应维度的图像窗口。以此得到图像前景和背景特征。

# class DeformableConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding):
#         super(DeformableConv2d, self).__init__()
#         self.offset_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
#         # self.deform_conv = ops.DeformConv2d(in_channels, out_channels, kernel_size, padding=padding)
#
#     def forward(self, x):
#         offset = self.offset_layer(x)  # 生成偏移量
#         # output = self.deform_conv(x, offset)  # 应用可形变卷积
#         return offset
#
#
# class local_scan_zero_ones(nn.Module):
#     def __init__(self, window_size):
#         super(local_scan_zero_ones, self).__init__()
#         self.window_size = window_size
#
#     def forward(self, mask, x):
#         """
#         Windows Masking Route (WMR)
#
#         Args:
#             mask: 二维掩码矩阵 (numpy array) with shape (B, H, W)
#             x: 输入特征图 (torch tensor) with shape (B, C, H, W)
#         Returns:
#             components_ones: 包含掩码为 1 的区域的特征
#             components_zeros: 包含掩码为 0 的区域的特征
#         """
#         # 获取掩码矩阵的大小
#         B, C, H, W = x.shape
#         k = self.window_size
#
#         unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size)
#         mask = mask.expand(-1, C,-1,-1)
#         mask = unfold(mask)     # (b, c*w**2, n)
#         x = unfold(x)
#
#         mask = rearrange(mask, "b (c w h) n -> b (w h) (c n)", c=C, w=k, h=k)
#         x = rearrange(x, "b (c w h) n -> b (w h) (c n)", c=C, w=k, h=k)
#
#         # 生成掩码
#         image_mask = mask.any(dim=1).unsqueeze(dim=1).expand(-1, k ** 2, -1).cpu()
#         # .unsqueeze(dim=1).expand(-1, k ** 2, -1).cpu()  # 形状为 (2, 200)，布尔值表示窗口是否包含1
#         # 保留包含1的窗口
#         components_ones = x[image_mask]  # 保留包含1的窗口
#         components_ones = components_ones.view(B, k ** 2, -1)
#
#         # 保留不包含1的窗口
#         components_zeros = x[~image_mask]  # 保留不包含1的窗口
#         components_zeros = components_zeros.view(B, k ** 2, -1)
#
#         return components_ones, components_zeros
#
#
# class WindowsMaskingRoute(nn.Module):
#     def __init__(self, in_channels=96, kernel_size=3, padding=1, window_size=4):
#         super(WindowsMaskingRoute, self).__init__()
#         self.deform_conv = DeformableConv2d(in_channels, 2, kernel_size, padding)
#         self.local_scan_zero_ones = local_scan_zero_ones(window_size)
#
#     @staticmethod
#     def apply_offset(locality, dx, dy):
#         """
#         应用偏移量 dx 和 dy 来处理矩阵 locality
#         """
#         offset_locality = np.zeros_like(locality)
#         batchs, chs, rows, cols = locality.shape
#
#         for b in range(batchs):
#             for c in range(chs):
#                 for i in range(rows):
#                     for j in range(cols):
#                         if locality[b, c, i, j] == 1.0:  # 如果当前元素值为 1
#                             delta_x = int(dx[b, c, i, j])  # 偏移量 x
#                             delta_y = int(dy[b, c, i, j])  # 偏移量 y
#
#                             new_i = i + delta_x
#                             new_j = j + delta_y
#
#                             # 检查偏移后的位置是否在矩阵范围内
#                             if 0 <= new_i < rows and 0 <= new_j < cols:
#                                 offset_locality[b, c, new_i, new_j] = 1.0
#                                 if new_i !=0 and new_j != 0:
#                                    locality[b, c, i, j] = 0.0
#
#         offset_locality = torch.tensor(offset_locality, dtype=torch.float, device='cuda')
#
#         return offset_locality
#
#
#     def forward(self, input_image, locality):
#         """
#         前向传播逻辑：
#         1. 使用可变形卷积生成偏移量。
#         2. 应用偏移量到局部性矩阵。
#         3. 使用窗口掩码路由处理偏移后的局部性矩阵。
#
#         Args:
#             input_image: 输入图像 (torch tensor) with shape (B, C, H, W)
#             locality: 局部性矩阵 (torch tensor) with shape (B, 1, H, W)
#         Returns:
#             ones: 包含掩码为 1 的区域的特征
#             zeros: 包含掩码为 0 的区域的特征
#         """
#         # 生成偏移量
#         offset = self.deform_conv(input_image)
#
#         dx = offset[:, 0:1, :, :]  # 偏移量 dx
#         dy = offset[:, 1:2, :, :]  # 偏移量 dy
#
#         # 将偏移量和局部性矩阵转换为 NumPy 数组
#         dx = dx.detach().cpu().numpy()
#         dy = dy.detach().cpu().numpy()
#         locality = locality.detach().cpu().numpy()
#
#         offset_locality = self.apply_offset(locality, dx, dy)
#
#         # 将偏移后的矩阵应用于原图像
#         ones, zeros = self.local_scan_zero_ones(offset_locality, input_image)
#
#         return ones, zeros


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DeformableConv2d, self).__init__()
        self.offset_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        offset = self.offset_layer(x)  # 生成偏移量

        return offset


class local_scan_zero_ones(nn.Module):
    def __init__(self, window_size):
        super(local_scan_zero_ones, self).__init__()
        self.window_size = window_size

    def forward(self, mask, x):
        """
        Windows Masking Route (WMR)

        Args:
            mask: 二维掩码矩阵 (numpy array) with shape (B, H, W)
            x: 输入特征图 (torch tensor) with shape (B, C, H, W)
        Returns:
            components_ones: 包含掩码为 1 的区域的特征
            components_zeros: 包含掩码为 0 的区域的特征
        """
        # 获取掩码矩阵的大小
        B, C, H, W = x.shape
        k = self.window_size

        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size)
        mask = mask.expand(-1, C,-1,-1)
        mask = unfold(mask)     # (b, c*w**2, n)
        x = unfold(x)

        mask = rearrange(mask, "b (c w h) n -> b (w h) (c n)", c=C, w=k, h=k)
        x = rearrange(x, "b (c w h) n -> b (w h) (c n)", c=C, w=k, h=k)

        # 生成掩码
        image_mask = mask.any(dim=1).unsqueeze(dim=1).expand(-1, k ** 2, -1)
        # .unsqueeze(dim=1).expand(-1, k ** 2, -1).cpu()  # 形状为 (2, 200)，布尔值表示窗口是否包含1
        # 保留包含1的窗口
        components_ones = x[image_mask]  # 保留包含1的窗口
        components_ones = components_ones.view(B, k ** 2, -1)

        # 保留不包含1的窗口
        components_zeros = x[~image_mask]  # 保留不包含1的窗口
        components_zeros = components_zeros.view(B, k ** 2, -1)

        return components_ones, components_zeros


class WindowsMaskingRoute(nn.Module):
    def __init__(self, in_channels=96, kernel_size=3, padding=1, window_size=4):
        super(WindowsMaskingRoute, self).__init__()
        self.deform_conv = DeformableConv2d(in_channels, 2, kernel_size, padding)
        self.local_scan_zero_ones = local_scan_zero_ones(window_size)

    def apply_offset(self, locality, dx, dy):
        """
        应用偏移量 dx 和 dy 来处理矩阵 locality
        """
        batchs, chs, rows, cols = locality.shape
        locality = rearrange(locality,"b c h w -> b h w c")
        dx = dx.squeeze(1)
        dy = dy.squeeze(1)

        # 计算新的索引位置
        new_i = torch.clamp(torch.arange(rows, device=dx.device).unsqueeze(1) + dx, 0, rows - 1).long()
        new_j = torch.clamp(torch.arange(cols, device=dx.device).unsqueeze(1) + dy, 0, cols - 1).long()

        # 使用高级索引更新 offset_locality
        offset_locality = torch.zeros_like(locality).to(dx.device).requires_grad_(True)
        # offset_locality[torch.arange(batchs).unsqueeze(1).unsqueeze(1).unsqueeze(1), new_i, new_j] = locality

        offset_locality = torch.where(
            (offset_locality[torch.arange(batchs).unsqueeze(1).unsqueeze(1), new_i, new_j] == 1),
            offset_locality[torch.arange(batchs).unsqueeze(1).unsqueeze(1), new_i, new_j],  # 保持原值
            locality  # 进行赋值
        )


        # 恢复为原始形状
        offset_locality = rearrange(offset_locality, "b h w c -> b c h w")

        return offset_locality


    def forward(self, input_image, locality):
        """
        前向传播逻辑：
        1. 使用可变形卷积生成偏移量。
        2. 应用偏移量到局部性矩阵。
        3. 使用窗口掩码路由处理偏移后的局部性矩阵。
        """
        # 生成偏移量
        offset = self.deform_conv(input_image)
        dx = offset[:, 0:1, :, :]
        dy = offset[:, 1:2, :, :]

        # 应用偏移量到局部性矩阵
        offset_locality = self.apply_offset(locality, dx, dy)

        # 将偏移后的矩阵应用于原图像
        ones, zeros = self.local_scan_zero_ones(offset_locality, input_image)

        return ones, zeros




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 输入图像

    input_image = torch.randn(2, 96, 80, 80).to(device)  # 示例输入图像
    input_image = input_image.float() / 255.0
    # locality = torch.randint(0, 2, (2, 1, 304, 304), dtype=torch.int64)

    locality = torch.zeros((2, 1, 80, 80)).to(device)
    # 填充矩阵中的1
    locality[:, :, ::5, ::5] = 1

    net = WindowsMaskingRoute(in_channels=96).to(device)
    ones, zeros = net(input_image, locality)
    print(ones.shape, zeros.shape)


    import time
    import numpy as np

    def calculate_fps(model, input_size, batch_size=1, num_iterations=100):
        t_all = []
        # 模型设置为评估模式
        model.eval()
        # 模拟输入数据

        # 运行推理多次
        with torch.no_grad():
            for _ in range(num_iterations):
                # 启动计时器
                start_time = time.time()
                ones, zeros = model(input_image, locality)
                print(ones.shape, zeros.shape)
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