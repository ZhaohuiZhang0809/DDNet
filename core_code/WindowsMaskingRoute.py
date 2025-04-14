import torch
import torch.nn as nn
from einops import rearrange


## Mask generation process: 1. Initialization: First, a fixed 0 mask matrix is given, then the matrix is marked with the window size as the horizontal and vertical stride (fixed window marking).
##            2. Marking offset: After obtaining the fixed marking, a learnable deviation is generated for each one using deformable convolution, moving the coordinates of the marking points
##            3. Window mask: Subsequently, after obtaining the image matrix and the mask matrix, window partitioning is performed. The image matrix is valued according to the mask matrix. If a mask window tensor contains 1, the corresponding dimension of the image window is retained. This way, the image foreground and background features are obtained.


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DeformableConv2d, self).__init__()
        self.offset_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        offset = self.offset_layer(x)  # Generate offset

        return offset


class local_scan_zero_ones(nn.Module):
    def __init__(self, window_size):
        super(local_scan_zero_ones, self).__init__()
        self.window_size = window_size

    def forward(self, mask, x):
        """
        Windows Masking Route (WMR)

        Args:
            mask: 2D mask matrix (numpy array) with shape (B, H, W)
            x: Input feature map (torch tensor) with shape (B, C, H, W)
        Returns:
            components_ones: Features containing regions where the mask is 1
            components_zeros: Features containing regions where the mask is 0
        """
        # Get the size of the mask matrix
        B, C, H, W = x.shape
        k = self.window_size

        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size)
        mask = mask.expand(-1, C,-1,-1)
        mask = unfold(mask)     # (b, c*w**2, n)
        x = unfold(x)

        mask = rearrange(mask, "b (c w h) n -> b (w h) (c n)", c=C, w=k, h=k)
        x = rearrange(x, "b (c w h) n -> b (w h) (c n)", c=C, w=k, h=k)

        # Generate mask
        image_mask = mask.any(dim=1).unsqueeze(dim=1).expand(-1, k ** 2, -1)
        # .unsqueeze(dim=1).expand(-1, k ** 2, -1).cpu()  # Shape is (2, 200), boolean values indicate whether the window contains 1
        # Retain windows containing 1
        components_ones = x[image_mask]  # Retain windows containing 1
        components_ones = components_ones.view(B, k ** 2, -1)

        # Retain windows not containing 1
        components_zeros = x[~image_mask]  # Retain windows not containing 1
        components_zeros = components_zeros.view(B, k ** 2, -1)

        return components_ones, components_zeros


class WindowsMaskingRoute(nn.Module):
    def __init__(self, in_channels=96, kernel_size=3, padding=1, window_size=4):
        super(WindowsMaskingRoute, self).__init__()
        self.deform_conv = DeformableConv2d(in_channels, 2, kernel_size, padding)
        self.local_scan_zero_ones = local_scan_zero_ones(window_size)

    def apply_offset(self, locality, dx, dy):
        """
        Apply offsets dx and dy to process the locality matrix
        """
        batchs, chs, rows, cols = locality.shape
        locality = rearrange(locality,"b c h w -> b h w c")
        dx = dx.squeeze(1)
        dy = dy.squeeze(1)

        # Calculate new index positions
        new_i = torch.clamp(torch.arange(rows, device=dx.device).unsqueeze(1) + dx, 0, rows - 1).long()
        new_j = torch.clamp(torch.arange(cols, device=dx.device).unsqueeze(1) + dy, 0, cols - 1).long()

        # Use advanced indexing to update offset_locality
        offset_locality = torch.zeros_like(locality).to(dx.device).requires_grad_(True)
        # offset_locality[torch.arange(batchs).unsqueeze(1).unsqueeze(1).unsqueeze(1), new_i, new_j] = locality

        offset_locality = torch.where(
            (offset_locality[torch.arange(batchs).unsqueeze(1).unsqueeze(1), new_i, new_j] == 1),
            offset_locality[torch.arange(batchs).unsqueeze(1).unsqueeze(1), new_i, new_j],  # Keep original value
            locality  # Assign value
        )


        # Restore to original shape
        offset_locality = rearrange(offset_locality, "b h w c -> b c h w")

        return offset_locality


    def forward(self, input_image, locality):
        """
        Forward propagation logic:
        1. Generate offsets using deformable convolution.
        2. Apply offsets to the locality matrix.
        3. Process the offset locality matrix using window masking route.
        """
        # Generate offsets
        offset = self.deform_conv(input_image)
        dx = offset[:, 0:1, :, :]
        dy = offset[:, 1:2, :, :]

        # Apply offsets to the locality matrix
        offset_locality = self.apply_offset(locality, dx, dy)

        # Apply the offset matrix to the original image
        ones, zeros = self.local_scan_zero_ones(offset_locality, input_image)

        return ones, zeros

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Input image

    input_image = torch.randn(2, 96, 80, 80).to(device)  # Example input image
    input_image = input_image.float() / 255.0
    # locality = torch.randint(0, 2, (2, 1, 304, 304), dtype=torch.int64)

    locality = torch.zeros((2, 1, 80, 80)).to(device)
    # Fill the matrix with 1s
    locality[:, :, ::5, ::5] = 1

    net = WindowsMaskingRoute(in_channels=96).to(device)
    ones, zeros = net(input_image, locality)
    print(ones.shape, zeros.shape)


    import time
    import numpy as np

    def calculate_fps(model, input_size, batch_size=1, num_iterations=100):
        t_all = []
        # Set the model to evaluation mode
        model.eval()
        # Simulate input data

        # Run inference multiple times
        with torch.no_grad():
            for _ in range(num_iterations):
                # Start the timer
                start_time = time.time()
                ones, zeros = model(input_image, locality)
                print(ones.shape, zeros.shape)
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
