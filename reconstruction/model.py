import torch
from torch import nn
from torch import from_numpy
import numpy as np
from reconstruction.CannyEdgePytorch.net_canny import Net

class OptimizerNN(nn.Sequential):
    def __init__(self, checkpoint_path=None):

        layers = []
        num_conv_layers = 10
        curr_channel_size = 3  # 3 for image +1 for Canny
        kernel_size = 5

        # canny
        CannyNet = Net(threshold=3.0, use_cuda=False)
        

        for i in range(num_conv_layers):
            next_channel_size = curr_channel_size + 3
            layers.append(ConvBatchNormRELU(curr_channel_size, next_channel_size, kernel_size))
            if (i == 5):
                layers.append(CannyNet)
            curr_channel_size = next_channel_size


        image_dim = 224 - num_conv_layers * (kernel_size - 1)
        super().__init__(*layers, BinaryLinearOutput(curr_channel_size * image_dim * image_dim))

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])


class ConvBatchNormRELU(nn.Sequential):
    """block consisting of conv2d, batch normalization and relu
    currently usign kernel of size 3, stride of 2, padding of 0, no bias
    padding of 0 is okay, because the patches are bigger than the face anyways
    because of batch normalization, no bias is needed"""

    def __init__(self, num_in_channels, num_out_channels, kernel_size):
        super().__init__(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_out_channels),
            nn.ReLU(inplace=True)  # inplace vs batch norm
        )

# class CannyDetector(nn.Module):
#     def __init__(self):
#         Net(self)

class BinaryLinearOutput(nn.Sequential):
    """block flattening the input, forwarding through a linear layer to get a scalar output which is the predicted
    weight for the data input """

    def __init__(self, dim):
        super().__init__(
            nn.Flatten(),
            nn.Linear(dim, 1),
            nn.ReLU()
        )


#class SobelLayer(nn.Module):
    """for detecting Edges"""

    # def __init__(self):
    #     super().__init__(self, num_in_channels, num_out_channels, kernel_size)
    #     # self.kernel_vertical = FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    #     # self.kernel_horizontal = FloatTensor([[1, 0, -1], [2,0 , -2], [1, 0, -1]])
    #     # TODO: maybe just clone https://github.com/DCurro/CannyEdgePytorch
    #     sobel_filter = np.array([[1, 0, -1],
    #                              [2, 0, -2],
    #                              [1, 0, -1]])

    #     self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
    #     self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)


    # def forward(self, x):
    #     # vertical = F.conv3d(x, self.kernel_vertical, stride = 1, padding=1) # keep dimensionality
    #     # horizontal = 
    #     # TODO

    #     grad_x = self.sobel_filter_horizontal(x)
    #     grad_y = self.sobel_filter_vertical(x)

    #     return np.abs(grad_x) + np.abs(grad_y)


