import torch
import reconstruction.canny as canny

from torch import nn
from utils_3DV import DEVICE


class OptimizerCanny(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.canny = canny.Net(threshold=3.0)
        self.tail = OptimizerNN(input_channels=6)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model'])
    
    def forward(self, image):
        edge_info = self.canny(image)
        out = torch.cat((image, *edge_info), dim=1)
        return self.tail(out)
        

class OptimizerNN(nn.Sequential):
    def __init__(self, checkpoint_path=None, input_channels=3):
        layers = nn.ModuleList()
        kernel_size = 3
        num_blocks = 3

        prev_channel_size = input_channels
        curr_channel_size = 64

        for _ in range(num_blocks):
            layers.append(ConvBatchNormRELU(prev_channel_size, curr_channel_size, kernel_size))
            layers.append(ConvBatchNormRELU(curr_channel_size, curr_channel_size, kernel_size))
            layers.append(ConvBatchNormRELU(curr_channel_size, curr_channel_size, kernel_size))
            prev_channel_size = curr_channel_size
            curr_channel_size *= 2

        out_dim = 224 - num_blocks * 3 * (kernel_size - 1)
        super().__init__(*layers, BinaryLinearOutput(prev_channel_size * out_dim * out_dim))

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model'])


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


class BinaryLinearOutput(nn.Sequential):
    """block flattening the input, forwarding through a linear layer to get a scalar output which is the predicted
    weight for the data input """

    def __init__(self, dim):
        super().__init__(
            nn.Flatten(),
            nn.Linear(dim, 1)
        )
