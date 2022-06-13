import torch
from torch import nn


class OptimizerNN(nn.Sequential):
    def __init__(self, checkpoint_path=None):
        layers = []
        kernel_size = 3

        prev_channel_size = 3
        curr_channel_size = 64

        for _ in range(4):
            layers.append(ConvBatchNormRELU(prev_channel_size, curr_channel_size, kernel_size))
            layers.append(ConvBatchNormRELU(curr_channel_size, curr_channel_size, kernel_size))
            layers.append(ConvBatchNormRELU(curr_channel_size, curr_channel_size, kernel_size))
            prev_channel_size = curr_channel_size
            curr_channel_size *= 2

        image_dim = 224 - 12 * (kernel_size - 1)
        super().__init__(*layers, BinaryLinearOutput(prev_channel_size * image_dim * image_dim))

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


class BinaryLinearOutput(nn.Sequential):
    """block flattening the input, forwarding through a linear layer to get a scalar output which is the predicted
    weight for the data input """

    def __init__(self, dim):
        super().__init__(
            nn.Flatten(),
            nn.Linear(dim, 1),
            nn.ReLU()
        )
