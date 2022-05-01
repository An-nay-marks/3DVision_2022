from turtle import forward
from torch import batch_norm, nn, FloatTensor, cat
from torch.nn import functional as F
class OptimizerNN(nn.Module):
    def __init__(self, num_conv_layers):
        super().__init__()
        # dynamically increase output size
        self.canny = ... # TODO
        
        layers = []
        curr_channel_size = 4 # 3 for image +1 for Canny
        for i in range(num_conv_layers):
            layers.append(ConvolutionBatchNormRELU(curr_channel_size, curr_channel_size+3))
            curr_channel_size += 3
        layers.append(BinaryLinearOutput())
        self.model = nn.Sequential(layers)
    
    def forward(self, x):
        canny = self.canny(x)
        merged = cat((canny, x), 1) # append detected edges with input image
        return self.model(merged)

class ConvolutionBatchNormRELU(nn.Module):
    '''block consisting of conv2d, batch normalization and relu
    currently usign kernel of size 3, stride of 2, padding of 0, no bias
    padding of 0 is okay, because the patches are bigger than the face anyways
    because of batch normalization, no bias is needed'''
    def __init__(self, num_in_channels, num_out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size=3, stride=2, padding=0, bias=0),
            nn.BatchNorm2d(num_out_channels),
            nn.ReLU(inplace=True) # inplace vs batch norm
        )
    def forward(self, x):
        return self.block(x)

class BinaryLinearOutput(nn.Module):
    '''block flattening the input, forwarding through a linear layer to get a scalar output which is the predicted weight for the data input'''
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = None
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.flatten(x)
        if self.linear is None: # dynamically initialize the linear layer depending on the dimensions.
            self.linear = nn.Linear(x.size(dim=1), 1)
        x = self.linear(x)
        return self.sigmoid(x)

class SobelLayer(nn.Module):
    '''for detecting Edges'''
    def __init__(self):
        super().__init__()
        #self.kernel_vertical = FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        #self.kernel_horizontal = FloatTensor([[1, 0, -1], [2,0 , -2], [1, 0, -1]])
        # TODO: maybe just clone https://github.com/DCurro/CannyEdgePytorch
    def forward(self, x):
        # vertical = F.conv3d(x, self.kernel_vertical, stride = 1, padding=1) # keep dimensionality
        # horizontal = 
        ... # TODO
        