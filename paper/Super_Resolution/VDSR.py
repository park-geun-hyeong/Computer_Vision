import torch
import torch.nn as nn
from math import sqrt

class Conv_Relu_block(nn.Module):
    def __init__(self):
        super(Conv_Relu_block, self).__init__()

        self.conv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()

        self.residual_layer = self.make_layer(Conv_Relu_block, 18)
        self.input_layer = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.output_layer = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n)) # Conv2d weight normalization


    def make_layer(self, block: classmethod, num_block: int):
        layers = []
        for _ in range(num_block):
            layers.append(block())

        return nn.Sequential(*layers)


    def forward(self, x):
        residual = x
        out = self.relu(self.input_layer(x))
        out = self.residual_layer(out)
        out = self.output_layer(out)

        final  = torch.add(residual, out)
        return final

if __name__ == "__main__":
    model = VDSR()
    print("success")

