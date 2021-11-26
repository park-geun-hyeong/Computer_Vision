import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64,kernel_size = 9, stride = 1, padding = 9//2)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64,kernel_size = 5, stride = 1, padding = 5//2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels = 1, kernel_size = 5, stride=1, padding = 5//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x

if __name__ == "__main__":
    model = SRCNN()
