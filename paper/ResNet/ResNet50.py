import torch
import torch.nn as nn

def conv_block_1(in_dim, out_dim, act_fn, stride=1):
    model = nn.Seqnential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0),
        act_fn

    )
    return model

def conv_block_3(in_dim, out_dim, act_fn, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
        act_fn
    )

    return model

class BottleNeck(nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, act_fn, down:bool = False):
        super(BottleNeck, self).__init__()

        self.act_fn = act_fn
        self.down = down

        if self.down:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim,act_fn, stride=2),
                conv_block_3(mid_dim, mid_dim),
                conv_block_1(mid_dim, out_dim)
            )
            self.downsample = nn.Conv2d(in_dim,out_dim,1,2) ## x의 차원 맞춰주기
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, act_fn, stride=1),
                conv_block_3(mid_dim, mid_dim, act_fn),
                conv_block_1(mid_dim, out_dim, act_fn)
            )

        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0) ## x와 H(x)의 차원 맞춰주기


    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out += downsample

        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)

            out += x

        return out

class ResNet50(nn.Module):

    def __init__(self, base_dim, num_classes=1000):
        super(ResNet50, self).__init__()

        self.act_fn = nn.ReLU()

        self.start_layer = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer2 = nn.Sequential(
            BottleNeck(in_dim = base_dim, mid_dim = base_dim, out_dim = base_dim*4, act_fn = self.act_fn, down=False),
            BottleNeck(in_dim = base_dim*4, mid_dim=base_dim, out_dim=base_dim * 4, act_fn = self.act_fn, down=False),
            BottleNeck(in_dim = base_dim*4, mid_dim=base_dim, out_dim=base_dim * 4, act_fn = self.act_fn, down=True)
        )

        self.layer3 = nn.Sequential(
            Bottleneck(in_dim = base_dim*4, mid_dim = base_dim*2, out_dim = base_dim*8,act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 8, mid_dim=base_dim * 2, out_dim=base_dim * 8, act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 8, mid_dim=base_dim * 2, out_dim=base_dim * 8, act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 8, mid_dim=base_dim * 2, out_dim=base_dim * 8, act_fn = self.act_fn, down=True),
        )

        self.layer4 = nn.Sequetial(
            Bottleneck(in_dim=base_dim * 8, mid_dim=base_dim * 4, out_dim=base_dim * 16, act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 16, mid_dim=base_dim * 4, out_dim=base_dim * 16, act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 16, mid_dim=base_dim * 4, out_dim=base_dim * 16, act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 16, mid_dim=base_dim * 4, out_dim=base_dim * 16, act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 16, mid_dim=base_dim * 4, out_dim=base_dim * 16, act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 16, mid_dim=base_dim * 4, out_dim=base_dim * 16, act_fn = self.act_fn, down=True)

        )

        self.layer5 = nn.Sequential(
            Bottleneck(in_dim=base_dim * 16, mid_dim=base_dim * 6, out_dim=base_dim * 32, act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 32, mid_dim=base_dim * 6, out_dim=base_dim * 32, act_fn = self.act_fn, down=False),
            Bottleneck(in_dim=base_dim * 32, mid_dim=base_dim * 6, out_dim=base_dim * 32, act_fn = self.act_fn, down=False)

        )

        self.AvgPool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc_layer = nn.Linear(base_dim*32, num_classes)

        def forward(self, x):
            out = self.start_layer(x),
            out = self.layer2(out),
            out = self.layer3(out),
            out = self.layer4(out),
            out = self.layer5(out),
            out = self.AvgPool(out),

            out = out.view(out.size(0), -1),
            out = self.fc_layer(out)

            return out
