import torch
import torch.nn.functional as F
from torch import nn

'''LeNet_cifar10 in PyTorch.'''

# codes are import from https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
# original author: xternalz

import math


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes,
                               out_planes,
                               kernel_size=3, stride=stride, padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes,
                               out_planes,
                               kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.drop_rate = dropRate

        self.In_equal_Out = (in_planes == out_planes)
        # convShortcut 就是1*1卷积, 用于跳跃分支对x进行升维和降维
        self.convShortcut = (not self.In_equal_Out) and nn.Conv2d(in_planes,
                                                                  out_planes,
                                                                  kernel_size=1, stride=stride, padding=0,
                                                                  bias=False) or None

    def forward(self, x):

        # if not self.In_equal_Out:
        #     x = self.relu1(self.bn1(x))
        # else:
        #     out = self.relu1(self.bn1(x))
        #
        # out = self.relu2(self.bn2(self.conv1(out if self.In_equal_Out else x)))
        #
        # if self.drop_rate > 0:
        #     out = F.dropout(out, p=self.drop_rate, training=self.training)
        # out = self.conv2(out)
        # # 这个torch.add 很重要
        # return torch.add(x if self.In_equal_Out else self.convShortcut(x), out)
        # -------------------------
        y = self.relu1(self.bn1(x))

        if self.In_equal_Out:
            res = x
        else:
            res = self.convShortcut(y)

        out = self.relu2(self.bn2(self.conv1(y)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(out, res)


# NetworkBlock 的作用只想当与一个函数, 用来产生网络中几个大组件模块
class NetworkBlock(nn.Module):
    def __init__(self, layer_repeat, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, layer_repeat, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, layer_repeat, stride, dropRate):
        layers = []
        for i in range(int(layer_repeat)):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes,
                                i == 0 and stride or 1,
                                dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet_3_64_64(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_3_64_64, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        layer_repeat = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0],
                               kernel_size=3, stride=1, padding=1,
                               bias=False)
        # 1st block
        self.block1 = NetworkBlock(layer_repeat, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(layer_repeat, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(layer_repeat, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.fc = nn.Linear(256, num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.modules_to_hook = ('conv1',
                                'block1.layer.0.relu1',
                                'block2.layer.0.relu1',
                                'block3.layer.0.relu1',
                                'fc')
        # self.modules_to_hook = (torch.nn.LeakyReLU,)

    def forward(self, x, _eval=False):
        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        self.train()

        return self.fc(out)


if __name__ == '__main__':
    a = WideResNet(depth=1 * 6 + 4, num_classes=200, widen_factor=1, dropRate=0.0)
    b = a(torch.rand((1, 3, 64, 64)))
