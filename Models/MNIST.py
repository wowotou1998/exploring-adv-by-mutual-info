from torch import nn
import torch.nn.functional as F


# [1024, 20, 20, 20]
class FC_Sigmoid(nn.Module):
    def __init__(self, Activation_F=nn.ReLU()):
        super().__init__()
        self.name = "FC_784_1024_20(3)_10_Sigmoid"
        # 最后是三个全连接层
        self.seq = nn.Sequential(
            nn.Linear(1 * 28 * 28, 784),
            Activation_F,
            nn.Linear(784, 1024),
            Activation_F,
            nn.Linear(1024, 20),
            Activation_F,
            nn.Linear(20, 20),
            Activation_F,
            nn.Linear(20, 20),
            Activation_F,
            nn.Linear(20, 10),
            nn.Sigmoid()
        )

    def do_forward_hook(self, layer_names, layer_activations, handle_list):
        for idx, layer_i in enumerate(self.seq):
            if isinstance(layer_i, (nn.ReLU, nn.Tanh)):  # 是元组中的一个返回 True
                layer_names.append(str(idx))
                handle = layer_i.register_forward_hook(
                    lambda torch_nn, input, output:
                    layer_activations.append(output.view(output.size(0), -1)))
                # 保存handle对象， 方便随时取消hook
                handle_list.append(handle)

    def get_selected_layers(self):
        layers = []
        for layer_i in self.seq:
            if isinstance(layer_i, (nn.ReLU, nn.Tanh)):
                layers.append(layer_i)
        return layers

    def forward(self, x):
        x = x.view(x.size(0), -1)
        """
        只要确保模型的forward只需要一步就可以计算出中间结果， 获取各层的激活值就没啥问题
        问题解决了， hook要选取指定的层时，使用named_modules, 
        千万不要使用 named_children!!!
        千万不要使用 named_children!!!
        千万不要使用 named_children!!!
        """
        y = self.seq(x)
        return y


class FC_2(nn.Module):
    def __init__(self, Activation_F=nn.ReLU()):
        super().__init__()
        self.name = "FC_784_1024_20(3)_10"
        # 最后是三个全连接层
        '''
        实验证明， 用瓶颈网络来压缩数据维数不大可行， 很难训练
        '''
        self.seq = nn.Sequential(
            nn.Linear(1 * 28 * 28, 784),
            # nn.Linear(784, 10),
            # nn.Linear(10, 784),
            Activation_F,
            nn.Linear(784, 1024),
            # nn.Linear(1024, 10),
            # nn.Linear(10, 1024),
            Activation_F,
            nn.Linear(1024, 20),
            Activation_F,
            nn.Linear(20, 20),
            Activation_F,
            nn.Linear(20, 20),
            Activation_F,
            nn.Linear(20, 10),
        )
        self.modules_to_hook = (nn.Linear,)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.seq(x)
        return y


class FC_3(nn.Module):
    def __init__(self, Activation_F):
        super().__init__()
        self.name = "FC_784_1024_20(3)_10"
        # 最后是三个全连接层
        self.seq = nn.Sequential(
            nn.Linear(1 * 28 * 28, 784),
            Activation_F,
            nn.Linear(784, 1024),
            Activation_F,
            nn.Linear(1024, 20),
            Activation_F
        )
        self.seq_2 = nn.Sequential(
            nn.Linear(20, 20),
            Activation_F,
            nn.Linear(20, 20),
            Activation_F,
            nn.Linear(20, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.seq(x)
        z = self.seq_2(y)
        return z


class LeNet_1_28_28(nn.Module):
    def __init__(self):
        super(LeNet_1_28_28, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.modules_to_hook = (nn.Conv2d, nn.Linear)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Net_mnist(nn.Module):
    def __init__(self):
        # the input size = Batch *1*28*28
        super(Net_mnist, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        r1 = self.seq(x)
        r1 = r1.view(r1.size(0), -1)
        r2 = self.seq2(r1)
        return r2
