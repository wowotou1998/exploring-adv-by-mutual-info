from torchviz import make_dot
from Models.CIFAR10 import WideResNet
from torchsummary import summary
import torch
from Models.MNIST import FC_Sigmoid, Net_mnist, FC_2
from Models.CIFAR10 import LeNet_cifar10, WideResNet, VGG_s, RestNet18, net_cifar10
import argparse

# Model, Model_Name = FC_2(Activation_F=nn.ReLU()), 'FC_2'
Model, Model_Name = LeNet_cifar10(), 'LeNet_cifar10'
# Model, Model_Name = net_cifar10(), 'net_cifar10'
# Model, Model_Name = VGG_s(), 'VGG_s_11'
# Model, Model_Name = resnet18(pretrained=False, num_classes=10), 'resnet18'
# Model, Model_Name = resnet34(pretrained=False, num_classes=10), 'resnet34'
# Model, Model_Name = vgg11(pretrained=False)
# Model, Model_Name = WideResNet(depth=1 * 6 + 4, num_classes=10, widen_factor=1, dropRate=0.0), 'WideResNet'
# Model, Model_Name = WideResNet(depth=1 * 6 + 4, num_classes=10, widen_factor=2, dropRate=0.0), 'WideResNet'
Model.eval()
summary(Model, (3, 32, 32), device="cpu")

# x = torch.randn(1, 3, 32, 32).requires_grad_(True)  # 定义一个网络的输入值
# y = Model(x)  # 获取网络的预测值
#
# MyConvNetVis = make_dot(y, params=dict(list(Model.named_parameters()) + [('x', x)]))
# MyConvNetVis.format = "pdf"
# # 指定文件生成的文件夹
# MyConvNetVis.directory = "Structure"
# # 生成文件
# MyConvNetVis.view()

# import hiddenlayer as h
#
# vis_graph = h.build_graph(Model, torch.zeros([1, 3, 32, 32]))  # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
# vis_graph.save("./Structure/%s_2" % Model_Name)  # 保存图像的路径
