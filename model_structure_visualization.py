from torchviz import make_dot
from Models.CIFAR10 import WideResNet
import torch

MyConvNet, Model_Name = WideResNet(depth=1 * 6 + 4, num_classes=10, widen_factor=2, dropRate=0.0), 'WideResNet'
MyConvNet.eval()
x = torch.randn(1, 3, 32, 32).requires_grad_(True)  # 定义一个网络的输入值
y = MyConvNet(x)  # 获取网络的预测值

MyConvNetVis = make_dot(y, params=dict(list(MyConvNet.named_parameters()) + [('x', x)]))
MyConvNetVis.format = "pdf"
# 指定文件生成的文件夹
MyConvNetVis.directory = "Structure"
# 生成文件
MyConvNetVis.view()

import hiddenlayer as h

vis_graph = h.build_graph(MyConvNet, torch.zeros([1, 3, 32, 32]))  # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
vis_graph.save("./Structure/%s" % Model_Name)  # 保存图像的路径
