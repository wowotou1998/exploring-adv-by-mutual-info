import os
import time

import torch
from torch import optim, nn
import matplotlib.pyplot as plt


# @torch.no_grad(
# this training function is only for classification task
@torch.no_grad()
def model_forward(model,
                  test_data_loader,
                  gpu_id=0,
                  enable_attack=False
                  ):
    device = torch.device("cuda:%d" % (gpu_id) if torch.cuda.is_available() else "cpu")
    test_acc_sum, n = 0.0, 0
    model = model.to(device)
    model.eval()
    """
    Register a hook for each layer
    """
    estimator_1.do_forward_hook(model)
    # model.do_forward_hook(layer_names, layer_activations, handle_list)

    sample_data, sample_true_label = list(test_data_loader)[0]
    # data moved to GPU or CPU
    sample_data = sample_data.to(device)
    sample_true_label = sample_true_label.to(device)
    sample_predicted_probability_label = model(sample_data)
    _, predicted_label = torch.max(sample_predicted_probability_label.data, 1)
    test_acc_sum += predicted_label.eq(sample_true_label.data).cpu().sum().item()
    # test_acc_sum += (sample_predicted_probability_label.argmax(dim=1) == sample_true_label).sum().item()

    """
    提取正常样本在非鲁棒（或者是鲁棒）神经网络传播过程中的互信息,计算每一层的互信息,使用KDE或者MINE, I(T;X),I(X;Y)
    只有第一个batch计算?, 还是所有batch会计算?, 还是若干batch会计算??
    计算完互信息之后，清空layer_activations，但不取消hook，因为接下来还要计算一次互信息
    sample_true_label是一个一维向量， 里面的元素个数为batch_size
    """
    print("---> layer activations size {} <---".format(len(estimator_1.layer_activations)))
    estimator_1.caculate_MI(sample_data, sample_true_label)
    estimator_1.clear_activations()
    estimator_1.cancel_hook()
    """
    提取对抗样本在非鲁棒（或者是鲁棒）神经网络传播过程中的互信息,计算每一层的互信息,使用KDE， I(T;X),I(X;Y)
    计算完互信息之后，清空layer_activations，取消hook
    """
    if enable_attack:
        from torchattacks import BIM
        atk = BIM(model, eps=5 / 255, alpha=1 / 255, steps=7)
        adv_sample_data = atk(sample_data, sample_true_label).to(device)
        """
        只拦截测试对抗样本时的输出，制造对抗样本时不进行hook
        """
        estimator_1.do_forward_hook(model)
        _ = model(adv_sample_data)
        print("---> layer activations size {} adv<---".format(len(estimator_1.layer_activations)))
        estimator_1.caculate_MI(adv_sample_data, sample_true_label)
        estimator_1.clear_activations()
        estimator_1.cancel_hook()


def show_model_performance(model_data):
    plt.figure()
    # show two accuracy rate at the same figure
    # 想要绘制线条的画需要记号中带有‘-’
    plt.title("the trend of model")
    for k, v in model_data.items():
        plt.plot(v)
    # plt.legend()
    plt.show()


import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ModelSet
from pylab import mpl
import datetime
from MI_estimator import mutual_info_estimator

mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 设置横纵坐标的名称以及对应字体格式
SaveModelPath = ''

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize([0.5], [0.5])
     ]
)

train_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=True, transform=data_tf, download=True)
test_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=False, transform=data_tf)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=500, shuffle=True)

EPOCH_NUM = 1
Learning_Rate = 1e-3
Enable_Attack = False
# 选择模型
from torchvision.models import resnet18, vgg11, inception_v3
from ModelSet import *
from pytorchcv.model_provider import get_model as ptcv_get_model

model = ptcv_get_model("nin_cifar10", pretrained=True, root='../Checkpoint')
Model_Name = 'nin_cifar10'
Activation_F = 'ReLU'
print("Model Structure ", model)
specified_modules = (torch.nn.AvgPool2d, torch.nn.MaxPool2d)
estimator_1 = mutual_info_estimator(specified_modules)
model_forward(model, test_loader, enable_attack=Enable_Attack)
# show_model_performance(acc_record)
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=EPOCH_NUM))


# sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=EPOCH_NUM))

def plot_mutual_info(epoch_MI_hM_X, epoch_MI_hM_Y, title):
    plt.figure()
    plt.xlabel('I(T;X)')
    plt.ylabel('I(T;Y)')
    # 开始，结束，步长
    for i in range(EPOCH_NUM):
        if i % 1 == 0:
            c = sm.to_rgba(i)
            I_TX, I_TY = epoch_MI_hM_X[i][::-1], epoch_MI_hM_Y[i][::-1]
            # I_TX, I_TY = epoch_MI_hM_X_bin[i][::-1], epoch_MI_hM_Y_bin[i][::-1]
            # I_TX, I_TY = epoch_MI_hM_X_mine[i][::-1], epoch_MI_hM_Y_mine[i][::-1]
            plt.plot(I_TX, I_TY,
                     color='lightgrey', marker='o',
                     linestyle='-', linewidth=0.1,
                     zorder=1
                     )
            plt.scatter(I_TX, I_TY,
                        color=c,
                        linestyle='-', linewidth=0.1,
                        zorder=2
                        )

    # plt.scatter(epoch_MI_hM_X_upper[0], epoch_MI_hM_Y_upper[0])
    # plt.legend()

    plt.title(title)
    plt.colorbar(sm, label='Epoch')
    fig = plt.gcf()
    plt.show()
    # fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
    fig.savefig('./%s.pdf' % (
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
                )


epoch_MI_hM_X_upper = estimator_1.epoch_MI_hM_X_upper
epoch_MI_hM_Y_upper = estimator_1.epoch_MI_hM_Y_upper

epoch_MI_hM_X_bin = estimator_1.epoch_MI_hM_X_bin
epoch_MI_hM_Y_bin = estimator_1.epoch_MI_hM_Y_bin

epoch_MI_hM_X_mine = []
epoch_MI_hM_Y_mine = []

if len(epoch_MI_hM_X_upper) > 0:
    title_std = "%s(%s),LR(%.3f),upper,std" % (Model_Name, Activation_F, Learning_Rate)
    plot_mutual_info(epoch_MI_hM_X_upper[0:EPOCH_NUM * 2:2],
                     epoch_MI_hM_Y_upper[0:EPOCH_NUM * 2:2],
                     title_std
                     )
    if Enable_Attack == True:
        title_std = "%s(%s),LR(%.3f),upper,adv" % (Model_Name, Activation_F, Learning_Rate)
        plot_mutual_info(epoch_MI_hM_X_upper[1:EPOCH_NUM * 2:2],
                         epoch_MI_hM_Y_upper[1:EPOCH_NUM * 2:2],
                         title_std
                         )
if len(epoch_MI_hM_X_bin) > 0:
    title_std = "%s(%s),LR(%.3f),bin,std" % (Model_Name, Activation_F, Learning_Rate)
    plot_mutual_info(epoch_MI_hM_X_bin[0:EPOCH_NUM * 2:2],
                     epoch_MI_hM_Y_bin[0:EPOCH_NUM * 2:2],
                     title_std
                     )
    if Enable_Attack == True:
        title_std = "%s(%s),LR(%.3f),bin,adv" % (Model_Name, Activation_F, Learning_Rate)
        plot_mutual_info(epoch_MI_hM_X_bin[0:EPOCH_NUM * 2:2],
                         epoch_MI_hM_Y_bin[0:EPOCH_NUM * 2:2],
                         title_std
                         )
print('end')

print('end')
