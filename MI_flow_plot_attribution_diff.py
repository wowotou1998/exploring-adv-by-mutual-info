# coding = UTF-8
import copy

import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy
from pylab import mpl
import torch.nn.functional as F

import pickle
from matplotlib import ticker
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel

from torchcam.methods import GradCAM, LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


# 相似性范围从-1到1：
# -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
# 0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。
def get_test_loader(Data_Set):
    data_tf_test = transforms.Compose([
        transforms.ToTensor(),
        # Saturation_Transform(saturation_level=1024.),
        # Patch_Transform(k=4),
        # Extra_Transform
    ])

    data_tf_mnist = transforms.Compose([
        transforms.ToTensor(),
    ])

    if Data_Set == 'CIFAR10':
        # train_dataset = datasets.CIFAR10(root='./DataSet/CIFAR10', train=True, transform=data_tf_cifar10,
        #                                  download=True)
        test_dataset = datasets.CIFAR10(root='./DataSet/CIFAR10', train=False, transform=data_tf_test)
    elif Data_Set == 'STL10':
        # train_dataset = datasets.CIFAR10(root='./DataSet/CIFAR10', train=True, transform=data_tf_cifar10,
        #                                  download=True)
        test_dataset = datasets.STL10(root='./DataSet/STL10', split='test', transform=data_tf_test)
    elif Data_Set == 'MNIST':
        # train_dataset = datasets.MNIST(root='./DataSet/MNIST', train=True, transform=data_tf_mnist, download=True)
        test_dataset = datasets.MNIST(root='./DataSet/MNIST', train=False, transform=data_tf_mnist)
    else:
        print(Data_Set)
        raise RuntimeError('Unknown Dataset')

    # Train_Loader = DataLoader(dataset=train_dataset, batch_size=self.Train_Batch_Size, shuffle=True)
    Test_Loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    return Test_Loader


def show_one_image(subplot, images, title, color):
    # C*H*W-->H*W*C
    c, h, w = images[0].shape
    image = numpy.transpose(images[0].cpu().detach().numpy(), (1, 2, 0))
    if c == 1:
        subplot.imshow(image, 'gray')
    else:
        subplot.imshow(images)
    # subplot.axis('off')  # 关掉坐标轴为 off
    # 显示坐标轴但是无刻度
    subplot.set_xticks([])
    subplot.set_yticks([])
    # 设定图片边框粗细
    subplot.spines['top'].set_linewidth('2.0')  # 设置边框线宽为2.0
    subplot.spines['bottom'].set_linewidth('2.0')  # 设置边框线宽为2.0
    subplot.spines['left'].set_linewidth('2.0')  # 设置边框线宽为2.0
    subplot.spines['right'].set_linewidth('2.0')  # 设置边框线宽为2.0
    # 设定边框颜色
    subplot.spines['top'].set_color(color)
    subplot.spines['bottom'].set_color(color)
    subplot.spines['left'].set_color(color)
    subplot.spines['right'].set_color(color)
    # subplot.set_title(title, y=-0.25, color=color, fontsize=8)  # 图像题目


def attribute_image_features(net, algorithm, input, label, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=label,
                                              **kwargs
                                              )

    return tensor_attributions


def plot_attributions(net, image, label, suptitle='suptitle'):
    def norm(attribution_abs):
        attr_min, attr_max = attribution_abs.min().item(), attribution_abs.max().item()
        attributions_abs_img = (attribution_abs - attr_min) / \
                               (attr_max - attr_min)
        return attributions_abs_img[0].cpu().detach().numpy().transpose(1, 2, 0)

    saliency = Saliency(net)
    NT = NoiseTunnel(saliency)
    IG = IntegratedGradients(net)
    s1 = attribute_image_features(net, saliency, image, label)  # labels[0].item()
    s2 = attribute_image_features(net, NT, image, label, nt_type='smoothgrad', nt_samples=100, stdevs=0.2)
    s3 = attribute_image_features(net, IG, image, label, baselines=images * 0, )

    cam = GradCAM(model, 'block3')
    scores = model(image)
    activation_map = cam(class_idx=label, scores=scores)
    s4 = overlay_mask(to_pil_image(image[0]), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

    extractor = LayerCAM(model, ['block1', 'block2', 'block3', ])
    scores = model(image)
    cams = extractor(class_idx=label, scores=scores)
    fused_cam = extractor.fuse_cams(cams)

    s5 = overlay_mask(to_pil_image(image[0]), to_pil_image(fused_cam[0].squeeze(0), mode='F'), alpha=0.5)

    num = 4
    fig, axes = plt.subplots(1, num, figsize=(2 * num, 2), layout='constrained', )
    fig.suptitle(suptitle)

    for i in range(num):
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    axes[0].imshow(image[0].cpu().detach().numpy().transpose(1, 2, 0))
    axes[0].set_title('Original')

    axes[1].imshow(norm(torch.abs(s1)))
    axes[1].set_title('Saliency map')
    # add color bar
    # s_cmap_std = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=attr_min, vmax=attr_max))
    # fig.colorbar(s_cmap_std, ax=axes[1], ticks=[attr_min, 0.5 * (attr_max - attr_min), attr_max])

    axes[2].imshow(norm(torch.abs(s2)))
    axes[2].set_title('SmoothGrad')

    axes[3].imshow(norm(torch.abs(s3)))
    axes[3].set_title('Integrated Gradient')

    # axes[4].imshow(s4)
    # axes[4].set_title('GradCAM')
    #
    # axes[5].imshow(s5)
    # axes[5].set_title('LayerCAM')

    plt.show()
    # plt.show(block=True)
    import datetime
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fig.savefig('attri_diff_%s_%s.pdf' % (suptitle, current_time))
    # print(s1, s2, s3)
    # grads = np.transpose(s1.squeeze().cpu().detach().numpy(), (1, 2, 0))


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    # 解决保存图像是负号'-'显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['savefig.dpi'] = 200  # 保存图片分辨率
    mpl.rcParams['figure.dpi'] = 200  # 分辨率
    mpl.rcParams['figure.constrained_layout.use'] = True
    # mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.05, 0.99
    # mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.07, 0.99
    # mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.1005, 0.1005
    # plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    # plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    # mpl.rcParams['figure.constrained_layout.use'] = True

    # 生成随机数，以便固定后续随机数，方便复现代码
    import random

    random.seed(1234)
    # 没有使用GPU的时候设置的固定生成的随机数
    np.random.seed(1234)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(1234)
    # torch.cuda.manual_seed()为当前GPU设置随机种子
    torch.cuda.manual_seed(1234)
    from utils import load_model

    Model_dict = {}
    # Model_dict['net_cifar10'] = net_cifar10()
    # Model_dict['VGG_s'] = VGG_s()
    # Model_dict['resnet18'] = resnet18(pretrained=False, num_classes=10)
    # Model_dict['resnet34'] = resnet34(pretrained=False, num_classes=10)
    # Model_dict['vgg11'] = vgg11(pretrained=False)
    # Model_dict['FC_2'] = FC_2(Activation_F=nn.ReLU())
    # Model_dict['LeNet_MNIST'] = LeNet_3_32_32()
    from Models.CIFAR10 import WideResNet
    from Models.Tiny_ImageNet import WideResNet_3_96_96

    Model_dict['WideResNet_CIFAR10'] = WideResNet(depth=1 * 6 + 4, num_classes=10, widen_factor=1, dropRate=0.0)
    Model_dict['WideResNet_STL10'] = WideResNet_3_96_96(depth=1 * 6 + 4, num_classes=10, widen_factor=1,
                                                        dropRate=0.0)
    Model_Name = 'WideResNet_STL10'
    model = WideResNet_3_96_96(depth=1 * 6 + 4, num_classes=10, widen_factor=1,
                               dropRate=0.0)
    model_adv = copy.deepcopy(model)

    load_model(model, './Checkpoint/%s/%s_std.pth' % (Model_Name, Model_Name))
    load_model(model_adv, './Checkpoint/%s/%s_adv.pth' % (Model_Name, Model_Name))
    print('--> Load %s checkpoint successfully! ' % Model_Name)
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model_adv = model_adv.to(device)
    model.eval()
    model_adv.eval()

    test_loader = get_test_loader('STL10')

    # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
    # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
    sample_num = 0
    for data in test_loader:
        original_images, original_labels = data
        original_images = original_images.to(device)
        original_labels = original_labels.to(device)
        # _, predict = torch.max(F.softmax(model(original_images), dim=1), 1)
        # # 选择预测正确的original_images和original_labels，剔除预测不正确的original_images和original_labels
        # # predict_answer为一维向量，大小为batch_size
        # predict_answer = (original_labels == predict)
        # # torch.nonzero会返回一个二维矩阵，大小为（nozero的个数）*（1）
        # no_zero_predict_answer = torch.nonzero(predict_answer)
        # # 我们要确保 predict_correct_index 是一个一维向量,因此使用flatten,其中的元素内容为下标
        # predict_correct_index = torch.flatten(no_zero_predict_answer)
        # # print('predict_correct_index', predict_correct_index)
        # images = torch.index_select(original_images, 0, predict_correct_index)
        # labels = torch.index_select(original_labels, 0, predict_correct_index)

        images = original_images
        labels = original_labels

        sample_num += 1
        if images.shape[0] > 0:
            print('correct prediction')
            images.requires_grad = True
            plot_attributions(model, images.clone().detach(), labels[0].item(), 'Standard training')
            plot_attributions(model_adv, images.clone().detach(), labels[0].item(), 'Adversarially training')
            # break
        else:
            print('wrong prediction')

        if sample_num >= 3:
            break

    # select_a_sample_to_plot(
    #     'ImageNet',
    #     'ResNet34_ImageNet'
    # )

    print("----ALL WORK HAVE BEEN DONE!!!----")

'''
    # sharex 和 sharey 表示坐标轴的属性是否相同，可选的参数：True，False，row，col，默认值均为False，表示画布中的四个ax是相互独立的；
    # True 表示所有子图的x轴（或者y轴）标签是相同的，
    # row 表示每一行之间的子图的x轴（或者y轴）标签是相同的（不同行的子图的轴标签可以不同），
    # col表示每一列之间的子图的x轴（或者y轴）标签是相同的（不同列的子图的轴标签可以不同）
'''
