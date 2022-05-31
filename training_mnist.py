import matplotlib.pyplot as plt
import numpy as np
from torch import optim, nn
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ModelSet
from pylab import mpl
import datetime
from MI_estimator import mutual_info_estimator
from utils import *

mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率

Enable_Show = False
Train_Batch_Size = 128
Forward_Size = 800
Forward_Repeat = 5
Std_Epoch_Num = 20

mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.1, 0.95
mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.1, 0.9
mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.2, 0.4


#
# def plot_performance(data, Enable_Adv_Training):
#     Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
#     save_array_dict(data, 'loss_and_acc')
#     # 想要绘制线条的画需要记号中带有‘-’
#     fig, axs = plt.subplots(1, 4, figsize=(10, 4))
#     for idx, (k, v) in enumerate(data.items()):
#         axs[idx].set_xlabel('epoch')
#         axs[idx].set_title(str(k))
#         axs[idx].plot(v, linestyle='-', linewidth=1)
#     title = 'Adv Training' if Enable_Adv_Training else 'Std Training'
#     fig.suptitle(title)
#     fig = plt.gcf()
#     if Enable_Show:
#         plt.show()
#     # fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
#     fig.savefig('./results_pdf/%s_performance_%s.pdf' % (Is_Adv_Training,
#                                                          datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
#                 )


def plot_mutual_info(std_estimator, adv_estimator, analytic_data, Enable_Adv_Training):
    std, adv = std_estimator, adv_estimator
    save_array_dict(analytic_data, 'loss_and_acc')
    save_array_dict(np.array([std.epoch_MI_hM_X_upper, std.epoch_MI_hM_Y_upper,
                              std.epoch_MI_hM_X_bin, std.epoch_MI_hM_Y_bin,
                              adv.epoch_MI_hM_X_upper, adv.epoch_MI_hM_Y_upper,
                              adv.epoch_MI_hM_X_bin, adv.epoch_MI_hM_Y_bin
                              ]), filename='loss_and_mutual_info')
    # sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
    global Forward_Repeat
    global Forward_Size
    Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
    title = "%s(%s),LR(%.3f),upper_bin,Clean(Adv),Sample_N(%d),%s" % (
        Model_Name, Activation_F, Learning_Rate, Forward_Repeat * Forward_Size, Is_Adv_Training
    )
    # fig size, 先列后行
    nrows = 3
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8), )
    for i in range(nrows - 1):
        for j in range(ncols):
            # axs[0].set_xlim(0, 2)
            if j % 2 == 0:
                axs[i][j].set_xlabel('layers')
                axs[i][j].set_ylabel('I(T;X)')
            # axs[0].grid(True)
            else:
                axs[i][j].set_xlabel('layers')
                axs[i][j].set_ylabel('I(T;Y)')
            # axs[1].grid(True)

    # 开始，结束，步长
    for i in range(Std_Epoch_Num):
        if i % 1 == 0:
            c = sm.to_rgba(i + 1)

            # layers = [i for i in range(1,len(I_TX)+1)]
            # std upper
            I_TX, I_TY = std.epoch_MI_hM_X_upper[i], std.epoch_MI_hM_Y_upper[i]
            axs[0][0].plot(I_TX,
                           color=c, marker='o',
                           linestyle='-', linewidth=1,
                           # zorder=1
                           )
            axs[0][1].plot(I_TY,
                           color=c, marker='o',
                           linestyle='-', linewidth=1,
                           # zorder=1
                           )
            # std bin
            I_TX, I_TY = std.epoch_MI_hM_X_bin[i], std.epoch_MI_hM_Y_bin[i]
            axs[0][2].plot(I_TX,
                           color=c, marker='o',
                           linestyle='-', linewidth=1,
                           # zorder=1
                           )
            axs[0][3].plot(I_TY,
                           color=c, marker='o',
                           linestyle='-', linewidth=1,
                           # zorder=1
                           )
            # adv upper
            I_TX, I_TY = adv.epoch_MI_hM_X_upper[i], adv.epoch_MI_hM_Y_upper[i]
            axs[1][0].plot(I_TX,
                           color=c, marker='o',
                           linestyle='-', linewidth=1,
                           # zorder=1
                           )
            axs[1][1].plot(I_TY,
                           color=c, marker='o',
                           linestyle='-', linewidth=1,
                           # zorder=1
                           )
            # adv bin
            I_TX, I_TY = adv.epoch_MI_hM_X_bin[i], adv.epoch_MI_hM_Y_bin[i]
            axs[1][2].plot(I_TX,
                           color=c, marker='o',
                           linestyle='-', linewidth=1,
                           # zorder=1
                           )
            axs[1][3].plot(I_TY,
                           color=c, marker='o',
                           linestyle='-', linewidth=1,
                           # zorder=1
                           )
            # plt.scatter(I_TX, I_TY,
            #             color=c,
            #             linestyle='-', linewidth=0.1,
            #             zorder=2
            #             )

    for idx, (k, v) in enumerate(analytic_data.items()):
        axs[nrows - 1][idx].set_xlabel('epochs')
        axs[nrows - 1][idx].set_title(str(k))
        axs[nrows - 1][idx].plot(v, linestyle='-', linewidth=1)
    # plt.scatter(epoch_MI_hM_X_upper[0], epoch_MI_hM_Y_upper[0])
    # plt.legend()

    fig.suptitle(title)
    fig.colorbar(sm, ax=axs, label='Epoch')

    fig = plt.gcf()
    if Enable_Show:
        plt.show()
    # fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
    fig.savefig('./results_pdf/%s_mutual_info_%s.pdf' % (Is_Adv_Training,
                                                         datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
                )


data_tf = transforms.Compose([
    # transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# 选择模型
# Activation_F = 'Tanh'
Activation_F = 'ReLU'

from torchvision.models import *
from ModelSet import *
from Models.VGG_s import *

# model, Model_Name = ModelSet.LeNet_3_32_32(), 'LeNet_3_32_32'
# model, Model_Name = VGG_s('VGG11'), 'VGG11'
# model, Model_Name = WideResNet_CIFAR10(depth=1 * 6 + 4, num_classes=10, widen_factor=2, dropRate=0.0), 'WideResNet_CIFAR10'
model, Model_Name = FC_2(Activation_F=torch.nn.ReLU()), 'FC_2'
# model, Model_Name = resnet34(pretrained=False, num_classes=10), 'resnet34'
print("Model Structure\n", model)

Learning_Rate = 0.5
optimizer = optim.SGD(model.parameters(),
                      lr=Learning_Rate,
                      momentum=0.9,
                      weight_decay=2e-4
                      )

milestones = [100]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
# Res18
modules_to_hook = (torch.nn.Sigmoid)
# Res34
# modules_to_hook = ('conv1',
#                    'layer1.2.conv2',
#                    'layer2.3.conv2',
#                    'layer3.5.conv2',
#                    'layer4.2.conv2',
#                    'fc')
# WideResNet_CIFAR10
# modules_to_hook = ('conv1',
#                    'block1.layer.0.relu2',
#                    'block2.layer.0.relu2',
#                    'block3.layer.0.relu2',
#                    'fc')
# net_cifar10, LeNet_3_32_32
# modules_to_hook = ('conv1',
#                    'conv2',
#                    'fc1',
#                    'fc2',
#                    'fc3')
# VGG11
# modules_to_hook = ('features.0',
#                    'features.8',
#                    'features.18',
#                    'features.25',
#                    'classifier')
std_estimator = mutual_info_estimator(modules_to_hook, By_Layer_Name=False)
adv_estimator = mutual_info_estimator(modules_to_hook, By_Layer_Name=False)

Device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
train_dataset = datasets.MNIST(root='../DataSet/MNIST', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='../DataSet/MNIST', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=Train_Batch_Size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=Forward_Size, shuffle=True)


@torch.no_grad()
def evaluate_acc(Keep_Clean):
    from torchattacks import PGD
    atk = PGD(model, eps=45 / 255, alpha=10 / 255, steps=7, random_start=False)
    model.eval()
    correct = 0.
    total = 0.
    for batch_images, batch_labels in test_loader:
        labels = batch_labels.to(Device)
        images = batch_images.to(Device)
        if not Keep_Clean:
            # print('Attacking')
            with torch.enable_grad():
                images = atk(batch_images, batch_labels)
        # forward
        outputs = model(images)

        _, predicted = torch.max(outputs.data, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct * 100. / total
    return acc


@torch.no_grad()
def get_clean_or_adv_image(Keep_Clean):
    from torchattacks import PGD
    atk = PGD(model, eps=45 / 255, alpha=10 / 255, steps=7, random_start=False)

    batch_images, batch_labels = next(iter(test_loader))
    batch_images = batch_images.to(Device)
    batch_labels = batch_labels.to(Device)
    if Keep_Clean:
        return batch_images, batch_labels

    else:
        with torch.enable_grad():
            adv_images = atk(batch_images, batch_labels)
            return adv_images, batch_labels


# 4.1 Standard Accuracy
@torch.no_grad()
def mutual_info_calculate(Keep_Clean):
    # 这里的epoch_i没必要指定，因为epochi就是列表当中的最后一个元素
    # a = list[-1]就是最后一个元素
    import numpy as np
    model.eval()

    image_chunk = []
    label_chunk = []
    layer_activation_chunk = []

    if Keep_Clean:
        estimator = std_estimator
    else:
        estimator = adv_estimator

    for i in range(Forward_Repeat):

        images, labels = get_clean_or_adv_image(Keep_Clean)

        # labels = labels.to(Device)
        # # print('std_test_size', images.size(0))
        # images = images.to(Device)

        """
        forward之前先clear
        """
        estimator.clear_activations()
        # register hook
        estimator.do_forward_hook(model)
        model(images)

        """
        发现并修改了一个重大bug, 这里每forward一次,caculate_MI 函数计算出的互信息值都直接挂在列表的后面，那么 Forward_Repeat 会成倍放大列表的长度
        且会混乱每一个 epoch 中的互信息变化情况，Forward_Repeat 一旦超过 epoch_num ，那么每一个 epoch 的曲线就会
        """
        # 给定初始值
        if i == 0:
            # print("---> layer activations size {} <---".format(layer_activations_size))
            image_chunk = images
            label_chunk = labels
            '''
            注意， 这里如果简单赋值就会出现传递引用的现象，需要手动调用copy,复制列表
            '''
            layer_activation_chunk = estimator.layer_activations.copy()
        # 计算所有循环的和
        else:
            image_chunk = torch.cat((image_chunk, images), dim=0)
            label_chunk = torch.cat((label_chunk, labels), dim=0)
            for idx, item in enumerate(estimator.layer_activations):
                layer_activation_chunk[idx] = torch.cat((layer_activation_chunk[idx], item), dim=0)
        """
        forward 之后例行收尾工作
        """
        estimator.clear_activations()
        estimator.cancel_hook()
    # 计算存储互信息
    # calculate mutual info
    """
    这里 layer_activations 是一个 list， list 里的每一个元素时 tesnor (gpu:0)
    """
    estimator.layer_activations = layer_activation_chunk
    estimator.caculate_MI(image_chunk, label_chunk)
    estimator.store_MI()


# this training function is only for classification task
def training(Enable_Adv_Training):
    global model
    train_clean_loss = []
    train_adv_loss = []
    train_acc = []
    test_clean_acc, test_adv_acc = [], []

    criterion = nn.CrossEntropyLoss()

    # Load checkpoint.
    if Enable_Adv_Training:
        # 装载训练好的模型
        print('--> %s is adv training...' % Model_Name)
        print('--> Loading model state dict..')
        load_model(model, './Checkpoint/%s_std.pth' % Model_Name)
        print('--> Load checkpoint successfully! ')

    model = model.to(Device)
    model.train()

    for epoch_i in range(Std_Epoch_Num):
        train_loss_sum, train_acc_sum, sample_sum = 0.0, 0.0, 0

        # 在每次训练之前，在验证集上计算干净样本和对抗样本互信息
        # if (epoch_i + 1) % 3 == 0:
        mutual_info_calculate(Keep_Clean=True)
        mutual_info_calculate(Keep_Clean=False)

        for batch_images, batch_labels in train_loader:

            # data moved to GPU
            batch_labels = batch_labels.to(Device)
            batch_images = batch_images.to(Device)

            if Enable_Adv_Training:
                from torchattacks import PGD
                atk = PGD(model, eps=8 / 255, alpha=2 / 255, steps=7, random_start=True)
                batch_images = atk(batch_images, batch_labels)

            outputs = model(batch_images)

            if epoch_i == 0 and sample_sum == 0:
                print(Device)
                print(batch_images.shape, batch_labels.shape, outputs.shape)
                # print(batch_labels, outputs)

            loss = criterion(outputs, batch_labels)

            # zero the gradient cache
            optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # update weights and bias
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            _, predicted_label = torch.max(outputs.data, dim=1)
            train_acc_sum += predicted_label.eq(batch_labels.data).cpu().sum().item()
            sample_sum += batch_images.shape[0]

        # 记录每一轮的训练集准确度，损失，测试集准确度
        train_clean_loss.append(train_loss_sum)
        # 训练准确率
        epoch_train_acc = (train_acc_sum / sample_sum) * 100.0
        train_acc.append(epoch_train_acc)
        # 在验证集上的干净样本准确率，对抗样本准确率
        epoch_test_clean_acc = evaluate_acc(Keep_Clean=True)
        test_clean_acc.append(epoch_test_clean_acc)
        epoch_test_adv_acc = evaluate_acc(Keep_Clean=False)
        test_adv_acc.append(epoch_test_adv_acc)

        # print some data
        print('epoch_i[%d], '
              'train loss[%.4f], train acc[%.2f%%], '
              'test clean acc[%.2f%%],test adv acc[%.2f%%]'
              % (epoch_i + 1,
                 train_loss_sum, epoch_train_acc,
                 epoch_test_clean_acc, epoch_test_adv_acc))

    # Save checkpoint.
    file_name = "./Checkpoint/%s_%s.pth" % (Model_Name, 'adv' if Enable_Adv_Training else 'std')
    save_model(model, file_name)

    analytic_data = {
        'train_clean_loss': train_clean_loss,
        'train_accuracy': train_acc,

        'test_clean_acc': test_clean_acc,
        'test_adv_acc': test_adv_acc
    }
    # plot_performance(analytic_data, Enable_Adv_Training)
    plot_mutual_info(std_estimator, adv_estimator, analytic_data, Enable_Adv_Training)

    return analytic_data


analytic_data = training(Enable_Adv_Training=False)
std_estimator.clear_all()
adv_estimator.clear_all()
analytic_data_2 = training(Enable_Adv_Training=True)

print('end')

"""
epoch_MI_hM_X_upper = std_estimator.epoch_MI_hM_X_upper
epoch_MI_hM_Y_upper = std_estimator.epoch_MI_hM_Y_upper
epoch_MI_hM_X_bin = std_estimator.epoch_MI_hM_X_bin
epoch_MI_hM_Y_bin = std_estimator.epoch_MI_hM_Y_bin

epoch_MI_hM_X_upper_adv = adv_estimator.epoch_MI_hM_X_upper
epoch_MI_hM_Y_upper_adv = adv_estimator.epoch_MI_hM_Y_upper
epoch_MI_hM_X_bin_adv = adv_estimator.epoch_MI_hM_X_bin
epoch_MI_hM_Y_bin_adv = adv_estimator.epoch_MI_hM_Y_bin


plt.figure()
plt.xlabel('I(T;X)')
plt.ylabel('I(T;Y)')
# 开始，结束，步长
for i in range(0, Std_Epoch_Num * 2, 2):
    if i % 1 == 0:
        c = sm.to_rgba(i)
        # I_TX, I_TY = epoch_MI_hM_X_upper[i][::-1], epoch_MI_hM_Y_upper[i][::-1]
        I_TX, I_TY = epoch_MI_hM_X_bin[i][::-1], epoch_MI_hM_Y_bin[i][::-1]
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
plt.title("%s(%s),LR(%.3f)" % (model.name, Activation_F, Learning_Rate))
plt.colorbar(sm, label='Epoch')
fig = plt.gcf()
plt.show()
# fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
fig.savefig('./%s_%s_%s_%s_std.pdf' % (
    model.name,
    Activation_F, str(Std_Epoch_Num),
    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))



# 4.1 Standard Accuracy
@torch.no_grad()
def mutual_info_calculate(Keep_Clean=True):
    # 这里的epoch_i没必要指定，因为epochi就是列表当中的最后一个元素
    # a = list[-1]就是最后一个元素
    import numpy as np
    model.eval()
    if Keep_Clean:
        estimator = std_estimator
    else:
        estimator = adv_estimator

    epoch_i_MI_hM_X_upper_sum = None
    epoch_i_MI_hM_Y_upper_sum = None
    epoch_i_MI_hM_X_bin_sum = None
    epoch_i_MI_hM_Y_bin_sum = None

    for i in range(Forward_Repeat):

        images, labels = get_clean_or_adv_image(Keep_Clean)

        labels = labels.to(Device)
        # print('std_test_size', images.size(0))
        images = images.to(Device)

        # register hook
        estimator.do_forward_hook(model)
        # forward
        model(images)
        # calculate mutual info
        estimator.caculate_MI(images, labels)
        layer_activations_size = len(estimator.layer_activations)
        estimator.clear_activations()
        estimator.cancel_hook()
        
        # 发现并修改了一个重大bug, 这里每forward一次,caculate_MI 函数计算出的互信息值都直接挂在列表的后面，那么 Forward_Repeat 会成倍放大列表的长度
        # 且会混乱每一个 epoch 中的互信息变化情况，Forward_Repeat 一旦超过 epoch_num ，那么每一个 epoch 的曲线就会

        # 给定初始值
        if i == 0:
            print("---> layer activations size {} <---".format(layer_activations_size))
            epoch_i_MI_hM_X_upper_sum = np.array(estimator.epoch_i_MI_hM_X_upper)
            epoch_i_MI_hM_Y_upper_sum = np.array(estimator.epoch_i_MI_hM_Y_upper)
            epoch_i_MI_hM_X_bin_sum = np.array(estimator.epoch_i_MI_hM_X_bin)
            epoch_i_MI_hM_Y_bin_sum = np.array(estimator.epoch_i_MI_hM_Y_bin)
        # 计算所有循环的和
        else:
            epoch_i_MI_hM_X_upper_sum += np.array(estimator.epoch_i_MI_hM_X_upper)
            epoch_i_MI_hM_Y_upper_sum += np.array(estimator.epoch_i_MI_hM_Y_upper)
            epoch_i_MI_hM_X_bin_sum += np.array(estimator.epoch_i_MI_hM_X_bin)
            epoch_i_MI_hM_Y_bin_sum += np.array(estimator.epoch_i_MI_hM_Y_bin)
    # 求平均
    estimator.epoch_i_MI_hM_X_upper = (epoch_i_MI_hM_X_upper_sum / Forward_Repeat).tolist()
    estimator.epoch_i_MI_hM_Y_upper = (epoch_i_MI_hM_Y_upper_sum / Forward_Repeat).tolist()
    estimator.epoch_i_MI_hM_X_bin = (epoch_i_MI_hM_X_bin_sum / Forward_Repeat).tolist()
    estimator.epoch_i_MI_hM_Y_bin = (epoch_i_MI_hM_Y_bin_sum / Forward_Repeat).tolist()
    # 存储互信息
    estimator.store_MI()
"""
"""
"""
