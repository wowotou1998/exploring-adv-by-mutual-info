import os
import time

import torch
from torch import optim, nn
import matplotlib.pyplot as plt
from utils import *


# @torch.no_grad()
def evaluate_accuracy(test_data_loader, model, device):
    test_acc_sum, n = 0.0, 0
    model = model.to(device)
    model.eval()
    """
    Register a hook for each layer
    """
    estimator_1.do_forward_hook(model)
    # model.do_forward_hook(layer_names, layer_activations, handle_list)

    for sample_data, sample_true_label in test_data_loader:
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
        estimator_1.store_MI()
        estimator_1.cancel_hook()
        """
        提取对抗样本在非鲁棒（或者是鲁棒）神经网络传播过程中的互信息,计算每一层的互信息,使用KDE， I(T;X),I(X;Y)
        计算完互信息之后，清空layer_activations，取消hook
        """
        from torchattacks import BIM
        atk = BIM(model, eps=45 / 255, alpha=10 / 255, steps=5)
        adv_sample_data = atk(sample_data, sample_true_label).cuda()
        """
        只拦截测试对抗样本时的输出，制造对抗样本时不进行hook
        """
        estimator_1.do_forward_hook(model)
        _ = model(adv_sample_data)
        print("---> layer activations size {} adv<---".format(len(estimator_1.layer_activations)))
        estimator_1.caculate_MI(adv_sample_data, sample_true_label)
        estimator_1.store_MI()
        estimator_1.clear_activations()
        estimator_1.cancel_hook()
        # named_children只输出了layer1和layer2两个子module，而named_modules输出了包括layer1和layer2下面所有的modolue。
        # 这两者均是迭代器
        n += sample_data.shape[0]
        break

    return (test_acc_sum / n) * 100.0


# this training function is only for classification task
def training(model,
             train_data_loader, test_data_loader,
             epochs, criterion, optimizer,
             enable_cuda,
             gpu_id=0,
             load_model_args=False,
             model_name='MNIST',
             ):
    loss_record, train_accuracy_record, test_accuracy_record = [], [], []

    # ---------------------------------------------------------------------
    if enable_cuda:
        device = torch.device("cuda:%d" % (gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=0.001)

    if criterion is None:
        # 直接计算batch size中的每一个样本的loss，然后再求平均值
        criterion = nn.CrossEntropyLoss()

    best_test_acc = 0
    start_epoch = 0

    # Load checkpoint.
    print('--> %s is training...' % model_name)
    # if load_model_args:
    #     print('--> Loading model state dict..')
    #     try:
    #         print('--> Resuming from checkpoint..')
    #         # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #         checkpoint = torch.load('../Checkpoint/%s.pth' % model_name)
    #         model.load_state_dict(checkpoint['model'])
    #         # data moved to GPU
    #         model = model.to(device)
    #         # 必须先将模型进行迁移,才能再装载optimizer,不然会出现数据在不同设备上的错误
    #         # optimizer.load_state_dict(checkpoint['optimizer'])
    #         best_test_acc = checkpoint['test_acc']
    #         start_epoch = checkpoint['epoch']
    #         print('--> Load checkpoint successfully! ')
    #     except Exception as e:
    #         print('--> %s\' checkpoint is not found ! ' % model_name)

    model = model.to(device)
    model.train()
    # train_data_loader is a iterator object, which contains data and label
    # sample_data is a tensor,the size is batch_size * sample_size
    # sample_true_label is the same, which is 1 dim tensor, and the length is batch_size, and each sample
    # has a scalar type value
    """
    on_train_begin
    """
    # on_train_begin(model)
    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss_sum, train_acc_sum, sample_sum = 0.0, 0.0, 0
        for sample_data, sample_true_label in train_data_loader:

            # data moved to GPU
            sample_data = sample_data.to(device)
            sample_true_label = sample_true_label.to(device)
            sample_predicted_probability_label = model(sample_data)

            if epoch == 0 and sample_sum == 0:
                print(device)
                print(sample_data.shape, sample_true_label.shape, sample_predicted_probability_label.shape)
                # print(sample_true_label, sample_predicted_probability_label)

            # loss = criterion(sample_predicted_probability_label, sample_true_label).sum()
            loss = criterion(sample_predicted_probability_label, sample_true_label)

            # zero the gradient cache
            optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # update weights and bias
            optimizer.step()

            train_loss_sum += loss.item()
            # argmax(dim=1) 中dim的不同值表示不同维度，argmax(dim=1) 返回列中最大值的下标
            # 特别的在dim=0表示二维中的行，dim=1在二维矩阵中表示列
            # train_acc_sum 表示本轮,本批次中预测正确的个数
            _, predicted_label = torch.max(sample_predicted_probability_label.data, 1)
            train_acc_sum += predicted_label.eq(sample_true_label.data).cpu().sum().item()
            # train_acc_sum += (sample_predicted_probability_label.argmax(dim=1) == sample_true_label).sum().item()
            # sample_data.shape[0] 为本次训练中样本的个数,一般大小为batch size
            # 如果总样本个数不能被 batch size整除的情况下，最后一轮的sample_data.shape[0]比batch size 要小
            # n 实际上为 len(train_data_loader)
            sample_sum += sample_data.shape[0]
            # if sample_sum % 30000 == 0:
            #     print('sample_sum %d' % (sample_sum))
            # if epochs == 1:
            #     print('GPU Memory was locked!')
            #     while True:
            #         pass

        # 每一轮都要干的事
        train_acc = (train_acc_sum / sample_sum) * 100.0
        test_acc = evaluate_accuracy(test_data_loader, model, device)

        # Save checkpoint.
        Enable_Adv_Training = False
        file_name = "./Checkpoint/%s_%s.pth" % (Model_Name, 'adv' if Enable_Adv_Training else 'std')
        save_model(model, file_name)
        # 记录每一轮的训练集准确度，损失，测试集准确度
        loss_record.append(train_loss_sum)
        train_accuracy_record.append(train_acc)
        test_accuracy_record.append(test_acc)

        print('epoch %d, train loss %.4f, train acc %.4f%%, test acc %.4f%%'
              % (epoch + 1, train_loss_sum, train_acc, test_acc))

    analytic_data = {
        'train_accuracy': train_accuracy_record,
        'test_accuracy': test_accuracy_record
    }

    return analytic_data, loss_record, best_test_acc


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

# train_dataset = datasets.MNIST(root='../DataSet/MNIST', train=True, transform=data_tf, download=True)
# test_dataset = datasets.MNIST(root='../DataSet/MNIST', train=False, transform=data_tf)

train_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=True, transform=data_tf, download=True)
test_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=False, transform=data_tf)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=800, shuffle=True)
modules_to_hook = ('conv1',
                   'conv2',
                   'fc1',
                   'fc2',
                   'fc3')
estimator_1 = mutual_info_estimator(modules_to_hook, By_Layer_Name=True)
EPOCH_NUM = 10
Learning_Rate = 0.1
# 选择模型
# Activation_F = 'Tanh'
Activation_F = 'ReLU'
activation_f = torch.nn.ReLU() if Activation_F == 'ReLU' else torch.nn.Tanh()
model = ModelSet.Alex_1_cifar10()
# model = ModelSet.FC_Sigmoid(activation_f)
Model_Name = 'LeNet_cifar10'
print("Model Structure ", model)
acc_record, loss_record, best_acc = training(model=model,
                                             train_data_loader=train_loader,
                                             test_data_loader=test_loader,
                                             epochs=EPOCH_NUM,
                                             criterion=None,
                                             optimizer=optim.SGD(model.parameters(),
                                                                 lr=Learning_Rate,
                                                                 momentum=0.9
                                                                 ),
                                             enable_cuda=True,
                                             model_name=Model_Name
                                             )
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
    # fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
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
    title_std = "%s(%s),LR(%.3f),bin,adv" % (Model_Name, Activation_F, Learning_Rate)
    plot_mutual_info(epoch_MI_hM_X_bin[1:EPOCH_NUM * 2:2],
                     epoch_MI_hM_Y_bin[1:EPOCH_NUM * 2:2],
                     title_std
                     )
print('end')

"""
plt.figure()
plt.xlabel('I(T;X)')
plt.ylabel('I(T;Y)')
# 开始，结束，步长
for i in range(0, EPOCH_NUM * 2, 2):
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
    Activation_F, str(EPOCH_NUM),
    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
"""
