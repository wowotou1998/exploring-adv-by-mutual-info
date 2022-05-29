import matplotlib.pyplot as plt
from torch import optim, nn
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ModelSet
from pylab import mpl
import datetime
from MI_estimator import mutual_info_estimator

mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize([0.5], [0.5])
     ]
)

batch_size = 128
Chunk_Size = 100
device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
train_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=True, transform=data_tf, download=True)
test_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=False, transform=data_tf)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)


@torch.no_grad()
def joint_std_images(image_chunk_size):
    std_images, std_labels = None, None
    total_num = 0
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        if total_num == 0:
            std_images = batch_images
            std_labels = batch_labels
        else:
            # 拼接对抗样本
            std_images = torch.cat((std_images, batch_images), dim=0)
            std_labels = torch.cat((std_labels, batch_labels), dim=0)

        total_num += batch_images.size(0)
        # 制造的对抗样本数量>=所需要的数量则停止制造
        if total_num >= image_chunk_size:
            break
    return std_images, std_labels


@torch.enable_grad()
def joint_adv_images(adv_images_num):
    from torchattacks import FGSM

    adv_images, labels = None, None
    total_num = 0
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        atk = FGSM(model, eps=5 / 255)

        adv_batch_images = atk(batch_images, batch_labels)
        if total_num == 0:
            adv_images = adv_batch_images
            labels = batch_labels
        else:
            # 拼接对抗样本
            adv_images = torch.cat((adv_images, adv_batch_images), dim=0)
            labels = torch.cat((labels, batch_labels), dim=0)

        total_num += batch_images.size(0)
        # 制造的对抗样本数量>=所需要的数量则停止制造
        if total_num >= adv_images_num:
            break
    return adv_images, labels


# 4.1 Standard Accuracy
@torch.no_grad()
def evaluate_acc_on_clean():
    model.eval()
    correct = 0.
    total = 0.
    for batch_images, batch_labels in test_loader:
        images = batch_images.to(device)
        labels = batch_labels.to(device)
        # forward
        outputs = model(images)

        _, predicted = torch.max(outputs.data, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    std_acc = correct * 100. / total
    return std_acc


@torch.no_grad()
def mutual_info_adv(images_num):
    model.eval()
    adv_images, labels = joint_adv_images(images_num)
    print('adv_test_size', adv_images.size(0))

    # register hook
    adv_estimator.do_forward_hook(model)
    # forward
    model(adv_images)
    # calculate mutual info
    print("---> layer activations size {} <---".format(len(adv_estimator.layer_activations)))
    adv_estimator.caculate_MI(adv_images, labels)
    adv_estimator.clear_activations()
    adv_estimator.cancel_hook()


# 4.1 Standard Accuracy
@torch.no_grad()
def mutual_info_std(images_num, epoch, repeat_num=2):
    import numpy as np
    model.eval()
    std_estimator.do_forward_hook(model)
    epoch_i_MI_hM_X_upper_sum = None
    epoch_i_MI_hM_Y_upper_sum = None
    epoch_i_MI_hM_X_bin_sum = None
    epoch_i_MI_hM_Y_bin_sum = None
    for i in range(repeat_num):
        images, labels = joint_std_images(images_num)
        # print('std_test_size', images.size(0))
        images = images.to(device)
        labels = labels.to(device)
        # register hook
        # forward
        model(images)
        # calculate mutual info
        layer_activations_size = len(std_estimator.layer_activations)
        std_estimator.caculate_MI(images, labels)
        std_estimator.clear_activations()

        # 给定初始值
        if i == 0:
            print("---> layer activations size {} <---".format(layer_activations_size))
            epoch_i_MI_hM_X_upper_sum = np.array(std_estimator.epoch_MI_hM_X_upper[epoch])
            epoch_i_MI_hM_Y_upper_sum = np.array(std_estimator.epoch_MI_hM_Y_upper[epoch])
            epoch_i_MI_hM_X_bin_sum = np.array(std_estimator.epoch_MI_hM_X_bin[epoch])
            epoch_i_MI_hM_Y_bin_sum = np.array(std_estimator.epoch_MI_hM_Y_bin[epoch])
        # 计算所有循环的和
        else:
            epoch_i_MI_hM_X_upper_sum += np.array(std_estimator.epoch_MI_hM_X_upper[epoch])
            epoch_i_MI_hM_Y_upper_sum += np.array(std_estimator.epoch_MI_hM_Y_upper[epoch])
            epoch_i_MI_hM_X_bin_sum += np.array(std_estimator.epoch_MI_hM_X_bin[epoch])
            epoch_i_MI_hM_Y_bin_sum += np.array(std_estimator.epoch_MI_hM_Y_bin[epoch])
    # 求平均
    std_estimator.epoch_MI_hM_X_upper[epoch] = epoch_i_MI_hM_X_upper_sum / repeat_num
    std_estimator.epoch_MI_hM_Y_upper[epoch] = epoch_i_MI_hM_Y_upper_sum / repeat_num
    std_estimator.epoch_MI_hM_X_bin[epoch] = epoch_i_MI_hM_X_bin_sum / repeat_num
    std_estimator.epoch_MI_hM_Y_bin[epoch] = epoch_i_MI_hM_Y_bin_sum / repeat_num
    std_estimator.cancel_hook()


def adv_training():
    global model
    loss_record = []
    train_acc = []
    test_acc = []
    std_train_rob_acc_record = []

    # 直接计算batch size中的每一个样本的loss，然后再求平均值
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0

    # Load checkpoint.
    print('--> %s is adv_training...' % Model_Name)
    print('--> Loading model state dict..')
    try:
        print('--> Resuming from checkpoint..')
        # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./Checkpoint/%s.pth' % Model_Name)
        model.load_state_dict(checkpoint['model'])
        # data moved to GPU
        model = model.to(device)
        # 必须先将模型进行迁移,才能再装载optimizer,不然会出现数据在不同设备上的错误
        # optimizer.load_state_dict(checkpoint['optimizer'])
        best_test_acc = checkpoint['epoch_test_acc']
        print('--> Load checkpoint successfully!')
    except Exception as e:
        print('--> %s\' checkpoint is not found !' % Model_Name)
    model.train()
    model = model.to(device)

    # train_loader is a iterator object, which contains data and label
    # batch_images is a tensor,the size is batch_size * sample_size
    # batch_labels is the same, which is 1 dim tensor, and the length is batch_size, and each sample
    # has a scalar type value
    """
    on_train_begin
    """
    # on_train_begin(model)
    for epoch in range(Std_Epoch_Num):
        train_loss_sum, train_acc_sum, sample_sum = 0.0, 0.0, 0
        for batch_images, batch_labels in train_loader:

            # data moved to GPU
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_images)

            if epoch == 0 and sample_sum == 0:
                print(device)
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
        loss_record.append(train_loss_sum)
        epoch_train_acc = (train_acc_sum / sample_sum) * 100.0
        epoch_test_acc = evaluate_acc_on_clean()

        train_acc.append(epoch_train_acc)
        test_acc.append(epoch_test_acc)

        mutual_info_std(Chunk_Size, epoch=epoch, repeat_num=2)

        if Enable_Attack:
            robust_acc = mutual_info_adv(Chunk_Size)
            std_train_rob_acc_record.append(robust_acc)

        print('epoch[%d], train loss[%.4f], train acc[%.2f%%], test acc[%.2f%%]'
              % (epoch + 1, train_loss_sum, epoch_train_acc, epoch_test_acc))

        # Save checkpoint.
        if epoch_test_acc > best_test_acc:
            print('Saving.. epoch_test_acc[%.2f%%] > best_test_acc[%.2f%%]' % (epoch_test_acc, best_test_acc))
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_test_acc': epoch_test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('./Checkpoint'):
                os.mkdir('./Checkpoint')
            torch.save(state, './Checkpoint/{}.pth'.format(Model_Name))
            best_test_acc = epoch_test_acc
        else:
            print('Not save.. epoch_test_acc[%.2f%%] < best_test_acc[%.2f%%]' % (epoch_test_acc, best_test_acc))

    analytic_data = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
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


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 设置横纵坐标的名称以及对应字体格式
SaveModelPath = ''
Enable_Attack = False
# modules_to_hook = ('conv1',
#                    'layer1.1.conv2',
#                    'layer2.1.conv2',
#                    'layer3.1.conv2',
#                    'layer4.1.conv2',
#                    'fc')
# WideResNet_CIFAR10
modules_to_hook = ('conv1',
                   'block1.layer.0.relu2',
                   'block2.layer.0.relu2',
                   'block3.layer.0.relu2',
                   'fc')
std_estimator = mutual_info_estimator(modules_to_hook, By_Layer_Name=True)
adv_estimator = mutual_info_estimator(modules_to_hook, By_Layer_Name=True)
Std_Epoch_Num = 50
Learning_Rate = 0.1

# 选择模型
# Activation_F = 'Tanh'
Activation_F = 'ReLU'
# model = ModelSet.FC_with_Sigmoid(Activation_F)
from torchvision.models import vgg11, vgg16, resnet18
from ModelSet import *

model = WideResNet(depth=1 * 6 + 4, num_classes=10, widen_factor=10, dropRate=0.0)
# model = resnet18(pretrained=False, num_classes=10)
Model_Name = 'resnet18'
print("Model Structure\n", model)
optimizer = optim.SGD(model.parameters(),
                      lr=Learning_Rate,
                      momentum=0.9,
                      weight_decay=2e-4
                      )

milestones = [10, 25]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
analytic_data, loss_record, best_test_acc = adv_training()


def plot_mutual_info(epoch_MI_hM_X, epoch_MI_hM_Y, title):
    # sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
    # 2, 1 表示两行一列， 先行后列
    # figsize, 先行后列
    fig, axs = plt.subplots(1, 2, figsize=(6, 2), )

    # axs[0].set_xlim(0, 2)
    axs[0].set_xlabel('layers')
    axs[0].set_ylabel('I(T;X)')
    # axs[0].grid(True)

    axs[1].set_xlabel('layers')
    axs[1].set_ylabel('I(T;Y)')
    # axs[1].grid(True)

    # 开始，结束，步长
    for i in range(Std_Epoch_Num):
        if i % 1 == 0:
            c = sm.to_rgba(i)
            I_TX, I_TY = epoch_MI_hM_X[i][::-1], epoch_MI_hM_Y[i][::-1]
            axs[0].plot(I_TX,
                        color=c, marker='o',
                        linestyle='-', linewidth=1,
                        # zorder=1
                        )
            axs[1].plot(I_TY,
                        color=c, marker='o',
                        linestyle='-', linewidth=1,
                        # zorder=1
                        )
            # plt.scatter(I_TX, I_TY,
            #             color=c,
            #             linestyle='-', linewidth=0.1,
            #             zorder=2
            #             )

    # plt.scatter(epoch_MI_hM_X_upper[0], epoch_MI_hM_Y_upper[0])
    # plt.legend()

    fig.suptitle(title)
    fig.colorbar(sm, label='Epoch')

    fig = plt.gcf()
    plt.show()
    # fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
    fig.savefig('./results_pdf/%s.pdf' % (
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
                )


epoch_MI_hM_X_upper = std_estimator.epoch_MI_hM_X_upper
epoch_MI_hM_Y_upper = std_estimator.epoch_MI_hM_Y_upper
epoch_MI_hM_X_bin = std_estimator.epoch_MI_hM_X_bin
epoch_MI_hM_Y_bin = std_estimator.epoch_MI_hM_Y_bin

epoch_MI_hM_X_upper_adv = adv_estimator.epoch_MI_hM_X_upper
epoch_MI_hM_Y_upper_adv = adv_estimator.epoch_MI_hM_Y_upper
epoch_MI_hM_X_bin_adv = adv_estimator.epoch_MI_hM_X_bin
epoch_MI_hM_Y_bin_adv = adv_estimator.epoch_MI_hM_Y_bin
epoch_MI_hM_X_mine = []
epoch_MI_hM_Y_mine = []

if len(epoch_MI_hM_X_upper) > 0:
    title_std = "%s(%s),LR(%.3f),upper,std" % (Model_Name, Activation_F, Learning_Rate)
    plot_mutual_info(epoch_MI_hM_X_upper,
                     epoch_MI_hM_Y_upper,
                     title_std
                     )
    title_std = "%s(%s),LR(%.3f),bin,std" % (Model_Name, Activation_F, Learning_Rate)
    plot_mutual_info(epoch_MI_hM_X_bin,
                     epoch_MI_hM_Y_bin,
                     title_std
                     )
    if Enable_Attack:
        title_std = "%s(%s),LR(%.3f),upper,adv" % (Model_Name, Activation_F, Learning_Rate)
        plot_mutual_info(epoch_MI_hM_X_upper_adv,
                         epoch_MI_hM_Y_upper_adv,
                         title_std
                         )

        title_std = "%s(%s),LR(%.3f),bin,adv" % (Model_Name, Activation_F, Learning_Rate)
        plot_mutual_info(epoch_MI_hM_X_bin_adv,
                         epoch_MI_hM_Y_bin_adv,
                         title_std
                         )
print('end')

"""
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
"""
