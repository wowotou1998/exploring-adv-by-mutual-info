from torch import optim, nn
import matplotlib.pyplot as plt
from simple_bin import bin_calc_information2
from MINE import calculate_MI_MINE


def cancel_hook(handle_list):
    # print("handle list len", len(handle_list))
    for handle in handle_list:
        handle.remove()
    """
    执行完remove()操作后 清除handle_list列表
    """
    handle_list.clear()


"""
hook 函数钩住的对象一旦不是模型上有关系的对象，就会出很多问题
"""


def hook(layer, input, output):
    # print("before hook, ", len(layer_activations))
    # print(layer)
    layer_activations.append(output.detach().clone().view(output.size(0), -1))
    # print("after hook, ", len(layer_activations))


def do_forward_hook(model):
    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and
        #         not isinstance(m, torch.nn.Sequential) and
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.ReLU):
            handle = m.register_forward_hook(hook)
            handle_list.append(handle)

    """
    named_children， hook的bug之源
    """
    # for name_i, layer_i in model.named_children():
    #     if isinstance(layer_i, nn.Sequential):
    #         for name_ii, layer_ii in layer_i.named_children():
    #             if isinstance(layer_ii, (nn.ReLU, nn.Tanh)):  # 是元组中的一个返回 True
    #                 layer_names.append(name_i + name_ii)
    #                 handle = layer_ii.register_forward_hook(hook)
    #                 # 保存handle对象， 方便随时取消hook
    #                 handle_list.append(handle)


def on_train_begin(model):
    for name, layer in model.named_children():
        # layer.__name__ = name
        layer_names.append(name)


#
# def on_epoch_begin():
#     pass
#
#
# def on_batch_begin():
#     pass
#
#
# def on_epoch_end():
#     pass

"""
此处注意， 
layer_names 和 layer_activations 不一定是一一对应的， 
layer_activations 可能会包含 nn.sequential 中的子模块， 可能实际 layer_activations 的元素要远远超出模型中 nn.sequential 的大小
只要是 torch.nn 类， hook 函数都会钩住
"""
handle_list = []
layer_names = []
layer_activations = []

from pytorch_kde import *
import torch.nn.functional as F


def caculate_MI(X, layer_activations, Y):
    MI_hM_X_upper = []
    MI_hM_Y_upper = []

    MI_hM_X_bin = []
    MI_hM_Y_bin = []

    MI_hM_X_mine = []
    MI_hM_Y_mine = []

    label_num = 10
    noise_variance = 0.1
    nats2bits = 1.0 / np.log(2)
    # print("layer_activations len", len(layer_activations))

    Y_one_hot = F.one_hot(Y, num_classes=label_num).float().to(Y.device)
    Y_probs = torch.mean(Y_one_hot, dim=0)
    Y_i_idx = []
    for i in range(label_num):
        """
        获取标签Y中等于label_i的下标集合, pytorch中的tensor可以使用布尔索引,布尔索引中的元素要为布尔值
        """
        Y_equal_label_i_index = torch.flatten(Y == i)
        Y_i_idx.append(Y_equal_label_i_index)

    saved_label_idx = {}
    for idx, value in enumerate(Y_i_idx):
        saved_label_idx[idx] = value.detach().cpu().clone().numpy()

    for layer_i in range(len(layer_activations)):

        # -------- I(T;X), I(T;Y)  MINE --------
        if DO_MINE:
            MI_hM_X_mine_i = calculate_MI_MINE(layer_activations[layer_i], X)
            MI_hM_Y_mine_i = calculate_MI_MINE(layer_activations[layer_i], Y)
            MI_hM_X_mine.append(nats2bits * MI_hM_X_mine_i)
            MI_hM_Y_mine.append(nats2bits * MI_hM_Y_mine_i)

        # -------- I(T;X), I(T;Y)  binning --------
        if DO_BIN:
            MI_hM_X_bin_i, MI_hM_Y_bin_i = bin_calc_information2(saved_label_idx,
                                                                 layer_activations[
                                                                     layer_i].detach().cpu().clone().numpy(),
                                                                 0.5)
            MI_hM_X_bin.append(nats2bits * MI_hM_X_bin_i)
            MI_hM_Y_bin.append(nats2bits * MI_hM_Y_bin_i)

        if DO_UPPER:
            # 最后一层输出\hat{y}也可以直接使用KDE来计算互信息,
            # 因为\hat{y}仅仅只是预测值,不是真实的标签y, 自然也可以当成隐藏层来计算互信息
            # -------- I(T;X) upper --------
            hM_upper = entropy_estimator_kl_simple(layer_activations[layer_i], noise_variance)
            hM_given_X = kde_multivariate_gauss_entropy(layer_activations[layer_i], noise_variance)
            MI_hM_X_upper.append(nats2bits * (hM_upper - hM_given_X))

            # -------- I(T;Y) upper --------
            hM_given_Y_upper = 0.
            for y_i in range(label_num):
                """
                依次选择激活层i中有关于标签j的激活值， 并计算这部分激活值的的互信息
                """
                # 获取第i层的激活值
                layer_i_activations = layer_activations[layer_i]
                # 获取第i层激活值关于标签i的部分， 使用bool索引
                activation_i_for_Y_i = layer_i_activations[Y_i_idx[y_i], :]
                hM_given_Y_i_upper = entropy_estimator_kl_simple(activation_i_for_Y_i, noise_variance)
                hM_given_Y_upper += Y_probs[y_i].item() * hM_given_Y_i_upper

            MI_hM_Y_upper.append(nats2bits * (hM_upper - hM_given_Y_upper))
    # 在计算完所有层的互信息之后，添加到epoch list中
    if DO_MINE:
        epoch_MI_hM_X_mine.append(MI_hM_X_mine)
        epoch_MI_hM_Y_mine.append(MI_hM_Y_mine)
    if DO_BIN:
        epoch_MI_hM_X_bin.append(MI_hM_X_bin)
        epoch_MI_hM_Y_bin.append(MI_hM_Y_bin)
    if DO_UPPER:
        epoch_MI_hM_X_upper.append(MI_hM_X_upper)
        epoch_MI_hM_Y_upper.append(MI_hM_Y_upper)


@torch.no_grad()
def evaluate_accuracy(test_data_loader, model, device):
    test_acc_sum, n = 0.0, 0
    model = model.to(device)
    model.eval()
    """
    Register a hook for each layer
    """
    do_forward_hook(model)
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
        开始提取互信息,计算每一层的互信息,使用KDE或者MINE, I(T;X),I(X;Y)
        只有第一个batch计算?, 还是所有batch会计算?, 还是若干batch会计算??
        计算完互信息之后，取消hook，清空layer_activations
        sample_true_label是一个一维向量， 里面的元素个数为batch_size
        """
        print("--->layer activations size {} <---".format(len(layer_activations)))
        caculate_MI(sample_data, layer_activations, sample_true_label)

        cancel_hook(handle_list)
        layer_activations.clear()
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
    if load_model_args:
        print('--> Loading model state dict..')
        try:
            print('--> Resuming from checkpoint..')
            # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('../Checkpoint/%s.pth' % model_name)
            model.load_state_dict(checkpoint['model'])
            # data moved to GPU
            model = model.to(device)
            # 必须先将模型进行迁移,才能再装载optimizer,不然会出现数据在不同设备上的错误
            # optimizer.load_state_dict(checkpoint['optimizer'])
            best_test_acc = checkpoint['test_acc']
            start_epoch = checkpoint['epoch']
            print('--> Load checkpoint successfully! ')
        except Exception as e:
            print('--> %s\' checkpoint is not found ! ' % model_name)

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
        if test_acc > best_test_acc:
            print('Saving.. test_acc %.2f%% > best_test_acc %.2f%%' % (test_acc, best_test_acc))
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('Checkpoint'):
                os.mkdir('Checkpoint')
            torch.save(state, './Checkpoint/{}.pth'.format(model_name))
            best_test_acc = test_acc
        else:
            print('Not save.. test_acc %.2f%% < best_test_acc %.2f%%' % (test_acc, best_test_acc))
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ModelSet
from pylab import mpl
import datetime

mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 设置横纵坐标的名称以及对应字体格式
SaveModelPath = ''

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize([0.5], [0.5])
     ]
)

train_dataset = datasets.MNIST(root='../DataSet/MNIST', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='../DataSet/MNIST', train=False, transform=data_tf)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

epoch_MI_hM_X_upper = []
epoch_MI_hM_Y_upper = []

epoch_MI_hM_X_bin = []
epoch_MI_hM_Y_bin = []

epoch_MI_hM_X_mine = []
epoch_MI_hM_Y_mine = []

DO_MINE = False
DO_UPPER = True
DO_BIN = False
EPOCH_NUM = 20
Learning_Rate = 0.1
# 选择模型
# Activation_F = 'Tanh'
Activation_F = 'ReLU'
activation_f = torch.nn.ReLU() if Activation_F == 'ReLU' else torch.nn.Tanh()
# model = ModelSet.FC_with_Sigmoid(Activation_F)
model = ModelSet.FC_Sigmoid(activation_f)

print("Model Structure ", model)
acc_record, loss_record, best_acc = training(model=model,
                                             train_data_loader=train_loader,
                                             test_data_loader=test_loader,
                                             epochs=EPOCH_NUM,
                                             criterion=None,
                                             optimizer=optim.SGD(model.parameters(),
                                                                 lr=Learning_Rate),
                                             enable_cuda=True)
# show_model_performance(acc_record)
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=EPOCH_NUM))
# sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=EPOCH_NUM))

plt.figure()
plt.xlabel('I(T;X)')
plt.ylabel('I(T;Y)')
for i in range(EPOCH_NUM):
    if i % 1 == 0:
        c = sm.to_rgba(i)
        I_TX, I_TY = epoch_MI_hM_X_upper[i][::-1], epoch_MI_hM_Y_upper[i][::-1]
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
plt.title("%s(%s),LR(%.3f)" % (model.name, Activation_F, Learning_Rate))
plt.colorbar(sm, label='Epoch')

fig = plt.gcf()
plt.show()
# fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
fig.savefig('./%s_%s_%s_%s.pdf' % (
    model.name,
    Activation_F, str(EPOCH_NUM),
    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
print('result has been saved')
print('end')
