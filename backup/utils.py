from torch import optim, nn
from lab_8.backup.MI_wrapper import MI_estimator
import matplotlib.pyplot as plt


def on_train_begin():
    pass


def on_epoch_begin():
    pass


def on_batch_begin():
    pass


def on_epoch_end():
    pass


def evaluate_accuracy(test_data_loader, model, device):
    test_acc_sum, n = 0.0, 0
    model = model.to(device)
    model.eval()
    with torch.no_grad():
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
            """
            if n == 0:
                # named_children只输出了layer1和layer2两个子module，
                # 而named_modules输出了包括layer1和layer2下面所有的modolue。
                # 这两者均是迭代器
                estimator = MI_estimator(model)
                x = estimator(sample_data)
                print(estimator.layer_name)
                # for i in estimator.activation_i:
                #     print(i.shape)
                estimator.caculate_MI(sample_true_label)
                epoch_MI_hM_X_upper.append(estimator.MI_hM_X_upper)
                epoch_MI_hM_Y_upper.append(estimator.MI_hM_Y_upper)
                estimator.MI_hM_X_upper.clear()
                estimator.MI_hM_Y_upper.clear()
                estimator.activation_list.clear()
                estimator.layer_name.clear()
                del estimator

                # plt.figure()
                # # show two accuracy rate at the same figure
                # # 想要绘制线条的画需要记号中带有‘-’
                # plt.title("the trend of model")
                # plt.plot(estimator.layer_name, estimator.MI_hM_X_upper)
                # plt.plot(estimator.layer_name, estimator.MI_hM_Y_upper)
                # plt.show()

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
             model_name='MNIST'):
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
    print('--> %s is training...' % (model_name))
    if load_model_args:
        print('--> Loading model state dict..')
        try:
            print('--> Resuming from checkpoint..')
            # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('../Checkpoint/%s.pth' % (model_name))
            model.load_state_dict(checkpoint['model'])
            # data moved to GPU
            model = model.to(device)
            # 必须先将模型进行迁移,才能再装载optimizer,不然会出现数据在不同设备上的错误
            # optimizer.load_state_dict(checkpoint['optimizer'])
            best_test_acc = checkpoint['test_acc']
            start_epoch = checkpoint['epoch']
            print('--> Load checkpoint successfully! ')
        except Exception as e:
            print('--> %s\' checkpoint is not found ! ' % (model_name))

    model = model.to(device)
    model.train()
    # train_data_loader is a iterator object, which contains data and label
    # sample_data is a tensor,the size is batch_size * sample_size
    # sample_true_label is the same, which is 1 dim tensor, and the length is batch_size, and each sample
    # has a scalar type value
    """
    on_train_begin
    """
    on_train_begin()
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
            if not os.path.isdir('./Checkpoint'):
                os.mkdir('./Checkpoint')
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

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=80, shuffle=True)

# 选择模型
model = ModelSet.FC_ReLU()

# model = net.Activation_Net(28 * 28, 300, 100, 10)
# model = net.Batch_Net(28 * 28, 300, 100, 10)

if __name__ == '__main__':
    from pylab import mpl
    import datetime

    mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率

    print("Model Structure ", model)
    EPOCH_NUM = 20
    epoch_MI_hM_X_upper = []
    epoch_MI_hM_Y_upper = []
    acc_record, loss_record, best_acc = training(model=model,
                                                 train_data_loader=train_loader,
                                                 test_data_loader=test_loader,
                                                 epochs=EPOCH_NUM,
                                                 criterion=None,
                                                 optimizer=optim.SGD(model.parameters(), lr=0.001),
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
            plt.plot(I_TX, I_TY,
                     color=c, marker='o',
                     linestyle='-', linewidth=0.1
                     )

    fig = plt.gcf()
    plt.show()
    fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
    fig.savefig('./%s.pdf' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
    print('result has been saved')
    print('end')
