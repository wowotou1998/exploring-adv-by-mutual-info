import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
import datetime
from utils import *
import pickle
from matplotlib.lines import Line2D
import math
import torch
import torch.nn.functional as F

# mpl.rcParams['font.sans-serif'] = ['Times New Roman']
# mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
mpl.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

Forward_Repeat, Forward_Size = 1, 2
line_styles = ['-', ':']
labels = ['std', 'adv']  # legend标签列表，上面的color即是颜色列表

# 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
# patches = [mpatches.Patch(linestyle=line_styles[i], label="{:s}".format(labels[i])) for i in range(len(line_styles))]

# color = 'purple' or 'orange'
line_legends = [Line2D([0], [0], color='purple', linewidth=1, linestyle='-', marker='o'),
                Line2D([0], [0], color='purple', linewidth=1, linestyle='--', marker='^')]


# fig, ax = plt.subplots()
# lines = ax.plot(data)
# ax.legend(custom_lines, ['Cold', 'Medium', 'Hot'])


def plot_mutual_info(Model_Name, Enable_Adv_Training):
    Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
    with open('./Checkpoint/%s/basic_info_%s.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        basic_info = pickle.load(f)
    with open('./Checkpoint/%s/loss_and_acc_%s.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        analytic_data = pickle.load(f)
    with open('./Checkpoint/%s/loss_and_mutual_info_%s_std.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        std = pickle.load(f)
    with open('./Checkpoint/%s/loss_and_mutual_info_%s_adv.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        adv = pickle.load(f)

    Forward_Size, Forward_Repeat = basic_info['Forward_Size'], basic_info['Forward_Repeat']
    # Model_Name = basic_info['Model']
    Activation_F = 'relu'
    Learning_Rate = 0.08

    Std_Epoch_Num = len(std.epoch_MI_hM_X_upper)
    Epochs = [i for i in range(Std_Epoch_Num)]
    Layer_Num = len(std.epoch_MI_hM_X_upper[0])
    Layer_Name = [str(i) for i in range(Layer_Num)]

    # sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))

    title = "%s(%s),LR(%.3f),Upper/Lower/Bin,Clean(Adv),Sample_N(%d),%s" % (
        Model_Name, Activation_F, Learning_Rate, Forward_Repeat * Forward_Size, Is_Adv_Training
    )

    def axs_plot(axs, std_I_TX, std_I_TY, adv_I_TX, adv_I_TY, Std_Epoch_Num, MI_Type):
        std_I_TX = np.array(std_I_TX)
        std_I_TY = np.array(std_I_TY)
        adv_I_TX = np.array(adv_I_TX)
        adv_I_TY = np.array(adv_I_TY)

        # 设定坐标范围
        i_tx_min = math.floor(min(np.min(std_I_TX), np.min(adv_I_TX))) - 0.5
        i_tx_max = math.ceil(max(np.max(std_I_TX), np.max(adv_I_TX))) + 0.5

        i_ty_min = math.floor(min(np.min(std_I_TY), np.min(adv_I_TY))) - 0.5
        i_ty_max = math.ceil(max(np.max(std_I_TY), np.max(adv_I_TY))) + 0.5

        for epoch_i in range(Std_Epoch_Num):
            c = sm.to_rgba(epoch_i + 1)
            # layers = [i for i in range(1,len(I_TX)+1)]
            std_I_TX_epoch_i, std_I_TY_epoch_i = std_I_TX[epoch_i], std_I_TY[epoch_i]
            adv_I_TX_epoch_i, adv_I_TY_epoch_i = adv_I_TX[epoch_i], adv_I_TY[epoch_i]

            axs[0].set_title(MI_Type)

            axs[0].legend(line_legends, ['std', 'adv'])
            axs[1].legend(line_legends, ['std', 'adv'])

            axs[0].plot(Layer_Name, std_I_TX_epoch_i,
                        color=c, marker='o',
                        linestyle='-', linewidth=1,
                        )
            axs[1].plot(Layer_Name, adv_I_TX_epoch_i,
                        color=c, marker='^',
                        linestyle='--', linewidth=1,
                        )

            axs[0].set_ylim((i_tx_min, i_tx_max))
            axs[1].set_ylim((i_tx_min, i_tx_max))

            axs[2].plot(Layer_Name, std_I_TY_epoch_i,
                        color=c, marker='o',
                        linestyle='-', linewidth=1,
                        )
            axs[3].plot(Layer_Name, adv_I_TY_epoch_i,
                        color=c, marker='^',
                        linestyle='--', linewidth=1,
                        )

            axs[2].set_ylim((i_ty_min, i_ty_max))
            axs[3].set_ylim((i_ty_min, i_ty_max))

    # fig size, 先列后行
    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), )

    # 初始化 xlabel, y_label
    for i in range(nrows - 1):
        for j in range(ncols):
            axs[i][j].grid(True)
            if j < 2:
                axs[i][j].set_xlabel('layers')
                axs[i][j].set_ylabel(r'$I(T;X)$')

            else:
                axs[i][j].set_xlabel('layers')
                axs[i][j].set_ylabel(r'$I(T;Y)$')

    # range(开始，结束，步长)
    # 绘制每一轮次的信息曲线

    # std/adv Upper
    axs_plot(axs[0],
             std.epoch_MI_hM_X_upper, std.epoch_MI_hM_Y_upper,
             adv.epoch_MI_hM_X_upper, adv.epoch_MI_hM_Y_upper,
             Std_Epoch_Num, MI_Type='upper'
             )
    # std/adv Lower
    axs_plot(axs[1],
             std.epoch_MI_hM_X_lower, std.epoch_MI_hM_Y_lower,
             adv.epoch_MI_hM_X_lower, adv.epoch_MI_hM_Y_lower,
             Std_Epoch_Num, MI_Type='lower'
             )
    # std/adv Bin
    axs_plot(axs[2],
             std.epoch_MI_hM_X_bin, std.epoch_MI_hM_Y_bin,
             adv.epoch_MI_hM_X_bin, adv.epoch_MI_hM_Y_bin,
             Std_Epoch_Num, MI_Type='bin'
             )

    # plt.scatter(I_TX, I_TY,
    #             color=c,
    #             linestyle='-', linewidth=0.1,
    #             zorder=2
    #             )
    # -------------------------------------------Loss and Accuracy Detail---------------------
    # for idx, (k, v) in enumerate(analytic_data.items()):
    axs[nrows - 1][0].set_xlabel('epochs')
    axs[nrows - 1][0].set_title('loss')
    axs[nrows - 1][0].plot(Epochs, analytic_data['train_loss'], label='train_loss')
    axs[nrows - 1][0].plot(Epochs, analytic_data['test_clean_loss'], label='test_clean_loss')
    axs[nrows - 1][0].plot(Epochs, analytic_data['test_adv_loss'], label='test_adv_loss')
    axs[nrows - 1][0].legend()
    # -------------------
    axs[nrows - 1][1].set_xlabel('epochs')
    axs[nrows - 1][1].set_title('acc')
    axs[nrows - 1][1].plot(Epochs, analytic_data['train_acc'], label='train_acc')
    axs[nrows - 1][1].plot(Epochs, analytic_data['test_clean_acc'], label='test_clean_acc')
    axs[nrows - 1][1].plot(Epochs, analytic_data['test_adv_acc'], label='test_adv_acc')
    axs[nrows - 1][1].legend()

    # plt.scatter(epoch_MI_hM_X_upper[0], epoch_MI_hM_Y_upper[0])
    # plt.legend()

    fig.suptitle(title)
    fig.colorbar(sm, ax=axs, label='Epoch')

    # fig = plt.gcf()
    # if Enable_Show:
    plt.show()
    fig.savefig('mutual_info_%s_%s_%s.pdf' % (
        Model_Name, Is_Adv_Training,
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

    # -------------------------------------------Mutual Information Detail---------------------
    # 设定坐标范围
    # i_tx_min = math.floor(min(np.min(std_I_TX), np.min(adv_I_TX))) - 0.5
    # i_tx_max = math.ceil(max(np.max(std_I_TX), np.max(adv_I_TX)))
    #
    # i_ty_min = math.floor(min(np.min(std_I_TY), np.min(adv_I_TY))) - 0.5
    # i_ty_max = math.ceil(max(np.max(std_I_TY), np.max(adv_I_TY)))
    fig, axs = plt.subplots(nrows=2, ncols=Layer_Num, figsize=(17, 7))
    std_lower_detail = np.array(std.epoch_MI_hM_Y_lower_detail)
    adv_lower_detail = np.array(adv.epoch_MI_hM_Y_lower_detail)
    # C0-C9 是 matplotlib 里经常使用的色条
    COLOR = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',
             'C6', 'C7', 'C8', 'C9', 'olive', 'peach', ]

    for layer_i in range(Layer_Num):
        axs[0][layer_i].set_xlabel('epochs')
        axs[0][layer_i].set_title('Std Layer %d' % layer_i)
        # epoch_i, layer_i, label_i
        axs[0][layer_i].plot(Epochs, std_lower_detail[..., layer_i, -1],
                             color=COLOR[0],
                             label=r'$H_{Lower}(T_i)$')

        axs[1][layer_i].set_xlabel('epochs')
        axs[1][layer_i].set_title('Adv Layer %d' % layer_i)
        axs[1][layer_i].plot(Epochs, adv_lower_detail[..., layer_i, -1],
                             color=COLOR[0],
                             label=r'$H_{Lower}(T_i)$')

        for label_i in [i for i in range(10)]:
            # epoch_i, layer_i, label_i
            std_temp_data = std_lower_detail[..., layer_i, label_i]
            axs[0][layer_i].plot(Epochs, std_temp_data,
                                 color=COLOR[label_i + 1],
                                 label=r'$H(T_i|y_%d)$' % (label_i))
            adv_temp_data = std_lower_detail[..., layer_i, label_i]
            axs[1][layer_i].plot(Epochs, adv_temp_data,
                                 color=COLOR[label_i + 1],
                                 label=r'$H(T_i|y_%d)$' % (label_i))
        if layer_i == 0:
            axs[0][0].legend(ncol=2)

    title = "%s(%s),LR(%.3f),MI Lower Bound detail,Clean(Adv),Sample_N(%d),%s" % (
        Model_Name, Activation_F, Learning_Rate, Forward_Repeat * Forward_Size, Is_Adv_Training
    )
    fig.suptitle(title)
    plt.show()
    fig.savefig('mutual_info_detail_%s_%s.pdf' % (Model_Name, Is_Adv_Training))
    print("Work has done!")


def plot_transfer_matrix(Model_Name, Enable_Adv_Training):
    Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
    with open('./Checkpoint/%s/transfer_matrix_%s.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        transfer_matrix = pickle.load(f)

    # print(transfer_matrix)
    label_num = 10
    label_chunk = transfer_matrix['label_chunk']
    label_std_chunk = transfer_matrix['label_std_chunk']
    label_prob_std_chunk = transfer_matrix['label_prob_std_chunk']
    label_adv_chunk = transfer_matrix['label_adv_chunk']
    label_prob_adv_chunk = transfer_matrix['label_prob_adv_chunk']
    """
    获取标签Y中label_i的经验概率
    """
    # Y_one_hot = F.one_hot(Y, num_classes=label_num).float().to(Y.device)
    # Y_probs = torch.mean(Y_one_hot, dim=0)

    """
    获取标签Y中等于label_i的下标集合, pytorch中的tensor可以使用布尔索引,布尔索引中的元素要为布尔值
    """

    # Y_i_idx = []
    # for i in range(label_num):
    #     Y_equal_label_i_index = torch.flatten(Y == i)
    #     Y_i_idx.append(Y_equal_label_i_index)
    #
    # saved_label_idx = {}
    # for idx, value in enumerate(Y_i_idx):
    #     saved_label_idx[idx] = value.clone().detach().cpu().numpy()

    def calculate_transfer_matrix(labels_origin, predict, probs, label_num):
        label_i2j = np.zeros(shape=(label_num, label_num), dtype=int)
        label_i2j_prob = np.zeros(shape=(label_num, label_num), dtype=float)
        for i in range(label_num):
            # 获取 源标签数组 中关于第i号标签的索引
            index_about_i = torch.flatten(labels_origin == i)
            # 根据索引在 预测标签数组/预测概率数组 获取相应的内容
            predict_about_i = predict[index_about_i]
            probs_about_i = probs[index_about_i]
            for j in range(label_num):
                # 获取 对标签i预测之后的数组中 对 第j号标签 的索引 (神经网络如果没做到100%正确， 那会把真实标签i预测为其他的标签j
                index_about_i2j = torch.flatten(predict_about_i == j)
                i2j_num = torch.sum(index_about_i2j).item()
                label_i2j[i][j] = i2j_num
                # 根据索引在 预测概率数组 获取相应的内容
                # 由于bool 索引中的数值相当于只要0、1值， 可以使用求和函数直接算出把第i类分别为j类的个数
                probs_about_i2j = probs_about_i[index_about_i2j]
                if i2j_num == 0:
                    label_i2j_prob[i][j] = 0.0
                else:
                    label_i2j_prob[i][j] = torch.sum(probs_about_i2j).item() / i2j_num
        return label_i2j, label_i2j_prob

    i2j_std, i2j_prob_std = calculate_transfer_matrix(label_chunk, label_std_chunk, label_prob_std_chunk, 10)
    i2j_adv, i2j_prob_adv = calculate_transfer_matrix(label_chunk, label_std_chunk, label_prob_std_chunk, 10)


#     plot


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='plot arguments')
    parser.add_argument('--Model_Name', default='LeNet_cifar10', type=str, help='The Model_Name.')

    args = parser.parse_args()
    Model_Name = args.Model_Name
    plot_transfer_matrix(Model_Name, Enable_Adv_Training=False)
    # plot_mutual_info(Model_Name, Enable_Adv_Training=False)
    # plot_mutual_info(Model_Name, Enable_Adv_Training=True)
