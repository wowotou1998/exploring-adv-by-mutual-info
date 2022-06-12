import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
import datetime
from utils import *
import pickle
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import math
import torch
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# Forward_Repeat, Forward_Size = 1, 2
# line_styles = ['-', ':']
# labels = ['std', 'adv']  # legend标签列表，上面的color即是颜色列表


# fig, ax = plt.subplots()
# lines = ax.plot(data)
# ax.legend(custom_lines, ['Cold', 'Medium', 'Hot'])
def plot_mutual_info_scatter(Model_Name, Enable_Adv_Training):
    # 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend patches = [mpatches.Patch(linestyle=line_styles[i],
    # label="{:s}".format(labels[i])) for i in range(len(line_styles))]

    # color = 'purple' or 'orange'
    line_legends = [
        Line2D([0], [0], color='C0', linewidth=1, linestyle='-', marker='o', markerfacecolor='none', markersize=10),
        Line2D([0], [0], color='Red', linewidth=1, linestyle='-', marker='+', markersize=10)]

    # linestyle='None' 设置为 None 就可以进行相应的 Marker
    marker_legends = [
        Line2D([0], [0], color='C0', linestyle='None', marker='o', markerfacecolor='none', markersize=10),
        Line2D([0], [0], color='Red', linestyle='None', marker='+', markersize=10)]

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

    # Green = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))

    std_color, adv_color = 'winter_r', 'autumn_r'
    # cmap_std = plt.get_cmap('coolwarm')
    # cmap_adv = plt.get_cmap('coolwarm')
    cmap_std = plt.get_cmap(std_color)
    cmap_adv = plt.get_cmap(adv_color)

    # cmap_std = plt.get_cmap('Blues')  # summer 偏绿色
    # cmap_adv = plt.get_cmap('Reds')  # summer 偏红色
    s_cmap_std = plt.cm.ScalarMappable(cmap=std_color, norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
    s_cmap_adv = plt.cm.ScalarMappable(cmap=adv_color, norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
    c_std = [cmap_std(i / Std_Epoch_Num * 1.0) for i in range(Std_Epoch_Num)]
    c_adv = [cmap_adv(i / Std_Epoch_Num * 1.0) for i in range(Std_Epoch_Num)]
    # Red = plt.cm.ScalarMappable(cmap='cmap_adv', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
    # sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))

    label_formatter_float = FormatStrFormatter('%.2f')  # 设置x轴标签文本的格式
    label_formatter_int = FormatStrFormatter('%d')  # 设置y轴标签文本的格式

    # subplot2grid, size = （行,列）, 块起始点坐标
    # grid_size = (4, Layer_Num)
    Fig_Size = (6, 10)
    fig = plt.figure(figsize=(Fig_Size[1], Fig_Size[0]), constrained_layout=True)
    spec = fig.add_gridspec(4, Layer_Num)

    # -------------------------------------------Loss and Accuracy Detail---------------------
    ax00 = fig.add_subplot(spec[0, 0])
    ax00.set_xlabel('Epochs')
    ax00.set_ylabel('Loss')
    ax00.plot(Epochs, analytic_data['train_loss'], label='Train set')
    ax00.plot(Epochs, analytic_data['test_clean_loss'], label='Clean test')
    ax00.plot(Epochs, analytic_data['test_adv_loss'], label='Adv test')
    # ax00.legend(prop={'size': 13})
    ax00.legend()
    # -------------------
    ax01 = fig.add_subplot(spec[0, 1])
    ax01.set_xlabel('Epochs')
    ax01.set_ylabel('Accuracy (%)')
    ax01.plot(Epochs, analytic_data['train_acc'], label='Train set')
    ax01.plot(Epochs, analytic_data['test_clean_acc'], label='Clean test')
    ax01.plot(Epochs, analytic_data['test_adv_acc'], label='Adv test')
    # ax01.legend(prop={'size': 13})
    ax01.legend()

    # -------------------------------------------overlook by Upper mutual info-------------------------
    ax02 = fig.add_subplot(spec[0, 2])
    ax02.set_xlabel('Layer index')
    ax02.set_ylabel(r'$I(T;X)$' + ' (bits)')
    ax02.set_title('The I(T;X) lower bound')
    # ax02.legend(line_legends, ['std', 'adv'], prop={'size': 13})
    ax02.legend(line_legends, ['std', 'adv'])

    ax03 = fig.add_subplot(spec[0, 3])
    ax03.set_xlabel('Layer index')
    ax03.set_ylabel(r'$I(T;Y)$' + ' (bits)')
    ax03.set_title('The I(T;Y) lower bound')

    for i in Epochs:
        # std.epoch_MI_hM_X_lower, std.epoch_MI_hM_Y_lower,
        # adv.epoch_MI_hM_X_lower, adv.epoch_MI_hM_Y_lower,

        ax02.plot(Layer_Name, std.epoch_MI_hM_X_lower[i], color=cmap_std(i / Std_Epoch_Num), marker='o')
        ax02.plot(Layer_Name, adv.epoch_MI_hM_X_lower[i], color=s_cmap_adv.to_rgba(i + 1), marker='+')

        ax03.plot(Layer_Name, std.epoch_MI_hM_Y_lower[i], color=s_cmap_std.to_rgba(i + 1), marker='o')
        ax03.plot(Layer_Name, adv.epoch_MI_hM_Y_lower[i], color=s_cmap_adv.to_rgba(i + 1), marker='+')

    # -------------------------------------------mutual information spilt by Layer---------------------
    def axs_plot(fig, std_I_TX, std_I_TY, adv_I_TX, adv_I_TY, Std_Epoch_Num, MI_Type, Row_i):
        std_I_TX = np.array(std_I_TX)
        std_I_TY = np.array(std_I_TY)
        adv_I_TX = np.array(adv_I_TX)
        adv_I_TY = np.array(adv_I_TY)
        # 坐标轴文本设置

        # 设定坐标范围
        i_tx_min = math.floor(min(np.min(std_I_TX), np.min(adv_I_TX))) - 0.5
        i_tx_max = math.ceil(max(np.max(std_I_TX), np.max(adv_I_TX))) + 0.5

        i_ty_min = math.floor(min(np.min(std_I_TY), np.min(adv_I_TY))) - 0.5
        i_ty_max = math.ceil(max(np.max(std_I_TY), np.max(adv_I_TY))) + 0.5

        for layer_i in range(Layer_Num):
            ax = fig.add_subplot(spec[Row_i, layer_i])

            # 最左侧的plot设置 y_label
            # 最下面的plot设置 x_label
            if layer_i == 0:
                ax.set_ylabel(MI_Type + '\n\n' + r'$I(T;Y)$' + ' (bits)')
            if Row_i == 3:
                ax.set_xlabel(r'$I(T;X)$' + ' (bits)')
            #  设置图例
            if layer_i == 0 and Row_i == 1:
                # ax.legend(marker_legends, ['std', 'adv'], prop={'size': 13})
                ax.legend(marker_legends, ['std', 'adv'])
            #  设置标题
            if Row_i == 1:
                ax.set_title('Layer %d' % (layer_i + 1))

            # c = np.array(Epochs)
            # c = Epochs
            # 将0-1之间得数值映射到颜色条上的点 colors.LinearSegmentColormap(Colormap)：是Colormap的子类。

            ax.scatter(std_I_TX[..., layer_i], std_I_TY[..., layer_i],
                       color='none',
                       marker='o',
                       s=60,
                       # facecolors='none',
                       edgecolors=c_std,
                       )
            # + 不适宜使用空心方案
            ax.scatter(adv_I_TX[..., layer_i], adv_I_TY[..., layer_i],
                       color=c_adv,
                       marker='+',
                       s=60,
                       )
            # 设定 x,y label 的显示样式
            # ax.xaxis.set_major_formatter(label_formatter_float)
            # ax.yaxis.set_major_formatter(label_formatter_float)

            # ax.set_ylim((i_tx_min, i_tx_max))
            # ax.set_ylim((i_tx_min, i_tx_max))
            #
            # ax.set_ylim((i_ty_min, i_ty_max))
            # ax.set_ylim((i_ty_min, i_ty_max))

            # 设置 color_bar
            if layer_i == (Layer_Num - 1) and Row_i == 1:
                fig.colorbar(s_cmap_std, ax=ax, ticks=[0, 100, 200], label='ST epoch')
            if layer_i == (Layer_Num - 1) and Row_i == 3:
                fig.colorbar(s_cmap_adv, ax=ax, ticks=[0, 100, 200], label='AT epoch')

    # std/adv Upper
    axs_plot(fig,
             std.epoch_MI_hM_X_upper, std.epoch_MI_hM_Y_upper,
             adv.epoch_MI_hM_X_upper, adv.epoch_MI_hM_Y_upper,
             Std_Epoch_Num, MI_Type='Upper', Row_i=1
             )
    # std/adv Lower
    axs_plot(fig,
             std.epoch_MI_hM_X_lower, std.epoch_MI_hM_Y_lower,
             adv.epoch_MI_hM_X_lower, adv.epoch_MI_hM_Y_lower,
             Std_Epoch_Num, MI_Type='Lower', Row_i=2
             )
    # std/adv Bin
    axs_plot(fig,
             std.epoch_MI_hM_X_bin, std.epoch_MI_hM_Y_bin,
             adv.epoch_MI_hM_X_bin, adv.epoch_MI_hM_Y_bin,
             Std_Epoch_Num, MI_Type='Bin', Row_i=3
             )
    title = "%s(%s),LR(%.3f),Upper/Lower/Bin,Clean(Adv),Sample_N(%d),%s" % (
        Model_Name, Activation_F, Learning_Rate, Forward_Repeat * Forward_Size, Is_Adv_Training
    )
    fig.suptitle(title)
    plt.show()

    # fig.savefig('mutual_info_%s_%s_%s.pdf' % (
    #     Model_Name, Is_Adv_Training,
    #     datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

    # -------------------------------------------Mutual Information Detail---------------------
    # 设定坐标范围
    # i_tx_min = math.floor(min(np.min(std_I_TX), np.min(adv_I_TX))) - 0.5
    # i_tx_max = math.ceil(max(np.max(std_I_TX), np.max(adv_I_TX)))
    #
    # i_ty_min = math.floor(min(np.min(std_I_TY), np.min(adv_I_TY))) - 0.5
    # i_ty_max = math.ceil(max(np.max(std_I_TY), np.max(adv_I_TY)))
    Fig_Size_2 = (4, 10)
    fig2, axs = plt.subplots(nrows=2, ncols=Layer_Num, figsize=(Fig_Size_2[1], Fig_Size_2[0]), constrained_layout=True)
    std_lower_detail = np.array(std.epoch_MI_hM_Y_lower_detail)
    adv_lower_detail = np.array(adv.epoch_MI_hM_Y_lower_detail)
    # C0-C9 是 matplotlib 里经常使用的色条
    COLOR = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',
             'C6', 'C7', 'C8', 'C9', 'olive', 'peach', ]

    for layer_i in range(Layer_Num):
        # 设定 y 标签的格式
        axs[0][layer_i].yaxis.set_major_formatter(label_formatter_int)
        axs[1][layer_i].yaxis.set_major_formatter(label_formatter_int)

        axs[0][layer_i].set_title('Layer %d' % layer_i)
        # epoch_i, layer_i, label_i
        axs[0][layer_i].plot(Epochs, std_lower_detail[..., layer_i, -1],
                             color=COLOR[0],
                             label=r'$H_{Lower}(T_i)$')

        axs[1][layer_i].set_xlabel('Epochs')
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
    # 只有第一个子图显示 legend 信息
    # axs[0][0].legend(ncol=2, prop={'size': 13})
    # axs[1][Layer_Num - 1].legend(ncol=2, loc='upper left', bbox_to_anchor=(1.1, 1), )
    # axs[1][Layer_Num - 1].legend(ncol=2, loc='upper left', bbox_to_anchor=(1, 1), )legen
    # axs[0][0].get_legend_handles_labels()

    # leg2 = ax2.legend(*ax.get_legend_handles_labels(), loc=(0.8, 0.85))
    # 只有最左侧的子图显示 y label 信息
    axs[0][0].set_ylabel('Std')
    axs[1][0].set_ylabel('Adv')

    title = "%s(%s),LR(%.3f),MI Lower Bound detail,Clean(Adv),Sample_N(%d),%s" % (
        Model_Name, Activation_F, Learning_Rate, Forward_Repeat * Forward_Size, Is_Adv_Training
    )
    fig2.suptitle(title)
    plt.show()
    # fig.savefig('mutual_info_detail_%s_%s.pdf' % (Model_Name, Is_Adv_Training))
    print("Work has done!")


def plot_mutual_info(Model_Name, Enable_Adv_Training):
    # 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
    # patches = [mpatches.Patch(linestyle=line_styles[i], label="{:s}".format(labels[i])) for i in range(len(line_styles))]

    # color = 'purple' or 'orange'
    line_legends = [Line2D([0], [0], color='purple', linewidth=1, linestyle='-', marker='o'),
                    Line2D([0], [0], color='purple', linewidth=1, linestyle='--', marker='^')]
    Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
    with open('./Checkpoint/%s/basic_info_%s.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        basic_info = pickle.load(f)
    with open('./Checkpoint/%s/loss_and_acc_%s.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        analytic_data = pickle.load(f)
    with open('./Checkpoint/%s/loss_and_mutual_info_%s_std.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        std = pickle.load(f)
    with open('./Checkpoint/%s/loss_and_mutual_info_%s_adv.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        adv = pickle.load(f)
    '''
    过渡方案， 读取之后消除layer_activations再保存回去
    '''
    with open('./Checkpoint/%s/loss_and_mutual_info_%s_std.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
        std.clear_activations()
        pickle.dump(std, f)
    with open('./Checkpoint/%s/loss_and_mutual_info_%s_adv.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
        adv.clear_activations()
        pickle.dump(adv, f)

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


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    import matplotlib
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_transfer_matrix(Model_Name, Enable_Adv_Training):
    Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
    with open('./Checkpoint/%s/transfer_matrix_%s.pkl' % (Model_Name, Is_Adv_Training), 'rb') as f:
        transfer_matrix = pickle.load(f)

    def plot_heat_map(matrix_1, row_labels, col_labels, ax, valfmt):
        import matplotlib

        fmt = matplotlib.ticker.StrMethodFormatter(valfmt)
        im = ax.imshow(matrix_1)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
        ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        # 在每一个格子里显示数值
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, fmt(matrix_1[i, j]),
                               ha="center", va="center", color="w")

        ax.set_title("matrix_1 of local col_labels (in tons/year)")
        ax.figure.colorbar(im, ax=ax)

    # print(transfer_matrix)
    label_num = 10
    label_chunk = transfer_matrix['label_chunk']
    label_std_chunk = transfer_matrix['label_std_chunk']
    label_prob_std_chunk = transfer_matrix['label_prob_std_chunk']
    label_adv_chunk = transfer_matrix['label_adv_chunk']
    label_prob_adv_chunk = transfer_matrix['label_prob_adv_chunk']

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
    i2j_adv, i2j_prob_adv = calculate_transfer_matrix(label_chunk, label_adv_chunk, label_prob_adv_chunk, 10)
    label_name = [str(i) for i in range(10)]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    plot_heat_map(i2j_std, row_labels=label_name, col_labels=label_name, ax=axs[0][0], valfmt="{x:}")
    plot_heat_map(i2j_prob_std, row_labels=label_name, col_labels=label_name, ax=axs[0][1], valfmt="{x:.2f}")

    plot_heat_map(i2j_adv, row_labels=label_name, col_labels=label_name, ax=axs[1][0], valfmt="{x:}")
    plot_heat_map(i2j_prob_adv, row_labels=label_name, col_labels=label_name, ax=axs[1][1], valfmt="{x:.2f}")

    fig.savefig('transfer_matrix_%s_%s.pdf' % (Model_Name, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))


if __name__ == '__main__':
    import matplotlib

    # matplotlib.use('agg')
    # matplotlib.get_backend()

    # mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['backend'] = 'agg'
    # mpl.rcParams["font.size"] = 18
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
    mpl.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    import argparse

    parser = argparse.ArgumentParser(description='plot arguments')
    # parser.add_argument('--Model_Name', default='LeNet_MNIST', type=str, help='The Model_Name.')
    # parser.add_argument('--Model_Name', default='LeNet_CIFAR10', type=str, help='The Model_Name.')
    # parser.add_argument('--Model_Name', default='FC_2', type=str, help='The Model_Name.')
    # parser.add_argument('--Model_Name', default='WideResNet_STL10', type=str, help='The Model_Name.')
    parser.add_argument('--Model_Name', default='WideResNet_CIFAR10', type=str, help='The Model_Name.')
    args = parser.parse_args()
    Model_Name = args.Model_Name
    # plot_transfer_matrix(Model_Name, Enable_Adv_Training=False)
    plot_mutual_info_scatter(Model_Name, Enable_Adv_Training=False)
    plot_mutual_info_scatter(Model_Name, Enable_Adv_Training=True)
    # plot_mutual_info(Model_Name, Enable_Adv_Training=True)

    # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
    #               "potato", "wheat", "barley"]
    # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    #
    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    # fig, ax = plt.subplots()
    #
    # im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
    #                    cmap="YlGn", cbarlabel="harvest [t/year]")
    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    #
    # # fig.tight_layout()
    # plt.show()
