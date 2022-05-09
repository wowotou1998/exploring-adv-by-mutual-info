import matplotlib.pyplot as plt
import numpy
import numpy as np
from pylab import mpl
import datetime
from utils import *
import pickle
from matplotlib.lines import Line2D

mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
mpl.rcParams['figure.constrained_layout.use'] = True

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


def plot_mutual_info(Enable_Adv_Training):
    Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
    with open('./Checkpoint/basic_info_%s.pkl' % Is_Adv_Training, 'rb') as f:
        basic_info = pickle.load(f)
    with open('./Checkpoint/loss_and_acc_%s.pkl' % Is_Adv_Training, 'rb') as f:
        analytic_data = pickle.load(f)
    with open('./Checkpoint/loss_and_mutual_info_%s_std.pkl' % Is_Adv_Training, 'rb') as f:
        std = pickle.load(f)
    with open('./Checkpoint/loss_and_mutual_info_%s_adv.pkl' % Is_Adv_Training, 'rb') as f:
        adv = pickle.load(f)

    Forward_Size, Forward_Repeat = basic_info['Forward_Size'], basic_info['Forward_Repeat']
    Model_Name = basic_info['Model']
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

    def axs_plot(axs, std_I_TX, std_I_TY, adv_I_TX, adv_I_TY, epoch_i, MI_Type):
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
        axs[0].set_ylim((-0.5, 13))
        axs[1].set_ylim((-0.5, 13))

        axs[2].plot(Layer_Name, std_I_TY_epoch_i,
                    color=c, marker='o',
                    linestyle='-', linewidth=1,
                    )
        axs[3].plot(Layer_Name, adv_I_TY_epoch_i,
                    color=c, marker='^',
                    linestyle='--', linewidth=1,
                    )
        axs[2].set_ylim((-0.5, 4))
        axs[3].set_ylim((-0.5, 4))

        # axs[2].set_title('adv_' + MI_Type)
        # axs[2].plot(Layer_Name, adv_I_TX_epoch_i,
        #             color=c, marker='o',
        #             linestyle='-', linewidth=1,
        #             )
        # axs[3].plot(Layer_Name, adv_I_TY_epoch_i,
        #             color=c, marker='o',
        #             linestyle='-', linewidth=1,
        #             )

    # fig size, 先列后行
    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), )
    for i in range(nrows - 1):
        for j in range(ncols):
            # axs[0].set_xlim(0, 2)
            if j < 2:
                axs[i][j].set_xlabel('layers')
                axs[i][j].set_ylabel('I(T;X)')
            # axs[0].grid(True)
            else:
                axs[i][j].set_xlabel('layers')
                axs[i][j].set_ylabel('I(T;Y)')
            # axs[1].grid(True)

    # 开始，结束，步长
    for epoch_i in range(Std_Epoch_Num):
        if epoch_i % 1 == 0:
            # std/adv upper
            axs_plot(axs[0],
                     std.epoch_MI_hM_X_upper, std.epoch_MI_hM_Y_upper,
                     adv.epoch_MI_hM_X_upper, adv.epoch_MI_hM_Y_upper,
                     epoch_i, MI_Type='upper'
                     )
            axs_plot(axs[1],
                     std.epoch_MI_hM_X_lower, std.epoch_MI_hM_Y_lower,
                     adv.epoch_MI_hM_X_lower, adv.epoch_MI_hM_Y_lower,
                     epoch_i, MI_Type='lower'
                     )
            # std/adv bin
            axs_plot(axs[2],
                     std.epoch_MI_hM_X_bin, std.epoch_MI_hM_Y_bin,
                     adv.epoch_MI_hM_X_bin, adv.epoch_MI_hM_Y_bin,
                     epoch_i, MI_Type='bin'
                     )

            # plt.scatter(I_TX, I_TY,
            #             color=c,
            #             linestyle='-', linewidth=0.1,
            #             zorder=2
            #             )

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
    # fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
    # fig.savefig('./results_pdf/mutual_info_%s_%s.pdf' % (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    #                                                      Is_Adv_Training
    #                                                      )
    #             )
    fig.savefig('mutual_info_%s_%s.pdf' % (Model_Name, Is_Adv_Training))

    fig, axs = plt.subplots(nrows=2, ncols=Layer_Num, figsize=(17, 7))
    # clean examples info flow
    data = numpy.array(std.epoch_MI_hM_Y_lower_detail)
    for layer_i, ax in enumerate(axs[0]):
        for label_i in [i for i in range(10)]:
            # epoch_i, layer_i, label_i
            temp_data = data[..., layer_i, 2 * label_i - 1]
            ax.plot(Epochs, temp_data, label=r'$H(T_%d|y_%d)$' % (layer_i, label_i))
        # plot the H(T_i) lower
        ax.plot(Epochs, data[..., layer_i, -1], label=r'$H(T_%d) Lower$' % (layer_i))
        ax.set_xlabel('epochs')
        if layer_i == 0:
            ax.legend(ncol=2)
        ax.set_title('std')

    # adv example info flow
    data = numpy.array(adv.epoch_MI_hM_Y_lower_detail)
    for layer_i, ax in enumerate(axs[1]):
        for label_i in [i for i in range(10)]:
            # epoch_i, layer_i, label_i
            temp_data = data[..., layer_i, 2 * label_i - 1]
            ax.plot(Epochs, temp_data, label=r'$H(T_%d|y_%d)$' % (layer_i, label_i))
            #     plot the H(T_i) lower
        ax.plot(Epochs, data[..., layer_i, -1], label=r'$H(T_%d) Lower$' % (layer_i))
        ax.set_xlabel('epochs')
        if layer_i == 0:
            ax.legend(ncol=2)
        ax.set_title('adv')
    fig.suptitle('I(T;Y) detail')
    plt.show()
    fig.savefig('mutual_info_detail_%s_%s.pdf' % (Model_Name, Is_Adv_Training))
    print("Work has done!")


plot_mutual_info(Enable_Adv_Training=False)
# plot_mutual_info(Enable_Adv_Training=True)
