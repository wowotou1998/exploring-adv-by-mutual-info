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
from torchattacks import PGD
import pickle
import torch.nn.functional as F


# mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
# mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.1, 0.95
# mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.1, 0.9
# mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.2, 0.4
# mpl.rcParams['figure.constrained_layout.use'] = True
# 选择模型
# Activation_F = 'Tanh'
# Activation_F = 'ReLU'


class Forward():
    def __init__(self, Origin_Model, args):
        self.Args = args
        self.Origin_Model = Origin_Model
        self.Model_Name = args.Model_Name
        self.Data_Set = args.Data_Set
        # self.Enable_Show = True
        # self.Std_Epoch_Num = args.Std_Epoch_Num
        self.Forward_Size, self.Forward_Repeat = args.Forward_Size, args.Forward_Repeat
        # self.Train_Batch_Size = args.batch_size
        self.Device = torch.device("cuda:%d" % (args.GPU) if torch.cuda.is_available() else "cpu")
        # 在 forward之前设定一下测试集的装载
        self.Test_Loader = None  # self.get_test_loader(Data_Set)
        self.std_estimator = mutual_info_estimator(self.Origin_Model.modules_to_hook, By_Layer_Name=False)
        self.adv_estimator = mutual_info_estimator(self.Origin_Model.modules_to_hook, By_Layer_Name=False)
        self.Patch_Split_L = [0, 2, 4, 8]  # 0
        self.Saturation_L = [2, 8, 16, 64, 1024]  # 2
        self.Loss_Acc = None

    def get_test_loader(self):
        # 全局取消证书验证
        import ssl
        import random
        ssl._create_default_https_context = ssl._create_unverified_context

        # class Saturation_Transform(object):
        #     '''
        #     for each pixel v: v' = sign(2v - 1) * |2v - 1|^{2/p}  * 0.5 + 0.5
        #     then clip -> (0, 1)
        #     '''
        # 
        #     def __init__(self, saturation_level=2.0):
        #         self.p = saturation_level
        # 
        #     def __call__(self, img):
        #         ones = torch.ones_like(img)
        #         # print(img.size(), torch.max(img), torch.min(img))
        #         ret_img = torch.sign(2 * img - ones) * torch.pow(torch.abs(2 * img - ones), 2.0 / self.p)
        # 
        #         ret_img = ret_img * 0.5 + ones * 0.5
        # 
        #         ret_img = torch.clamp(ret_img, 0, 1)
        # 
        #         return ret_img
        # 
        # class Patch_Transform(object):
        #     def __init__(self, k=2):
        #         self.k = k
        # 
        #     def __call__(self, xtensor: torch.Tensor):
        #         '''
        #         X: torch.Tensor of shape(c, h, w)   h % self.k == 0
        #         :param xtensor:
        #         :return:
        #         '''
        #         patches = []
        #         # K==0则不分割数据
        #         if self.k == 0:
        #             return xtensor
        # 
        #         c, h, w = xtensor.size()
        #         dh = h // self.k
        #         dw = w // self.k
        # 
        #         # print(dh, dw)
        #         sh = 0
        #         for i in range(h // dh):
        #             eh = sh + dh
        #             eh = min(eh, h)
        #             sw = 0
        #             for j in range(w // dw):
        #                 ew = sw + dw
        #                 ew = min(ew, w)
        #                 patches.append(xtensor[:, sh:eh, sw:ew])
        # 
        #                 # print(sh, eh, sw, ew)
        #                 sw = ew
        #             sh = eh
        # 
        #         random.shuffle(patches)
        # 
        #         start = 0
        #         imgs = []
        #         for i in range(self.k):
        #             end = start + self.k
        #             imgs.append(torch.cat(patches[start:end], dim=1))
        #             start = end
        #         img = torch.cat(imgs, dim=2)
        #         return img
        # 
        # if Transform_Type == 'Saturation':
        #     Extra_Transform = Saturation_Transform(saturation_level=Level)
        # elif Transform_Type == 'Patch':
        #     Extra_Transform = Patch_Transform(k=Level)
        # else:
        #     raise RuntimeError('Unknown Transformation')

        data_tf_test = transforms.Compose([
            transforms.ToTensor(),
            # Saturation_Transform(saturation_level=1024.),
            # Patch_Transform(k=4),
            # Extra_Transform
        ])

        data_tf_mnist = transforms.Compose([
            transforms.ToTensor(),
        ])

        Data_Set = self.Data_Set

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
        Test_Loader = DataLoader(dataset=test_dataset, batch_size=self.Forward_Size, shuffle=True)
        return Test_Loader

    def train_attack(self, Model, Random_Start=False):
        # atk = PGD(Model, eps=args.Eps, alpha=args.Eps * 1.2 / 7, steps=7, random_start=Random_Start)
        atk = PGD(Model, eps=self.Args.Eps, alpha=self.Args.Alpha, steps=self.Args.Step, random_start=Random_Start)
        # atk = PGD(Model, eps=30 / 255, alpha=5 / 255, steps=7, random_start=Random_Start)
        return atk

    def test_attack(self, Model, Random_Start=False):
        # atk = PGD(Model, eps=args.Eps, alpha=args.Eps * 1.2 / 7, steps=7, random_start=Random_Start)
        atk = PGD(Model, eps=self.Args.Eps, alpha=self.Args.Alpha, steps=self.Args.Step, random_start=Random_Start)
        # atk = PGD(Model, eps=12 / 255, alpha=3 / 255, steps=7, random_start=Random_Start)
        # atk = PGD(Model, eps=16 / 255, alpha=4 / 255, steps=7, random_start=Random_Start)
        # atk = PGD(Model, eps=30 / 255, alpha=5 / 255, steps=7, random_start=Random_Start)
        return atk

    def save_mutual_info_data(self, Transform_Type, Enable_Adv_Training):
        Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
        Model_Name, Forward_Size, Forward_Repeat = self.Model_Name, self.Forward_Size, self.Forward_Repeat
        dir = 'Checkpoint/%s/%s' % (Model_Name, Transform_Type)
        # 对于每一个模型产生的数据, 使用一个文件夹单独存放
        if not os.path.exists(dir):
            os.makedirs(dir)

        mi_loss_acc = {'Model': Model_Name,
                       'Enable_Adv_Training': Enable_Adv_Training,
                       'Forward_Size': Forward_Size,
                       'Forward_Repeat': Forward_Repeat,
                       'std_estimator': self.std_estimator,
                       'adv_estimator': self.adv_estimator,
                       'loss_acc': self.Loss_Acc
                       }

        with open('./Checkpoint/%s/%s/mi_loss_acc_%s.pkl' % (Model_Name, Transform_Type, Is_Adv_Training), 'wb') as f:
            pickle.dump(mi_loss_acc, f)

    @torch.no_grad()
    def get_clean_or_adv_image(self, Model, Keep_Clean):
        atk = self.test_attack(Model, Random_Start=False)

        batch_images, batch_labels = next(iter(self.Test_Loader))
        batch_images = batch_images.to(self.Device)
        batch_labels = batch_labels.to(self.Device)
        if Keep_Clean:
            return batch_images, batch_labels

        else:
            with torch.enable_grad():
                adv_images = atk(batch_images, batch_labels)
                return adv_images, batch_labels

    @torch.no_grad()
    def calculate_acc_and_mutual_info(self, Model, Transform_Type, Level, Keep_Clean):
        import random
        # 这里的epoch_i没必要指定，因为epochi就是列表当中的最后一个元素
        # a = list[-1]就是最后一个元素
        Model.eval()

        correct_N = 0
        total_N = 0
        loss = 0.

        image_chunk = None
        label_chunk = None
        layer_activation_chunk = None

        def Saturation_Transform(batch_images, level=2):
            '''
            for each pixel v: v' = sign(2v - 1) * |2v - 1|^{2/p}  * 0.5 + 0.5
            then clip -> (0, 1)
            '''
            p = level * 1.0
            ones = torch.ones_like(batch_images)
            # print(img.size(), torch.max(img), torch.min(img))
            ret_img = torch.sign(2 * batch_images - ones) * torch.pow(torch.abs(2 * batch_images - ones), 2.0 / p)

            ret_img = ret_img * 0.5 + ones * 0.5

            ret_img = torch.clamp(ret_img, 0, 1)

            return ret_img

        def Patch_Transform(batch_images, level=2):
            '''
            X: torch.Tensor of shape(c, h, w)   h % self.k == 0
            :param images:
            :return:
            '''
            # K==0则不分割数据
            k = level
            if k == 0:
                return batch_images

            b, c, h, w = batch_images.size()

            for idx in range(b):
                patches = []
                images = batch_images[idx]

                dh = h // k
                dw = w // k

                # print(dh, dw)
                sh = 0
                for i in range(h // dh):
                    eh = sh + dh
                    eh = min(eh, h)
                    sw = 0
                    for j in range(w // dw):
                        ew = sw + dw
                        ew = min(ew, w)
                        patches.append(images[:, sh:eh, sw:ew])

                        # print(sh, eh, sw, ew)
                        sw = ew
                    sh = eh

                random.shuffle(patches)

                start = 0
                imgs = []
                for i in range(k):
                    end = start + k
                    imgs.append(torch.cat(patches[start:end], dim=1))
                    start = end
                img = torch.cat(imgs, dim=2)
                batch_images[idx] = img
            return batch_images

        if Transform_Type == 'Saturation':
            Extra_Transform = Saturation_Transform
        elif Transform_Type == 'Patch':
            Extra_Transform = Patch_Transform
        else:
            raise RuntimeError('Unknown Transformation')

        if Keep_Clean:
            estimator = self.std_estimator
        else:
            estimator = self.adv_estimator

        for i in range(self.Forward_Repeat):

            batch_images, labels = self.get_clean_or_adv_image(Model, Keep_Clean)
            '''
            对正常样本和对抗样本进行变换
            '''
            images = Extra_Transform(batch_images, Level)

            # labels = labels.to(Device)
            # # print('std_test_size', images.size(0))
            # images = images.to(Device)

            """
            forward之前先clear
            """
            estimator.clear_activations()
            # register hook
            estimator.do_forward_hook(Model)
            """
            计算模型的准确率
            """
            outputs = Model(images)
            loss_i = F.cross_entropy(outputs, labels)
            # predicted_prob, predicted, labels 都可以看成是一个列表或者是一个向量，列表中元素的个数为 batch_size 个
            # 先对神经网络的输出结果做一个 softmax 获取概率值
            # predicted_prob, predicted = torch.max(F.softmax(outputs, dim=1), dim=1)
            predicted_prob, predicted = torch.max(outputs, dim=1)
            correct_N += (predicted == labels).sum().item()
            total_N += labels.size(0)
            loss += loss_i.item()

            """
            发现并修改了一个重大bug, 这里每forward一次,caculate_MI 函数计算出的互信息值都直接挂在列表的后面，那么 Forward_Repeat 会成倍放大列表的长度
            且会混乱每一个 epoch 中的互信息变化情况，Forward_Repeat 一旦超过 epoch_num ，那么每一个 epoch 的曲线就会
            """
            # 给定初始值
            if i == 0:
                # print("---> layer activations size {} <---".format(layer_activations_size))
                image_chunk = images.clone().detach()
                label_chunk = labels.clone().detach()
                '''
                注意， 这里如果简单赋值就会出现传递引用的现象，需要手动调用copy,复制列表
                '''
                layer_activation_chunk = estimator.layer_activations.copy()
            # 计算所有循环的和
            else:
                image_chunk = torch.cat((image_chunk, images.clone().detach()), dim=0)
                label_chunk = torch.cat((label_chunk, labels.clone().detach()), dim=0)
                """
                这里 layer_activations 是一个 list, list 里的每一个元素时 tesnor (gpu:0)
                """
                for idx, item in enumerate(estimator.layer_activations):
                    layer_activation_chunk[idx] = torch.cat((layer_activation_chunk[idx], item.clone().detach()), dim=0)
            """
            forward 之后例行收尾工作
            """
            estimator.cancel_hook()
            estimator.clear_activations()
        # 计算存储互信息
        # calculate mutual info
        estimator.layer_activations = layer_activation_chunk
        estimator.caculate_MI(image_chunk.cpu(), label_chunk.cpu())
        estimator.store_MI()

        acc = correct_N * 100. / total_N
        return acc, loss / self.Forward_Repeat

    def forward(self, Model, Transform_Type, Enable_Adv_Training):
        test_clean_acc_L, test_adv_acc_L = [], []
        test_clean_loss_L, test_adv_loss_L = [], []

        # Load checkpoint.
        if Enable_Adv_Training:
            # 装载训练好的模型
            print('--> Loading AT-Model %s state dict..' % self.Model_Name)
            load_model(Model, './Checkpoint/%s/%s_adv.pth' % (self.Model_Name, self.Model_Name))
        else:
            print('--> Loading STD-Model %s state dict..' % self.Model_Name)
            load_model(Model, './Checkpoint/%s/%s_std.pth' % (self.Model_Name, self.Model_Name))
        print('--> Load checkpoint successfully! ')

        Model = Model.to(self.Device)
        Model.eval()

        if Transform_Type == 'Saturation':
            Level_L = self.Saturation_L
        elif Transform_Type == 'Patch':
            Level_L = self.Patch_Split_L
        else:
            Level_L = None

        for level in Level_L:
            # TODO: 这里的工作流程需要变动一下, 应该是先产生样本, 再对样本进行切片和饱和度调整,
            # 设定好特定的装载程序之后前，在验证集上计算干净样本和对抗样本互信息并且计算准确率
            self.Test_Loader = self.get_test_loader()

            level_i_test_clean_acc, level_i_test_clean_loss = self.calculate_acc_and_mutual_info(Model,
                                                                                                 Transform_Type=Transform_Type,
                                                                                                 Level=level,
                                                                                                 Keep_Clean=True)
            level_i_test_adv_acc, level_i_test_adv_loss = self.calculate_acc_and_mutual_info(Model,
                                                                                             Transform_Type=Transform_Type,
                                                                                             Level=level,
                                                                                             Keep_Clean=False)
            # 在验证集上的干净样本准确率，对抗样本准确率,loss
            test_clean_acc_L.append(level_i_test_clean_acc)
            test_adv_acc_L.append(level_i_test_adv_acc)

            test_clean_loss_L.append(level_i_test_clean_loss)
            test_adv_loss_L.append(level_i_test_adv_loss)

            # print some data
            print('%s level_i[%d] '
                  'test_clean_loss[%.2f], test_adv_loss[%.2f] '
                  'test_clean_acc[%.2f%%],test_adv_acc[%.2f%%]'
                  % (Transform_Type, level,
                     level_i_test_clean_loss, level_i_test_adv_loss,
                     level_i_test_clean_acc, level_i_test_adv_acc))

        loss_acc = {
            'test_clean_loss': test_clean_loss_L,
            'test_adv_loss': test_adv_loss_L,
            'test_clean_acc': test_clean_acc_L,
            'test_adv_acc': test_adv_acc_L
        }
        # plot_performance(analytic_data, Enable_Adv_Training)
        self.Loss_Acc = loss_acc
        '''
        在保存数据之前，一定要清除layer_activations, layer_activations数据量真的太大了 
        '''
        self.std_estimator.clear_activations()
        self.adv_estimator.clear_activations()

        self.save_mutual_info_data(Transform_Type, Enable_Adv_Training)
        """
        在退出训练之前完成清理工作
        """
        self.std_estimator.clear_all()
        self.adv_estimator.clear_all()
        self.Loss_Acc = None
        print('the training has completed')

    def plot_data(self, Transform_Type, Enable_Adv_Training):
        from pylab import mpl

        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        # mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
        mpl.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

        from matplotlib.lines import Line2D
        line_legends = [Line2D([0], [0], color='purple', linewidth=1, linestyle='-', marker='o'),
                        Line2D([0], [0], color='purple', linewidth=1, linestyle='--', marker='^')]
        import math
        Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
        Model_Name = self.Model_Name
        with open('./Checkpoint/%s/%s/mi_loss_acc_%s.pkl' % (Model_Name, Transform_Type, Is_Adv_Training), 'rb') as f:
            mi_loss_acc = pickle.load(f)

        Forward_Size, Forward_Repeat = mi_loss_acc['Forward_Size'], mi_loss_acc['Forward_Repeat']
        std, adv = mi_loss_acc['std_estimator'], mi_loss_acc['adv_estimator']
        # Model_Name = basic_info['Model']
        Activation_F = 'relu'
        Learning_Rate = 0.08

        Std_Epoch_Num = len(std.epoch_MI_hM_X_upper)
        Epochs = [i for i in range(Std_Epoch_Num)]
        Layer_Num = len(std.epoch_MI_hM_X_upper[0])
        Layer_Name = [str(i) for i in range(Layer_Num)]

        # sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
        sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))

        title = "%s(%s),LR(%.3f),Upper/Lower/Bin,Clean(Adv),Sample_N(%d),%s,%s" % (
            Model_Name, Activation_F, Learning_Rate, Forward_Repeat * Forward_Size, Is_Adv_Training, Transform_Type
        )

        def axs_plot(axs, std_I_TX, std_I_TY, adv_I_TX, adv_I_TY, Std_Epoch_Num, MI_Type):
            std_I_TX = np.array(std_I_TX)
            std_I_TY = np.array(std_I_TY)
            adv_I_TX = np.array(adv_I_TX)
            adv_I_TY = np.array(adv_I_TY)

            # 设定坐标范围
            # i_tx_min = math.floor(min(np.min(std_I_TX), np.min(adv_I_TX))) - 0.1
            # i_tx_max = math.ceil(max(np.max(std_I_TX), np.max(adv_I_TX))) + 0.1
            #
            # i_ty_min = math.floor(min(np.min(std_I_TY), np.min(adv_I_TY))) - 0.1
            # i_ty_max = math.ceil(max(np.max(std_I_TY), np.max(adv_I_TY))) + 0.1

            i_tx_min = min(np.min(std_I_TX), np.min(adv_I_TX)) - 0.1
            i_tx_max = max(np.max(std_I_TX), np.max(adv_I_TX)) + 0.1

            i_ty_min = min(np.min(std_I_TY), np.min(adv_I_TY)) - 0.1
            i_ty_max = max(np.max(std_I_TY), np.max(adv_I_TY)) + 0.1

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
                # 设定 x 轴坐标范围
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
                # 设定 y 轴坐标范围
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
        # axs[nrows - 1][0].plot(Epochs, mi_loss_acc['train_loss'], label='train_loss')
        axs[nrows - 1][0].plot(Epochs, mi_loss_acc['loss_acc']['test_clean_loss'], label='test_clean_loss')
        axs[nrows - 1][0].plot(Epochs, mi_loss_acc['loss_acc']['test_adv_loss'], label='test_adv_loss')
        axs[nrows - 1][0].legend()
        # -------------------
        axs[nrows - 1][1].set_xlabel('epochs')
        axs[nrows - 1][1].set_title('acc')
        # axs[nrows - 1][1].plot(Epochs, analytic_data['train_acc'], label='train_acc')
        axs[nrows - 1][1].plot(Epochs, mi_loss_acc['loss_acc']['test_clean_acc'], label='test_clean_acc')
        axs[nrows - 1][1].plot(Epochs, mi_loss_acc['loss_acc']['test_adv_acc'], label='test_adv_acc')
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
        Enable_Detail = False
        if Enable_Detail:
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

    # 模型的类型固定，分别绘制不同程度的饱和度和分块设置在同一张图里面
    def plot_data_2(self):
        # TODO: 把不同模型的在 分块，饱和度实验下的结果一起展示。
        from pylab import mpl

        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        # mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
        mpl.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

        from matplotlib.lines import Line2D

        import math
        Is_Adv_Training = 'Std_Train'
        Model_Name = self.Model_Name
        Transform_Type = 'Saturation'
        with open('./Checkpoint/%s/%s/mi_loss_acc_%s.pkl' % (Model_Name, Transform_Type, Is_Adv_Training), 'rb') as f:
            st_saturation_mi_loss_acc = pickle.load(f)
        Transform_Type = 'Patch'
        with open('./Checkpoint/%s/%s/mi_loss_acc_%s.pkl' % (Model_Name, Transform_Type, Is_Adv_Training), 'rb') as f:
            st_patch_mi_loss_acc = pickle.load(f)

        Is_Adv_Training = 'Adv_Train'
        Model_Name = self.Model_Name
        Transform_Type = 'Saturation'
        with open('./Checkpoint/%s/%s/mi_loss_acc_%s.pkl' % (Model_Name, Transform_Type, Is_Adv_Training), 'rb') as f:
            at_saturation_mi_loss_acc = pickle.load(f)
        Transform_Type = 'Patch'
        with open('./Checkpoint/%s/%s/mi_loss_acc_%s.pkl' % (Model_Name, Transform_Type, Is_Adv_Training), 'rb') as f:
            at_patch_mi_loss_acc = pickle.load(f)

        Forward_Size, Forward_Repeat = st_saturation_mi_loss_acc['Forward_Size'], \
                                       st_saturation_mi_loss_acc['Forward_Repeat']
        st_saturation_std, st_saturation_adv = st_saturation_mi_loss_acc['std_estimator'], \
                                               st_saturation_mi_loss_acc['adv_estimator']

        st_patch_std, st_patch_adv = st_patch_mi_loss_acc['std_estimator'], \
                                     st_patch_mi_loss_acc['adv_estimator']

        at_saturation_std, at_saturation_adv = at_saturation_mi_loss_acc['std_estimator'], \
                                               at_saturation_mi_loss_acc['adv_estimator']
        at_patch_std, at_patch_adv = at_patch_mi_loss_acc['std_estimator'], \
                                     at_patch_mi_loss_acc['adv_estimator']
        # Model_Name = basic_info['Model']

        Level_Num_Max = max(len(st_saturation_std.epoch_MI_hM_X_upper), len(st_patch_std.epoch_MI_hM_X_upper))
        print('Level_Num_Max', Level_Num_Max)
        Layer_Num = len(st_saturation_std.epoch_MI_hM_X_upper[0])
        Layer_Name = [str(i + 1) for i in range(Layer_Num)]
        Saturation_L = [str(i) for i in self.Saturation_L]
        Patch_L = [str(i) for i in self.Patch_Split_L]

        # sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
        # bins = [i for i in range(Level_Num_Max)]
        # nbin = len(bins) - 1
        # import matplotlib as mpl
        # cmap = mpl.cm.get_cmap('viridis', nbin)
        # bd_norm = mpl.colors.BoundaryNorm(bins, nbin)
        # sm = mpl.cm.ScalarMappable(norm=bd_norm, cmap=cmap)
        # Level_Num_Max = 5 则数据应该在 -0.5 ~ +4.5之间
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=-0.5, vmax=Level_Num_Max - 0.5))

        title = "%s,Upper/Lower/Bin,Clean(Adv),Sample_N(%d),Std/Adv train" % (
            Model_Name, Forward_Repeat * Forward_Size,
        )

        COLOR = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',
                 'C6', 'C7', 'C8', 'C9', 'olive', 'peach', ]

        # fig size, 先列后行
        nrows = 2
        ncols = 4
        # px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        px = 1
        plt.show()
        Fig_Size = (5, 11)
        fig, axs = plt.subplots(nrows, ncols, figsize=(Fig_Size[1] * px, Fig_Size[0] * px), )

        # -------------Saturation (Standard Training) ST Loss and Accuracy Detail -------------------------
        axs[0][0].set_xlabel('Saturation level')
        axs[0][0].set_ylabel('Loss')

        # axs[0][0].plot(Epochs, st_saturation_mi_loss_acc['train_loss'], label='train_loss')
        # 颜色表示是ST还是AT, 实线和虚线表示正常样本还是对抗样本, marker 表示是 Saturation 还是 Patch
        axs[0][0].plot(Saturation_L, st_saturation_mi_loss_acc['loss_acc']['test_clean_loss'],
                       linestyle='-', c='C0', marker='x', markerfacecolor='none',
                       label='ST clean test')
        axs[0][0].plot(Saturation_L, st_saturation_mi_loss_acc['loss_acc']['test_adv_loss'],
                       linestyle=':', c='C0', marker='x', markerfacecolor='none',
                       label='ST adv test')

        axs[0][0].plot(Saturation_L, at_saturation_mi_loss_acc['loss_acc']['test_clean_loss'],
                       linestyle='-', c='C1', marker='x', markerfacecolor='none',
                       label='AT clean test')
        axs[0][0].plot(Saturation_L, at_saturation_mi_loss_acc['loss_acc']['test_adv_loss'],
                       linestyle=':', c='C1', marker='x', markerfacecolor='none',
                       label='AT adv test')
        axs[0][0].legend()

        axs[0][1].set_xlabel('Saturation level')
        axs[0][1].set_ylabel('Accuracy (%)')
        # axs[0][1].set_title('Standard training')
        # axs[0][1].plot(Epochs, analytic_data['train_acc'], label='train_acc')
        axs[0][1].plot(Saturation_L, st_saturation_mi_loss_acc['loss_acc']['test_clean_acc'],
                       linestyle='-', c='C0', marker='x', markerfacecolor='none', markersize=7)
        axs[0][1].plot(Saturation_L, st_saturation_mi_loss_acc['loss_acc']['test_adv_acc'],
                       linestyle=':', c='C0', marker='x', markerfacecolor='none', markersize=7)

        axs[0][1].plot(Saturation_L, at_saturation_mi_loss_acc['loss_acc']['test_clean_acc'],
                       linestyle='-', c='C1', marker='x', markerfacecolor='none', markersize=7)
        axs[0][1].plot(Saturation_L, at_saturation_mi_loss_acc['loss_acc']['test_adv_acc'],
                       linestyle=':', c='C1', marker='x', markerfacecolor='none', markersize=7)

        # ------------- Patch AT Loss and Accuracy Detail -------------------------
        axs[0][2].set_xlabel('Patch level')
        axs[0][2].set_ylabel('Loss')
        # axs[0][2].set_title('Adversarial training')
        # axs[0][0].plot(Epochs, st_saturation_mi_loss_acc['train_loss'], label='train_loss')
        axs[0][2].plot(Patch_L, st_patch_mi_loss_acc['loss_acc']['test_clean_loss'],
                       linestyle='-', c='C0', marker='s', markerfacecolor='none', markersize=7,
                       label='ST clean test')
        axs[0][2].plot(Patch_L, st_patch_mi_loss_acc['loss_acc']['test_adv_loss'],
                       linestyle=':', c='C0', marker='s', markerfacecolor='none', markersize=7,
                       label='ST adv test')

        axs[0][2].plot(Patch_L, at_patch_mi_loss_acc['loss_acc']['test_clean_loss'],
                       linestyle='-', c='C1', marker='s', markerfacecolor='none', markersize=7,
                       label='AT clean test')
        axs[0][2].plot(Patch_L, at_patch_mi_loss_acc['loss_acc']['test_adv_loss'],
                       linestyle=':', c='C1', marker='s', markerfacecolor='none', markersize=7,
                       label='AT adv test')

        axs[0][2].legend()
        # -------------------
        axs[0][3].set_xlabel('Patch level')
        axs[0][3].set_ylabel('Accuracy (%)')
        # axs[0][3].set_title('Adversarial training')
        # axs[0][1].plot(Epochs, analytic_data['train_acc'], label='train_acc')
        axs[0][3].plot(Patch_L, st_patch_mi_loss_acc['loss_acc']['test_clean_acc'],
                       linestyle='-', c='C0', marker='s', markerfacecolor='none', markersize=7)
        axs[0][3].plot(Patch_L, st_patch_mi_loss_acc['loss_acc']['test_adv_acc'],
                       linestyle=':', c='C0', marker='s', markerfacecolor='none', markersize=7)

        axs[0][3].plot(Patch_L, at_patch_mi_loss_acc['loss_acc']['test_clean_acc'],
                       linestyle='-', c='C1', marker='s', markerfacecolor='none', markersize=7)
        axs[0][3].plot(Patch_L, at_patch_mi_loss_acc['loss_acc']['test_adv_acc'],
                       linestyle=':', c='C1', marker='s', markerfacecolor='none', markersize=7)
        # axs[0][1].legend()

        # 初始化 xlabel, y_label
        for i in range(nrows):
            for j in range(ncols):
                axs[i][j].grid(True)

        # range(开始，结束，步长)
        # 绘制每一轮次的信息曲线
        def axs_plot(axs, std_I_TX, std_I_TY, adv_I_TX, adv_I_TY, levels, transform_type, MI_Type):
            std_I_TX = np.array(std_I_TX)
            std_I_TY = np.array(std_I_TY)
            adv_I_TX = np.array(adv_I_TX)
            adv_I_TY = np.array(adv_I_TY)
            marker_style = 's' if transform_type == 'Patch' else 'x'

            # 设定坐标范围
            # i_tx_min = math.floor(min(np.min(std_I_TX), np.min(adv_I_TX))) - 0.1
            # i_tx_max = math.ceil(max(np.max(std_I_TX), np.max(adv_I_TX))) + 0.1
            #
            # i_ty_min = math.floor(min(np.min(std_I_TY), np.min(adv_I_TY))) - 0.1
            # i_ty_max = math.ceil(max(np.max(std_I_TY), np.max(adv_I_TY))) + 0.1

            i_tx_min = min(np.min(std_I_TX), np.min(adv_I_TX)) - 0.1
            i_tx_max = max(np.max(std_I_TX), np.max(adv_I_TX)) + 0.1

            i_ty_min = min(np.min(std_I_TY), np.min(adv_I_TY)) - 0.1
            i_ty_max = max(np.max(std_I_TY), np.max(adv_I_TY)) + 0.1

            for idx, level_i in enumerate(levels):
                # c = COLOR[idx]
                c = sm.to_rgba(idx)
                # layers = [i for i in range(1,len(I_TX)+1)]
                std_I_TX_level_i, std_I_TY_level_i = std_I_TX[idx], std_I_TY[idx]
                adv_I_TX_level_i, adv_I_TY_level_i = adv_I_TX[idx], adv_I_TY[idx]

                # axs[0].set_title(MI_Type)
                # axs[0].grid()
                axs[0].set_xlabel('Layer index')
                axs[0].set_ylabel(r'$I(T;X)$' + ' ' + '(bits)')

                # axs[1].grid()
                axs[1].set_xlabel('Layer index')
                axs[1].set_ylabel(r'$I(T;Y)$' + ' ' + '(bits)')

                # axs[1].legend(line_legends, ['st_saturation_std', 'st_saturation_adv'])

                axs[0].plot(Layer_Name, std_I_TX_level_i,
                            linestyle='-',  # 对抗样本还是普通样本
                            color=c,  # level 水平
                            marker=marker_style,  # 是 saturation 还是 Patch
                            markerfacecolor='none',
                            linewidth=1,
                            label='%s %s(%s)' % ('std', transform_type, level_i)
                            )
                axs[0].plot(Layer_Name, adv_I_TX_level_i,
                            linestyle=':',
                            color=c, marker=marker_style, markerfacecolor='none',
                            linewidth=1,
                            label='%s %s(%s)' % ('adv', transform_type, level_i)
                            )
                # 设定 x 轴坐标范围
                # axs[0].set_ylim((i_tx_min, i_tx_max))
                # axs[1].set_ylim((i_tx_min, i_tx_max))

                axs[1].plot(Layer_Name, std_I_TY_level_i,
                            color=c, marker=marker_style,
                            linestyle='-', linewidth=1, markerfacecolor='none',
                            label='%s %s(%s)' % ('std', transform_type, level_i)
                            )
                axs[1].plot(Layer_Name, adv_I_TY_level_i,
                            color=c, marker=marker_style, markerfacecolor='none',
                            linestyle=':', linewidth=1,
                            label='%s %s(%s)' % ('adv', transform_type, level_i)
                            )
                # 设定 y 轴坐标范围
                # axs[2].set_ylim((i_ty_min, i_ty_max))
                # axs[3].set_ylim((i_ty_min, i_ty_max))

                # axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,prop={'size':8},ncol=2)
                # axs[0].legend(bbox_to_anchor=(0, -1.5, 1, 1), loc='upper left', borderaxespad=0.,
                #               prop={'size': 8}, ncol=len(self.Saturation_L)*2)

        # st_saturation_std/st_saturation_adv Upper
        saturation_levels = [str(i) for i in self.Saturation_L]
        patch_levels = [str(i) for i in self.Patch_Split_L]

        # axs_plot(axs[1],
        #          st_saturation_std.epoch_MI_hM_X_upper, st_saturation_std.epoch_MI_hM_Y_upper,
        #          st_saturation_adv.epoch_MI_hM_X_upper, st_saturation_adv.epoch_MI_hM_Y_upper,
        #          saturation_levels, transform_type='Saturation', MI_Type='upper'
        #          )
        #
        # axs_plot(axs[1],
        #          st_patch_std.epoch_MI_hM_X_upper, st_patch_std.epoch_MI_hM_Y_upper,
        #          st_patch_adv.epoch_MI_hM_X_upper, st_patch_adv.epoch_MI_hM_Y_upper,
        #          patch_levels, transform_type='Patch', MI_Type='upper'
        #          )

        # st_saturation_std/st_saturation_adv Lower
        axs[1][0].set_title('Standard training')
        axs[1][1].set_title('Standard training')
        axs_plot([axs[1][0], axs[1][1]],
                 st_saturation_std.epoch_MI_hM_X_lower, st_saturation_std.epoch_MI_hM_Y_lower,
                 st_saturation_adv.epoch_MI_hM_X_lower, st_saturation_adv.epoch_MI_hM_Y_lower,
                 saturation_levels, transform_type='Saturation', MI_Type='lower'
                 )
        # # st_saturation_std/st_saturation_adv Bin
        axs_plot([axs[1][0], axs[1][1]],
                 st_patch_std.epoch_MI_hM_X_lower, st_patch_std.epoch_MI_hM_Y_lower,
                 st_patch_adv.epoch_MI_hM_X_lower, st_patch_adv.epoch_MI_hM_Y_lower,
                 patch_levels, transform_type='Patch', MI_Type='lower'
                 )
        # axs[1][0].legend(ncol=1, prop={'size': 10}, loc='upper right', bbox_to_anchor=(-0.2, 1))
        # axs[1][0].legend(ncol=2, prop={'size': 10})
        # axs[1][0].legend(ncol=2,prop={'size': 10})

        # st_saturation_std/st_saturation_adv Lower
        axs[1][2].set_title('Adversarial training')
        axs[1][3].set_title('Adversarial training')
        axs_plot([axs[1][2], axs[1][3]],
                 at_saturation_std.epoch_MI_hM_X_lower, at_saturation_std.epoch_MI_hM_Y_lower,
                 at_saturation_adv.epoch_MI_hM_X_lower, at_saturation_adv.epoch_MI_hM_Y_lower,
                 saturation_levels, transform_type='Saturation', MI_Type='lower'
                 )
        # # st_saturation_std/st_saturation_adv Bin
        axs_plot([axs[1][2], axs[1][3]],
                 at_patch_std.epoch_MI_hM_X_lower, at_patch_std.epoch_MI_hM_Y_lower,
                 at_patch_adv.epoch_MI_hM_X_lower, at_patch_adv.epoch_MI_hM_Y_lower,
                 patch_levels, transform_type='Patch', MI_Type='lower'
                 )
        # axs[1][3].legend(ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
        # axs[1][3].legend(ncol=1, prop={'size': 10})
        line_legends = [Line2D([0], [0], linestyle='-', c='C0', marker='x', markerfacecolor='none'),
                        Line2D([0], [0], linestyle=':', c='C0', marker='x', markerfacecolor='none'),
                        Line2D([0], [0], linestyle='-', c='C0', marker='s', markerfacecolor='none'),
                        Line2D([0], [0], linestyle=':', c='C0', marker='s', markerfacecolor='none')
                        ]
        axs[1][0].legend(line_legends, ['Saturation clean', 'Saturation adv', 'Patch clean', 'Patch adv'])
        ticks_2_labels = ['Saturation 2 (Patch 0)', 'Saturation 8 (Patch 2)', 'Saturation 16 (Patch 4)',
                          'Saturation 64 (Patch 8)', 'Saturation 1024']
        import matplotlib
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: ticks_2_labels[pos])  # print(x,pos)
        fig.colorbar(sm, ax=axs[1][3], ticks=[i for i in range(Level_Num_Max)], format=fmt)
        # orientation='horizontal'

        # fig.suptitle(title)
        # fig.colorbar(sm, ax=axs, label='Epoch')

        # if Enable_Show:
        # plt.show()
        fig.savefig('mi_plane_transformation_%s.pdf' % (Model_Name.lower()))

        print("Work has done!")

    def plot_data_by_layer_index(self, Transform_Type, Enable_Adv_Training):
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        # 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend patches = [mpatches.Patch(linestyle=line_styles[i],
        # label="{:s}".format(labels[i])) for i in range(len(line_styles))]
        from matplotlib.lines import Line2D

        # color = 'purple' or 'orange'
        line_legends = [
            Line2D([0], [0], color='C0', linewidth=1, linestyle='-', marker='o', markerfacecolor='none', markersize=10),
            Line2D([0], [0], color='Red', linewidth=1, linestyle='-', marker='+', markersize=10)]

        # linestyle='None' 设置为 None 就可以进行相应的 Marker
        marker_legends = [
            Line2D([0], [0], color='C0', linestyle='None', marker='o', markerfacecolor='none', markersize=10),
            Line2D([0], [0], color='Red', linestyle='None', marker='+', markersize=10)]

        if Transform_Type == 'Saturation':
            Level_L = self.Saturation_L
        elif Transform_Type == 'Patch':
            Level_L = self.Patch_Split_L
        else:
            raise RuntimeError('Unknown Transform_Type: %s' % Transform_Type)
        Level_L = [str(i) for i in Level_L]

        Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'

        with open('./Checkpoint/%s/%s/mi_loss_acc_%s.pkl' % (self.Model_Name, Transform_Type, 'Std_Train'),
                  'rb') as f:
            ST_mi_loss_acc = pickle.load(f)
        with open('./Checkpoint/%s/%s/mi_loss_acc_%s.pkl' % (self.Model_Name, Transform_Type, 'Adv_Train'),
                  'rb') as f:
            AT_mi_loss_acc = pickle.load(f)

        Forward_Size, Forward_Repeat = ST_mi_loss_acc['Forward_Size'], ST_mi_loss_acc['Forward_Repeat']
        st_std, st_adv = ST_mi_loss_acc['std_estimator'], ST_mi_loss_acc['adv_estimator']
        at_std, at_adv = AT_mi_loss_acc['std_estimator'], AT_mi_loss_acc['adv_estimator']

        # Forward_Size, Forward_Repeat = basic_info['Forward_Size'], basic_info['Forward_Repeat']
        Activation_F = 'relu'
        Learning_Rate = 0.1
        # epoch(level) layer value
        Level_L_N = len(Level_L)
        Layer_Num = len(st_std.epoch_MI_hM_X_upper[0])
        Layer_Name = [str(i) for i in range(Layer_Num)]

        # Green = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))

        std_color, adv_color = 'winter_r', 'autumn_r'
        # cmap_std = plt.get_cmap('coolwarm')
        # cmap_adv = plt.get_cmap('coolwarm')
        # cmap_std = plt.get_cmap(std_color)
        # cmap_adv = plt.get_cmap(adv_color)

        # cmap_std = plt.get_cmap('Blues')  # summer 偏绿色
        # cmap_adv = plt.get_cmap('Reds')  # summer 偏红色
        # s_cmap_std = plt.cm.ScalarMappable(cmap=std_color, norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
        # s_cmap_adv = plt.cm.ScalarMappable(cmap=adv_color, norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
        # c_std = [cmap_std(i / Std_Epoch_Num * 1.0) for i in range(Std_Epoch_Num)]
        # c_adv = [cmap_adv(i / Std_Epoch_Num * 1.0) for i in range(Std_Epoch_Num)]
        # Red = plt.cm.ScalarMappable(cmap='cmap_adv', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
        # sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))

        label_formatter_float = FormatStrFormatter('%.2f')  # 设置x轴标签文本的格式
        label_formatter_int = FormatStrFormatter('%d')  # 设置y轴标签文本的格式

        # subplot2grid, size = （行,列）, 块起始点坐标
        # grid_size = (4, Layer_Num)
        Fig_Size = (25.6, 14.4)
        fig = plt.figure(figsize=Fig_Size, constrained_layout=True)
        spec = fig.add_gridspec(3, Layer_Num)

        # -------------------------------------------Loss and Accuracy Detail---------------------
        ax00 = fig.add_subplot(spec[0, 0])
        ax00.set_xlabel(Transform_Type)
        ax00.set_ylabel('Loss')
        # ax00.plot(Level_L, ST_mi_loss_acc['train_loss'], label='Train set')
        ax00.plot(Level_L, ST_mi_loss_acc['loss_acc']['test_clean_loss'], label='Std Clean test')
        ax00.plot(Level_L, ST_mi_loss_acc['loss_acc']['test_adv_loss'], label='Std Adv test')
        ax00.plot(Level_L, AT_mi_loss_acc['loss_acc']['test_clean_loss'], label='Adv Clean test')
        ax00.plot(Level_L, AT_mi_loss_acc['loss_acc']['test_adv_loss'], label='Adv Adv test')
        ax00.legend(prop={'size': 13})
        # -------------------
        ax01 = fig.add_subplot(spec[0, 1])
        ax01.set_xlabel(Transform_Type)
        ax01.set_ylabel('Accuracy (%)')
        # ax01.plot(Level_L, ST_mi_loss_acc['train_acc'], label='Train set')
        ax01.plot(Level_L, ST_mi_loss_acc['loss_acc']['test_clean_acc'], label='Std Clean test')
        ax01.plot(Level_L, ST_mi_loss_acc['loss_acc']['test_adv_acc'], label='Std Adv test')
        ax01.plot(Level_L, AT_mi_loss_acc['loss_acc']['test_clean_acc'], label='Adv Clean test')
        ax01.plot(Level_L, AT_mi_loss_acc['loss_acc']['test_adv_acc'], label='Adv Adv test')
        ax01.legend(prop={'size': 13})

        # -------------------------------------------Overlook by Upper mutual info-------------------------
        # ax02 = fig.add_subplot(spec[0, 2])
        # ax02.set_xlabel('Layer index')
        # ax02.set_ylabel(r'$I(T;X)$' + ' (bits)')
        # ax02.set_title('The I(T;X) lower bound')
        # ax02.legend(line_legends, ['st_std', 'st_adv'], prop={'size': 13})
        # 
        # ax03 = fig.add_subplot(spec[0, 3])
        # ax03.set_xlabel('Layer index')
        # ax03.set_ylabel(r'$I(T;Y)$' + ' (bits)')
        # ax03.set_title('The I(T;Y) lower bound')

        # for i, level_i in enumerate(Level_L):
        #     # st_std.epoch_MI_hM_X_lower, st_std.epoch_MI_hM_Y_lower,
        #     # st_adv.epoch_MI_hM_X_lower, st_adv.epoch_MI_hM_Y_lower,
        # 
        #     ax02.plot(Layer_Name, st_std.epoch_MI_hM_X_lower[i], color='Blue', marker='o')
        #     ax02.plot(Layer_Name, st_adv.epoch_MI_hM_X_lower[i], color='Red', marker='+')
        # 
        #     ax03.plot(Layer_Name, st_std.epoch_MI_hM_Y_lower[i], color='Blue', marker='o')
        #     ax03.plot(Layer_Name, st_adv.epoch_MI_hM_Y_lower[i], color='Red', marker='+')

        # -------------------------------------------Mutual Information spilt by Layer---------------------
        # 设定坐标范围
        # i_tx_min = math.floor(min(np.min(std_I_TX), np.min(adv_I_TX))) - 0.5
        # i_tx_max = math.ceil(max(np.max(std_I_TX), np.max(adv_I_TX)))
        #
        # i_ty_min = math.floor(min(np.min(std_I_TY), np.min(adv_I_TY))) - 0.5
        # i_ty_max = math.ceil(max(np.max(std_I_TY), np.max(adv_I_TY)))
        # TODO : 添加其他的数据
        st_std_itx_lower = np.array(st_std.epoch_MI_hM_X_lower)
        st_std_ity_lower = np.array(st_std.epoch_MI_hM_Y_lower)
        st_adv_itx_lower = np.array(st_adv.epoch_MI_hM_X_lower)
        st_adv_ity_lower = np.array(st_adv.epoch_MI_hM_Y_lower)

        at_std_itx_lower = np.array(at_std.epoch_MI_hM_X_lower)
        at_std_ity_lower = np.array(at_std.epoch_MI_hM_Y_lower)
        at_adv_itx_lower = np.array(at_adv.epoch_MI_hM_X_lower)
        at_adv_ity_lower = np.array(at_adv.epoch_MI_hM_Y_lower)
        # C0-C9 是 matplotlib 里经常使用的色条
        COLOR = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',
                 'C6', 'C7', 'C8', 'C9', 'olive', 'peach', ]

        for layer_i in range(Layer_Num):
            ax_itx = fig.add_subplot(spec[1, layer_i])
            ax_ity = fig.add_subplot(spec[2, layer_i])
            # 设定 y 标签的格式
            # ax_itx.yaxis.set_major_formatter(label_formatter_int)
            # ax_ity.yaxis.set_major_formatter(label_formatter_int)

            ax_itx.set_title('Layer %d' % layer_i)
            # epoch_i, layer_i, label_i
            ax_itx.plot(Level_L, st_std_itx_lower[..., layer_i], color='Blue', marker='o', label='st_std')
            ax_itx.plot(Level_L, st_adv_itx_lower[..., layer_i], color='Blue', linestyle=':', marker='^',
                        label='st_adv')

            ax_itx.plot(Level_L, at_std_itx_lower[..., layer_i], color='Red', marker='o', label='at_std')
            ax_itx.plot(Level_L, at_adv_itx_lower[..., layer_i], color='Red', linestyle=':', marker='+', label='at_adv')

            ax_ity.set_xlabel('%s level' % Transform_Type)
            ax_ity.plot(Level_L, st_std_ity_lower[..., layer_i], color='Blue', marker='o', label='st_std')
            ax_ity.plot(Level_L, st_adv_ity_lower[..., layer_i], color='Blue', linestyle=':', marker='^',
                        label='st_adv')

            ax_ity.plot(Level_L, at_std_ity_lower[..., layer_i], color='Red', marker='o', label='at_std')
            ax_ity.plot(Level_L, at_adv_ity_lower[..., layer_i], color='Red', linestyle=':', marker='^', label='at_adv')

            if layer_i == 0:
                # 只有第一个子图显示 legend 信息
                ax_itx.legend(ncol=1, prop={'size': 13})
                # 只有最左侧的子图显示 y label 信息
                ax_itx.set_ylabel(r'$I(T;X)$')
                ax_ity.set_ylabel(r'$I(T;Y)$')

        title = "%s(%s),LR(%.3f),MI Lower Bound detail,Clean(Adv),Sample_N(%d),%s" % (
            self.Model_Name, Activation_F, Learning_Rate, Forward_Repeat * Forward_Size, Is_Adv_Training
        )
        fig.suptitle(title)
        plt.show()
        # fig.savefig('mutual_info_detail_%s_%s.pdf' % (Model_Name, Is_Adv_Training))
        print("Work has done!")

    def calculate_transfer_matrix(self, Model, Enable_Adv_Training=False):
        # 计算模型的对样本的分类情况，以及置信度
        # 这里的epoch_i没必要指定，因为epochi就是列表当中的最后一个元素
        # a = list[-1]就是最后一个元素
        Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
        Model = Model.to(self.Device)
        Model.eval()

        label_chunk = None
        label_std_chunk, label_prob_std_chunk = None, None
        label_adv_chunk, label_prob_adv_chunk = None, None

        Test_loader_Iter = iter(self.Test_Loader)

        for i in range(self.Forward_Repeat):
            images_clean, labels = next(Test_loader_Iter)
            # 1. 保存真实标签
            # 2. 保存模型对 干净样本 的预测标签和概率
            # 3. 保存模型对 对抗样本 的预测标签和概率
            images_clean = images_clean.to(self.Device)
            labels = labels.to(self.Device)

            atk = self.test_attack(Model, Random_Start=False)

            images_adv = atk(images_clean, labels)

            """
            计算模型的准确率
            """
            # loss_i = F.cross_entropy(outputs, labels)
            # predicted_prob, predicted, labels 都可以看成是一个列表或者是一个向量，列表中元素的个数为 batch_size 个
            # 先对神经网络的输出结果做一个 softmax 获取概率值
            outputs_std = Model(images_clean)
            label_prob_std, label_std = torch.max(F.softmax(outputs_std, dim=1), dim=1)

            outputs_adv = Model(images_adv)
            label_prob_adv, label_adv = torch.max(F.softmax(outputs_adv, dim=1), dim=1)

            # correct_N += (predicted_std == labels).sum().item()
            # total_N += labels.size(0)
            # loss += loss_i.item()

            """
            发现并修改了一个重大bug, 这里每forward一次,caculate_MI 函数计算出的互信息值都直接挂在列表的后面，那么 Forward_Repeat 会成倍放大列表的长度
            且会混乱每一个 epoch 中的互信息变化情况，Forward_Repeat 一旦超过 epoch_num ，那么每一个 epoch 的曲线就会
            """
            # 给定初始值
            if i == 0:
                label_chunk = labels.clone().detach()
                # std
                label_std_chunk = label_std.clone().detach()
                label_prob_std_chunk = label_prob_std.clone().detach()
                # adv
                label_adv_chunk = label_adv.clone().detach()
                label_prob_adv_chunk = label_prob_adv.clone().detach()

                # 计算所有循环的和
            else:
                label_chunk = torch.cat((label_chunk, labels.clone().detach()), dim=0)
                # std
                label_std_chunk = torch.cat((label_std_chunk, label_std.clone().detach()), dim=0)
                label_prob_std_chunk = torch.cat((label_prob_std_chunk, label_prob_std.clone().detach()), dim=0)
                # adv
                label_adv_chunk = torch.cat((label_adv_chunk, label_adv.clone().detach()), dim=0)
                label_prob_adv_chunk = torch.cat((label_prob_adv_chunk, label_prob_adv.clone().detach()), dim=0)

        dir = 'Checkpoint/%s' % self.Model_Name
        # 对于每一个模型产生的数据, 使用一个文件夹单独存放
        if not os.path.exists(dir):
            os.makedirs(dir)

        transfer_matrix = {'label_chunk': label_chunk,
                           'label_std_chunk': label_std_chunk,
                           'label_prob_std_chunk': label_prob_std_chunk,
                           'label_adv_chunk': label_adv_chunk,
                           'label_prob_adv_chunk': label_prob_adv_chunk,
                           }

        with open('./Checkpoint/%s/transfer_matrix_%s.pkl' % (self.Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(transfer_matrix, f)
        print('Calculating Transfer Matrix was Done')


if __name__ == '__main__':
    # mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['backend'] = 'agg'
    # mpl.rcParams["font.size"] = 18
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
    mpl.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    # Random_Seed = 123
    # torch.manual_seed(Random_Seed)
    # torch.cuda.manual_seed(Random_Seed)  # 设置当前GPU的随机数生成种子
    # torch.cuda.manual_seed_all(Random_Seed)  # 设置所有GPU的随机数生成种子

    # analytic_data = training(Enable_Adv_Training=False)

    # analytic_data_2 = training(Model, Enable_Adv_Training=True)
    check_dir = ['DataSet/MNIST', 'DataSet/CIFAR10', 'Checkpoint']
    for dir in check_dir:
        if not os.path.exists(dir):
            os.makedirs(dir)

    from torchvision.models import *
    from Models.MNIST import FC_Sigmoid, Net_mnist, FC_2
    from Models.CIFAR10 import LeNet_3_32_32, WideResNet, VGG_s, RestNet18, net_cifar10
    from Models.Tiny_ImageNet import WideResNet_3_64_64, WideResNet_3_96_96
    import argparse

    Model_dict = {}
    # Model_dict['net_cifar10'] = net_cifar10()
    # Model_dict['VGG_s'] = VGG_s()
    # Model_dict['resnet18'] = resnet18(pretrained=False, num_classes=10)
    # Model_dict['resnet34'] = resnet34(pretrained=False, num_classes=10)
    # Model_dict['vgg11'] = vgg11(pretrained=False)
    # Model_dict['FC_2'] = FC_2(Activation_F=nn.ReLU())
    # Model_dict['LeNet_MNIST'] = LeNet_3_32_32()
    Model_dict['WideResNet_CIFAR10'] = WideResNet(depth=1 * 6 + 4, num_classes=10, widen_factor=1, dropRate=0.0)
    Model_dict['WideResNet_STL10'] = WideResNet_3_96_96(depth=1 * 6 + 4, num_classes=10, widen_factor=1,
                                                        dropRate=0.0)

    parser = argparse.ArgumentParser(description='Training arguments with PyTorch')
    parser.add_argument('--Model_Name', default='WideResNet_STL10', type=str, help='The Model_Name.')
    # parser.add_argument('--Model_Name', default='WideResNet_CIFAR10', type=str, help='The Model_Name.')

    parser.add_argument('--Data_Set', default='STL10', type=str, help='The Data_Set.')
    # parser.add_argument('--Data_Set', default='CIFAR10', type=str, help='The Data_Set.')

    parser.add_argument('--Forward_Size', default=500, type=int, help='Forward_Size.')
    parser.add_argument('--Forward_Repeat', default=10, type=int, help='Forward_Repeat')

    parser.add_argument('--GPU', default=0, type=int, help='The GPU id.')

    parser.add_argument('--Eps', default=4 / 255, type=float, help='perturbation magnitude')
    parser.add_argument('--Alpha', default=2 / 255, type=float, help='the perturbation in each step')
    parser.add_argument('--Step', default=7, type=int, help='the step')

    args = parser.parse_args()

    Model = Model_dict[args.Model_Name]
    Forward_0 = Forward(Model, args)
    # Forward_0.calculate_transfer_matrix(Model, Enable_Adv_Training=False)

    # Forward_0.forward(Model, Transform_Type='Saturation', Enable_Adv_Training=False)
    # Forward_0.forward(Model, Transform_Type='Saturation', Enable_Adv_Training=True)
    #
    # Forward_0.forward(Model, Transform_Type='Patch', Enable_Adv_Training=False)
    # Forward_0.forward(Model, Transform_Type='Patch', Enable_Adv_Training=True)
    #
    # Forward_0.plot_data(Transform_Type='Saturation', Enable_Adv_Training=False)
    # Forward_0.plot_data(Transform_Type='Saturation', Enable_Adv_Training=True)
    #
    # Forward_0.plot_data(Transform_Type='Patch', Enable_Adv_Training=False)
    # Forward_0.plot_data(Transform_Type='Patch', Enable_Adv_Training=True)

    # Forward_0.plot_data_by_layer_index(Transform_Type='Patch', Enable_Adv_Training=True)
    # Forward_0.plot_data_by_layer_index(Transform_Type='Saturation', Enable_Adv_Training=True)
    # Forward_0.plot_data_by_layer_index(Transform_Type='Patch', Enable_Adv_Training=True)

    Forward_0.plot_data_2()
    # Forward_0.plot_data_2(Enable_Adv_Training=True)
    # pass

"""
    def plot_mutual_info_2(epoch_MI_hM_X, epoch_MI_hM_Y, title):
        sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))

        plt.figure()
        plt.xlabel('I(T;X)')
        plt.ylabel('I(T;Y)')
        # 开始，结束，步长
        for i in range(Std_Epoch_Num):
            if i % 1 == 0:
                c = sm.to_rgba(i)
                I_TX, I_TY = epoch_MI_hM_X[i], epoch_MI_hM_Y[i]
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
        Is_Adv_Training = 'std_train'
        fig.savefig(
            './results_pdf/mutual_info_%s_%s.pdf' % (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                                                     Is_Adv_Training
                                                     ))

# # Res18
# modules_to_hook = ('conv1',
#                    'layer1.1.conv2',
#                    'layer2.1.conv2',
#                    'layer3.1.conv2',
#                    'layer4.1.conv2',
#                    'fc')
# Res34
# modules_to_hook = ('conv1',
#                    'layer1.2.conv2',
#                    'layer2.3.conv2',
#                    'layer3.5.conv2',
#                    'layer4.2.conv2',
#                    'fc')
# VGG11
# modules_to_hook = ('features.0',
#                    'features.7',
#                    'features.14',
#                    'features.21',
#                    'features.28',
#                    'classifier')

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
plt.title("%s(%s),LR(%.3f)" % (Model.name, Activation_F, Learning_Rate))
plt.colorbar(sm, label='Epoch')
fig = plt.gcf()
plt.show()
# fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
fig.savefig('./%s_%s_%s_%s_std.pdf' % (
    Model.name,
    Activation_F, str(Std_Epoch_Num),
    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))



# 4.1 Standard Accuracy
@torch.no_grad()
def mutual_info_calculate(Keep_Clean=True):
    # 这里的epoch_i没必要指定，因为epochi就是列表当中的最后一个元素
    # a = list[-1]就是最后一个元素
    import numpy as np
    Model.eval()
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
        estimator.do_forward_hook(Model)
        # forward
        Model(images)
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

"""
