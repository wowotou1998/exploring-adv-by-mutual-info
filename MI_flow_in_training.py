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
from Tiny_ImageNet_Loader import *


# mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
# mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.1, 0.95
# mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.1, 0.9
# mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.2, 0.4
# mpl.rcParams['figure.constrained_layout.use'] = True
# 选择模型
# Activation_F = 'Tanh'
# Activation_F = 'ReLU'


class Trainer():
    def __init__(self, Origin_Model, args):
        self.Args = args
        self.Model_Name = args.Model_Name
        self.Origin_Model = Origin_Model
        # self.Enable_Show = True
        self.Std_Epoch_Num = args.Std_Epoch_Num
        self.Forward_Size, self.Forward_Repeat = args.Forward_Size, args.Forward_Repeat
        self.Learning_Rate = args.Learning_Rate
        self.Train_Batch_Size = args.batch_size
        self.Device = torch.device("cuda:%d" % (args.GPU) if torch.cuda.is_available() else "cpu")
        self.Train_Loader, self.Test_Loader = self.get_train_test_loader()
        self.std_estimator = mutual_info_estimator(self.Origin_Model.modules_to_hook,
                                                   By_Layer_Name=False,
                                                   Label_Num=args.Label_Num,
                                                   Enable_Detail=True)
        self.adv_estimator = mutual_info_estimator(self.Origin_Model.modules_to_hook,
                                                   By_Layer_Name=False,
                                                   Label_Num=args.Label_Num,
                                                   Enable_Detail=True)

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

    def get_train_test_loader(self):
        # 全局取消证书验证
        import ssl
        import random
        ssl._create_default_https_context = ssl._create_unverified_context

        data_tf_3_32_32 = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        data_tf_3_64_64 = transforms.Compose([
            transforms.RandomCrop(64, padding=4, fill=0, padding_mode='constant'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        data_tf_3_96_96 = transforms.Compose([
            transforms.RandomCrop(96, padding=4, fill=0, padding_mode='constant'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        tensor_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        Data_Set = self.Args.Data_Set
        if Data_Set == 'CIFAR10':
            train_dataset = datasets.CIFAR10(root='./DataSet/CIFAR10', train=True, transform=data_tf_3_32_32,
                                             download=True)
            test_dataset = datasets.CIFAR10(root='./DataSet/CIFAR10', train=False, transform=tensor_transform,
                                            download=True)
        elif Data_Set == 'STL10':
            train_dataset = datasets.STL10(root='./DataSet/STL10', split='train', transform=data_tf_3_96_96,
                                           download=True)
            test_dataset = datasets.STL10(root='./DataSet/STL10', split='test', transform=tensor_transform,
                                          download=True)
        elif Data_Set == 'SVHN':
            train_dataset = datasets.SVHN(root='./DataSet/SVHN', split='train', transform=data_tf_3_64_64,
                                          download=True)
            test_dataset = datasets.SVHN(root='./DataSet/SVHN', split='test', transform=tensor_transform,
                                         download=True)
        elif Data_Set == 'TinyImageNet':
            train_dataset = TrainTinyImageNetDataset(id=get_id_dict(), transform=data_tf_3_64_64)
            test_dataset = TestTinyImageNetDataset(id=get_id_dict(), transform=tensor_transform)
        elif Data_Set == 'MNIST':
            train_dataset = datasets.MNIST(root='./DataSet/MNIST', train=True, transform=tensor_transform,
                                           download=True)
            test_dataset = datasets.MNIST(root='./DataSet/MNIST', train=False, transform=tensor_transform)
        else:
            raise RuntimeError('invaild data set')

        Train_Loader = DataLoader(dataset=train_dataset, batch_size=self.Train_Batch_Size, shuffle=True)
        Test_Loader = DataLoader(dataset=test_dataset, batch_size=self.Forward_Size, shuffle=True)
        return Train_Loader, Test_Loader

    def save_mutual_info_data(self, std_estimator, adv_estimator, analytic_data, Enable_Adv_Training):
        Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
        Model_Name, Forward_Size, Forward_Repeat = self.Model_Name, self.Forward_Size, self.Forward_Repeat
        dir = 'Checkpoint/%s' % Model_Name
        # 对于每一个模型产生的数据, 使用一个文件夹单独存放
        if not os.path.exists(dir):
            os.makedirs(dir)

        basic_info = {'Model': Model_Name,
                      'Enable_Adv_Training': Enable_Adv_Training,
                      'Forward_Size': Forward_Size,
                      'Forward_Repeat': Forward_Repeat,
                      }

        std, adv = std_estimator, adv_estimator
        with open('./Checkpoint/%s/basic_info_%s.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(basic_info, f)
        with open('./Checkpoint/%s/loss_and_acc_%s.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(analytic_data, f)
        with open('./Checkpoint/%s/loss_and_mutual_info_%s_std.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(std, f)
        with open('./Checkpoint/%s/loss_and_mutual_info_%s_adv.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(adv, f)

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

    # this training function is only for classification task

    @torch.no_grad()
    def calculate_acc_and_mutual_info(self, Model, Keep_Clean):
        # 这里的epoch_i没必要指定，因为epochi就是列表当中的最后一个元素
        # a = list[-1]就是最后一个元素
        Model.eval()

        correct_N = 0
        total_N = 0
        loss = 0.

        image_chunk = None
        label_chunk = None
        layer_activation_chunk = None

        if Keep_Clean:
            estimator = self.std_estimator
        else:
            estimator = self.adv_estimator

        for i in range(self.Forward_Repeat):

            images, labels = self.get_clean_or_adv_image(Model, Keep_Clean)

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
        estimator.caculate_MI(image_chunk, label_chunk)
        estimator.store_MI()

        acc = correct_N * 100. / total_N
        return acc, loss / self.Forward_Repeat

    def training(self, Enable_Adv_Training):
        checkpoint_path_dir = "Checkpoint/%s" % (self.Model_Name)
        if not os.path.exists(checkpoint_path_dir):
            os.makedirs(checkpoint_path_dir)

        import copy
        Model = copy.deepcopy(self.Origin_Model)

        train_loss = []
        train_acc = []
        test_clean_acc, test_adv_acc = [], []
        test_clean_loss, test_adv_loss = [], []

        optimizer = optim.SGD(Model.parameters(),
                              lr=self.Learning_Rate,
                              momentum=0.9,
                              )
        if Enable_Adv_Training:
            optimizer = optim.SGD(Model.parameters(),
                                  lr=self.Learning_Rate,
                                  momentum=0.9,
                                  # weight_decay=2e-4
                                  )
            # optimizer = optim.Adam(Model.parameters(),
            #                        lr=self.Learning_Rate)

        # milestones = [int(self.Std_Epoch_Num * 0.2) + 1, int(self.Std_Epoch_Num * 0.6) + 1]
        # milestones = [200]
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60], gamma=0.5)

        criterion = nn.CrossEntropyLoss()

        # Load checkpoint.
        # if Enable_Adv_Training:
        #     # 装载训练好的模型
        #     print('--> %s is adv training...' % Model_Name)
        #     print('--> Loading Model state dict..')
        #     load_model(Model, './Checkpoint/%s_std.pth' % Model_Name)
        #     print('--> Load checkpoint successfully! ')

        Model = Model.to(self.Device)
        Model.train()

        for epoch_i in range(self.Std_Epoch_Num):

            train_loss_sum, train_acc_sum, sample_sum = 0.0, 0.0, 0

            # 在每次训练之前，在验证集上计算干净样本和对抗样本互信息并且计算准确率
            # if (epoch_i + 1) % 3 == 0:
            epoch_test_clean_acc, epoch_test_clean_loss = self.calculate_acc_and_mutual_info(Model, Keep_Clean=True)
            epoch_test_adv_acc, epoch_test_adv_loss = self.calculate_acc_and_mutual_info(Model, Keep_Clean=False)
            # 在验证集上的干净样本准确率，对抗样本准确率,loss
            test_clean_acc.append(epoch_test_clean_acc)
            test_adv_acc.append(epoch_test_adv_acc)

            test_clean_loss.append(epoch_test_clean_loss)
            test_adv_loss.append(epoch_test_adv_loss)

            for batch_images, batch_labels in self.Train_Loader:

                # data moved to GPU
                batch_labels = batch_labels.to(self.Device)
                batch_images = batch_images.to(self.Device)

                if Enable_Adv_Training:
                    atk = self.train_attack(Model, Random_Start=True)
                    batch_images = atk(batch_images, batch_labels)

                outputs = Model(batch_images)

                if epoch_i == 0 and sample_sum == 0:
                    print(self.Device)
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
            train_loss.append(train_loss_sum / len(self.Train_Loader))
            # 训练准确率
            epoch_train_acc = (train_acc_sum / sample_sum) * 100.0
            train_acc.append(epoch_train_acc)

            # print some data
            print('epoch_i[%d] '
                  'train_loss[%.2f], test_clean_loss[%.2f], test_adv_loss[%.2f] '
                  'train_acc[%.2f%%],test_clean_acc[%.2f%%],test_adv_acc[%.2f%%]'
                  % (epoch_i + 1,
                     train_loss_sum / len(self.Train_Loader), epoch_test_clean_loss, epoch_test_adv_loss,
                     epoch_train_acc, epoch_test_clean_acc, epoch_test_adv_acc))

        # Save checkpoint.
        checkpoint_path = "./Checkpoint/%s/%s_%s.pth" % (
            self.Model_Name,
            self.Model_Name,
            'adv' if Enable_Adv_Training else 'std')
        save_model(Model, checkpoint_path)

        analytic_data = {
            'train_loss': train_loss,
            'test_clean_loss': test_clean_loss,
            'test_adv_loss': test_adv_loss,
            'train_acc': train_acc,
            'test_clean_acc': test_clean_acc,
            'test_adv_acc': test_adv_acc
        }
        # plot_performance(analytic_data, Enable_Adv_Training)
        self.save_mutual_info_data(self.std_estimator, self.adv_estimator, analytic_data, Enable_Adv_Training)
        # plot_mutual_info_2(std_estimator.epoch_MI_hM_X_upper, std_estimator.epoch_MI_hM_Y_upper, title='std_upper')
        # plot_mutual_info_2(std_estimator.epoch_MI_hM_X_bin, std_estimator.epoch_MI_hM_Y_bin, title='std_bin')
        # plot_mutual_info_2(adv_estimator.epoch_MI_hM_X_upper, adv_estimator.epoch_MI_hM_Y_upper, title='adv_upper')
        # plot_mutual_info_2(adv_estimator.epoch_MI_hM_X_bin, adv_estimator.epoch_MI_hM_Y_bin, title='adv_bin')
        """
        在退出训练之前完成清理工作
        """
        self.std_estimator.clear_all()
        self.adv_estimator.clear_all()
        print('the training has completed')
        return analytic_data

    def only_forward(self, Model, Enable_Adv_Training):

        Model = Model.to(self.Device)
        Model.eval()

        epoch_test_clean_acc, epoch_test_clean_loss = self.calculate_acc_and_mutual_info(Model, Keep_Clean=True)
        epoch_test_adv_acc, epoch_test_adv_loss = self.calculate_acc_and_mutual_info(Model, Keep_Clean=False)
        print('test_clean_acc[%.2f], test_clean_loss[%.2f],test_adv_acc[%.2f], test_adv_loss[%.2f]' % (
            epoch_test_clean_acc, epoch_test_clean_loss, epoch_test_adv_acc, epoch_test_adv_loss))
        analytic_data = {
            'train_loss': [0.],
            'test_clean_loss': [epoch_test_clean_loss],
            'test_adv_loss': [epoch_test_adv_loss],
            'train_acc': [0],
            'test_clean_acc': [epoch_test_clean_acc],
            'test_adv_acc': [epoch_test_adv_acc]
        }

        # plot_performance(analytic_data, Enable_Adv_Training)

        Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
        Model_Name, Forward_Size, Forward_Repeat = self.Model_Name, self.Forward_Size, self.Forward_Repeat
        dir = 'Checkpoint/%s/only_forward' % Model_Name
        # 对于每一个模型产生的数据, 使用一个文件夹单独存放
        if not os.path.exists(dir):
            os.makedirs(dir)

        basic_info = {'Model': Model_Name,
                      'Enable_Adv_Training': Enable_Adv_Training,
                      'Forward_Size': Forward_Size,
                      'Forward_Repeat': Forward_Repeat,
                      }

        with open('./Checkpoint/%s/basic_info_%s.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(basic_info, f)
        with open('./Checkpoint/%s/loss_and_acc_%s.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(analytic_data, f)
        with open('./Checkpoint/%s/loss_and_mutual_info_%s_std.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(self.std_estimator, f)
        with open('./Checkpoint/%s/loss_and_mutual_info_%s_adv.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(self.adv_estimator, f)

        """
        在退出之前完成清理工作
        """
        self.std_estimator.clear_all()
        self.adv_estimator.clear_all()
        print('the only forward has completed')
        return analytic_data

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

        dir = 'Checkpoint/%s' % Model_Name
        # 对于每一个模型产生的数据, 使用一个文件夹单独存放
        if not os.path.exists(dir):
            os.makedirs(dir)

        transfer_matrix = {'label_chunk': label_chunk,
                           'label_std_chunk': label_std_chunk,
                           'label_prob_std_chunk': label_prob_std_chunk,
                           'label_adv_chunk': label_adv_chunk,
                           'label_prob_adv_chunk': label_prob_adv_chunk,
                           }

        with open('./Checkpoint/%s/transfer_matrix_%s.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(transfer_matrix, f)
        print('Calculating Transfer Matrix was Done')


if __name__ == '__main__':
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
    from Models.CIFAR10 import LeNet_cifar10, WideResNet, VGG_s, RestNet18, net_cifar10
    from Models.Tiny_ImageNet import WideResNet_3_64_64, WideResNet_3_96_96
    import argparse

    Model_dict = {}
    Model_dict['FC_2'] = FC_2(Activation_F=nn.ReLU())
    Model_dict['LeNet_cifar10'] = LeNet_cifar10()
    Model_dict['net_cifar10'] = net_cifar10()
    Model_dict['VGG_s'] = VGG_s()
    Model_dict['resnet18'] = resnet18(pretrained=False, num_classes=10)
    Model_dict['resnet34'] = resnet34(pretrained=False, num_classes=10)
    Model_dict['vgg11'] = vgg11(pretrained=False)
    Model_dict['WideResNet'] = WideResNet(depth=1 * 6 + 4, num_classes=10, widen_factor=1, dropRate=0.0)
    Model_dict['WideResNet_SVHN'] = WideResNet(depth=1 * 6 + 4, num_classes=10, widen_factor=1, dropRate=0.0)
    Model_dict['WideResNet_Tiny_ImageNet'] = WideResNet_3_64_64(depth=1 * 6 + 4, num_classes=200, widen_factor=1,
                                                                dropRate=0.0)
    Model_dict['WideResNet_STL10'] = WideResNet_3_96_96(depth=1 * 6 + 4, num_classes=10, widen_factor=1,
                                                        dropRate=0.0)

    parser = argparse.ArgumentParser(description='Training arguments with PyTorch')
    # parser.add_argument('--Model_Name', default='LeNet_cifar10', type=str, help='The Model_Name.')
    parser.add_argument('--Model_Name', default='WideResNet_STL10', type=str, help='The Model_Name.')
    parser.add_argument('--Data_Set', default='STL10', type=str, help='The Data_Set.')
    parser.add_argument('--Label_Num', default=10, type=int, help='The Label_Num.')

    parser.add_argument('--Std_Epoch_Num', default=100, type=int, help='The epochs.')
    parser.add_argument('--Learning_Rate', default=0.1, type=float, help='The learning rate.')
    parser.add_argument('--Forward_Size', default=250, type=int, help='Forward_Size.')
    parser.add_argument('--Forward_Repeat', default=4, type=bool, help='Forward_Repeat')
    parser.add_argument('--GPU', default=0, type=int, help='The GPU id.')
    parser.add_argument('--batch_size', default=128, type=int, help='The Train_Batch_Size.')

    parser.add_argument('--Eps', default=4 / 255, type=float, help='perturbation magnitude')
    parser.add_argument('--Alpha', default=2 / 255, type=float, help='the perturbation in each step')
    parser.add_argument('--Step', default=7, type=int, help='the step')

    args = parser.parse_args()

    Data_Set = args.Data_Set
    Model_Name = args.Model_Name
    Model = Model_dict[Model_Name]
    Trainer_0 = Trainer(Model, args)
    Trainer_0.training(Enable_Adv_Training=True)
    Trainer_0.training(Enable_Adv_Training=False)

    # Trainer_0.calculate_transfer_matrix(Model, Enable_Adv_Training=False)

    # load_model(Model, './Checkpoint/%s_std.pth' % Model_Name)
    # Trainer_0.only_forward(Model, Enable_Adv_Training=False)
    # load_model(Model, './Checkpoint/%s_adv.pth' % Model_Name)
    # Trainer_0.only_forward(Model, Enable_Adv_Training=True)

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
