import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data_utils

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
from torchattacks.attack import Attack


# Loss function: Cross Entropy Loss (CE) + beta*KL divergence
def vib_loss_function(y_pred, y, mu, std, beta):
    """
    y_pred : [batch_size,10]
    y : [batch_size,10]
    mu : [batch_size,z_dim]
    std: [batch_size,z_dim]
    """
    import math
    nats2bits = 1.0 / math.log(2)
    Batch_N = y.size(0) * 1.

    # 交叉熵损失 = -I(Z;Y)_lower_bound
    CE = F.cross_entropy(y_pred, y, reduction='mean')
    izy_lower_bound = math.log(10, 2) - CE * nats2bits

    # KL信息损失 = I(Z;X)_upper_bound
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2 * std.log() - 1) / Batch_N
    izx_upper_bound = KL * nats2bits
    # 先相加后取平均
    return izy_lower_bound, izx_upper_bound, CE + beta * KL


class PGD_VIB(Attack):
    def __init__(self, model, eps=0.3,
                 alpha=2 / 255, steps=40, vib_beta=1e-3, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.vib_beta = vib_beta

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self._targeted:
        #     target_labels = self._get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            # outputs = self.model(adv_images)
            outputs, mu, std = self.model(adv_images)

            # Calculate loss
            izy_lower_bound_i, izx_upper_bound_i, loss_i = vib_loss_function(outputs, labels,
                                                                             mu, std, self.vib_beta)
            if self._targeted:
                # cost = -loss(outputs, target_labels)
                cost = -loss_i

            else:
                # cost = loss(outputs, labels)
                cost = loss_i
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class DeepVIB(nn.Module):
    def __init__(self, input_shape, output_shape, z_dim):
        """
        Deep VIB Model.

        Arguments:
        ----------
        input_shape : `int`
            Flattened size of image. (Default=784)
        output_shape : `int`
            Number of classes. (Default=10)            
        z_dim : `int`
            The dimension of the latent variable z. (Default=256)
        """
        super(DeepVIB, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.z_dim = z_dim

        # build encoder
        self.encoder = nn.Sequential(nn.Linear(input_shape, 1024),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(1024, 2 * self.z_dim))

        # build decoder
        self.decoder = nn.Linear(self.z_dim, output_shape)

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        encoder_output = self.encoder(x)
        mu = encoder_output[:, 0:self.z_dim]
        sigma = F.softplus(encoder_output[:, self.z_dim:] - 5, beta=1)
        # return self.fc_mu(x), F.softplus(self.fc_std(x) - 5, beta=1)
        return mu, sigma

    def decode(self, z):
        """
        z : [batch_size, z_dim]
        """
        return self.decoder(z)

    def reparameterize(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """
        Forward pass 

        Parameters:
        -----------
        x : [batch_size,28,28]
        """
        # flatten image
        x_flat = x.view(x.size(0), -1)
        mu, std = self.encode(x_flat)
        z = self.reparameterize(mu, std)
        return self.decode(z), mu, std


class Train_VIB(object):
    # Device Config
    def __init__(self, ):
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.beta = 1e-3
        self.z_dim = 256
        self.epochs = 100
        self.batch_size = 128
        self.data_set_name = 'MNIST'
        self.learning_rate = 1e-4
        self.decay_rate = 0.97
        self.n_classes = 10
        self.Model_Name = 'VIB'
        self.Forward_Size = 1000
        self.Forward_Repeat = 5

        self.Model = DeepVIB(1 * 28 * 28, self.n_classes, self.z_dim)

        self.Train_loader, self.Test_Loader = self.get_train_test_data()

    def train_attack(self, Model, Random_Start=False):
        atk = PGD_VIB(Model, eps=45 / 255, alpha=9 / 255, steps=7, vib_beta=1e-3, random_start=Random_Start)
        return atk

    def test_attack(self, Model, Random_Start=False):
        atk = PGD_VIB(Model, eps=45 / 255, alpha=9 / 255, steps=7, vib_beta=1e-3, random_start=Random_Start)
        return atk

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

    def get_train_test_data(self):
        import torchvision.transforms as transforms
        import torchvision
        transform_train_cifar10 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_train_mnist = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set, test_set = None, None
        data_set_name = self.data_set_name
        if data_set_name == 'CIFAR10':
            train_set = torchvision.datasets.CIFAR10(root='./DataSet/' + data_set_name, train=True, download=True,
                                                     transform=transform_train_cifar10)
            test_set = torchvision.datasets.CIFAR10(root='./DataSet/' + data_set_name, train=False, download=True,
                                                    transform=transform_test)
        if data_set_name == 'MNIST':
            train_set = torchvision.datasets.MNIST(root='./DataSet/' + data_set_name, train=True, download=True,
                                                   transform=transform_train_mnist)
            test_set = torchvision.datasets.MNIST(root='./DataSet/' + data_set_name, train=False, download=True,
                                                  transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, )
        return [train_loader, test_loader]

    @torch.no_grad()
    def calculate_acc_and_mutual_info(self, Model, Keep_Clean):
        # 这里的epoch_i没必要指定，因为epochi就是列表当中的最后一个元素
        # a = list[-1]就是最后一个元素
        # TODO 对抗样本的生成的loss函数需要重写。
        # defaultdict 它的其他功能与dict相同，但会为一个不存在的键提供默认值

        Model.eval()

        correct_N = 0
        total_N = 0
        loss = 0.
        izy_lower_bound, izx_upper_bound = 0., 0.
        measures = defaultdict(float)

        for i in range(self.Forward_Repeat):
            images, labels = self.get_clean_or_adv_image(Model, Keep_Clean)
            # 模型输出 logits, mu, std
            outputs, mu, std = Model(images)
            # 使用模型的三个输出来计算互信息的上下界, 还有loss
            izy_lower_bound_i, izx_upper_bound_i, loss_i = vib_loss_function(outputs, labels, mu, std, self.beta)

            predicted_prob, predicted = torch.max(outputs, dim=1)
            correct_N += (predicted == labels).sum().item()
            total_N += labels.size(0)
            loss += loss_i.item()
            izy_lower_bound += izy_lower_bound_i.item()
            izx_upper_bound += izx_upper_bound_i.item()

        measures['izy'] = izy_lower_bound / self.Forward_Repeat
        # Save average loss per epoch
        measures['izx'] = izx_upper_bound / self.Forward_Repeat
        # Save accuracy per epoch
        measures['loss'] = loss / self.Forward_Repeat
        # print(acc_N, sample_N)
        measures['acc'] = correct_N * 100. / total_N
        return measures

    def train_vib(self, Enable_Adv_Training=False):
        # Initialize Deep VIB

        # vib = DeepVIB(3 * 32 * 32, self.n_classes, self.z_dim)
        # Send to GPU if available
        # put Deep VIB into train mode
        import copy
        vib = copy.deepcopy(self.Model)
        vib.train()
        vib.to(self.Device)
        print("Device: ", self.Device)
        print(vib)

        # Optimizer
        optimizer = torch.optim.Adam(vib.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.decay_rate)

        # Training
        from collections import defaultdict
        measures = defaultdict(list)
        start_time = time.time()

        for epoch in range(self.epochs):
            # TODO: 对抗训练和普通训练
            # TODO 在对抗样本和正常样本上的互信息.
            mi_loss_acc_i_std = self.calculate_acc_and_mutual_info(vib, Keep_Clean=True)
            measures['izy_test_std'].append(mi_loss_acc_i_std['izy'])
            measures['izx_test_std'].append(mi_loss_acc_i_std['izx'])
            measures['loss_test_std'].append(mi_loss_acc_i_std['loss'])
            measures['acc_test_std'].append(mi_loss_acc_i_std['acc'])

            mi_loss_acc_i_adv = self.calculate_acc_and_mutual_info(vib, Keep_Clean=False)
            measures['izy_test_adv'].append(mi_loss_acc_i_adv['izy'])
            measures['izx_test_adv'].append(mi_loss_acc_i_adv['izx'])
            measures['loss_test_adv'].append(mi_loss_acc_i_adv['loss'])
            measures['acc_test_adv'].append(mi_loss_acc_i_adv['acc'])
            epoch_start_time = time.time()

            # exponential decay of learning rate every 2 epochs
            if epoch % 2 == 0 and epoch > 0:
                scheduler.step()

            loss_sum = 0
            acc_N = 0
            sample_N = 0
            izy_lower_bound_total, izx_upper_bound_total = 0., 0.
            for _, (X, y) in enumerate(self.Train_loader):
                X = X.to(self.Device)
                y = y.to(self.Device)

                if Enable_Adv_Training:
                    atk = self.train_attack(Model=vib, Random_Start=True)
                    X = atk(X, y)

                # forward pass through Deep VIB
                y_pred, mu, std = vib(X)

                # Calculate loss
                izy_lower_bound, izx_upper_bound, loss = vib_loss_function(y_pred, y, mu, std, self.beta)
                # Backpropagation: calculating gradients
                loss.backward()
                # Update parameters of generator
                optimizer.step()
                # Zero accumulated gradients
                vib.zero_grad()

                # save mutual info per batch
                izy_lower_bound_total += izy_lower_bound.item()
                izx_upper_bound_total += izx_upper_bound.item()
                # Save loss per batch
                loss_sum += loss.item()
                # Save accuracy per batch
                y_pred = torch.argmax(y_pred, dim=1)
                acc_N += y_pred.eq(y.data).cpu().sum().item()
                sample_N += y.size(0)
            # 在验证集上检验acc, loss,MI

            # Save average mutual info per epoch
            measures['izy_train'].append(izy_lower_bound_total / len(self.Train_loader))
            # Save average loss per epoch
            measures['izx_train'].append(izx_upper_bound_total / len(self.Train_loader))
            # Save accuracy per epoch
            measures['loss_train'].append(loss_sum / len(self.Train_loader))
            # print(acc_N, sample_N)
            measures['acc_train'].append(acc_N * 100. / sample_N)

            print("Epoch: [%d]/[%d] " % (epoch + 1, self.epochs),
                  "izy/izx: [%.2f]/[%.2f] " % (measures['izy_train'][-1], measures['izx_train'][-1]),
                  "Ave Loss: [%.2f] " % (measures['loss_train'][-1]),
                  "Accuracy: [%.2f%%] " % (measures['acc_train'][-1]),
                  "Time Taken: [%.2f] seconds " % (time.time() - epoch_start_time))

        print("Total Time Taken: [%.2f] seconds" % (time.time() - start_time))

        Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'
        Model_Name, Forward_Size, Forward_Repeat = self.Model_Name, self.Forward_Size, self.Forward_Repeat
        dir = 'Checkpoint/%s' % (Model_Name)
        # 对于每一个模型产生的数据, 使用一个文件夹单独存放
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open('./Checkpoint/%s/mi_loss_acc_%s.pkl' % (Model_Name, Is_Adv_Training), 'wb') as f:
            pickle.dump(measures, f)
        # Save checkpoint.
        file_name = "./Checkpoint/%s/%s_%s.pth" % (
            self.Model_Name,
            self.Model_Name, 'adv' if Enable_Adv_Training else 'std')
        save_model(vib, file_name)

    def plot_data(self, Enable_Adv_Training=False):
        import matplotlib.pyplot as plt
        import numpy as np
        from pylab import mpl
        import datetime
        import pickle
        from matplotlib.lines import Line2D
        import math

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

        Is_Adv_Training = 'Adv_Train' if Enable_Adv_Training else 'Std_Train'

        with open('./Checkpoint/%s/mi_loss_acc_%s.pkl' % (self.Model_Name, Is_Adv_Training), 'rb') as f:
            measures = pickle.load(f)

        Std_Epoch_Num = len(measures['acc_train'])
        Epochs = [i for i in range(Std_Epoch_Num)]

        # sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))
        sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=Std_Epoch_Num))

        title = "%s,Lower,Clean(Adv),Sample_N(%d),%s" % (
            self.Model_Name, Forward_Repeat * Forward_Size, Is_Adv_Training
        )

        # fig size, 先列后行
        nrows = 1
        ncols = 4
        # A4 纸大小 8.27 * 11.69
        fig, axs = plt.subplots(nrows, ncols, figsize=(8, 2), squeeze=False)

        # 初始化 xlabel, y_label
        for i in range(nrows - 1):
            for j in range(ncols):
                axs[i][j].grid(True)
                axs[i][j].set_xlabel('layers')

        # plt.scatter(I_TX, I_TY,
        #             color=c,
        #             linestyle='-', linewidth=0.1,
        #             zorder=2
        #             )
        # -------------------------------------------Loss and Accuracy Detail---------------------
        # for idx, (k, v) in enumerate(analytic_data.items()):
        axs[nrows - 1][0].set_xlabel('epochs')
        axs[nrows - 1][0].set_title('loss')
        axs[nrows - 1][0].plot(Epochs, measures['loss_train'], label='loss_train')
        axs[nrows - 1][0].plot(Epochs, measures['loss_test_std'], label='loss_test_std')
        axs[nrows - 1][0].plot(Epochs, measures['loss_test_adv'], label='loss_test_adv')
        axs[nrows - 1][0].legend()
        # -------------------
        axs[nrows - 1][1].set_xlabel('epochs')
        axs[nrows - 1][1].set_title('acc')
        axs[nrows - 1][1].plot(Epochs, measures['acc_train'], label='acc_train')
        axs[nrows - 1][1].plot(Epochs, measures['acc_test_std'], label='acc_test_std')
        axs[nrows - 1][1].plot(Epochs, measures['acc_test_adv'], label='acc_test_adv')
        axs[nrows - 1][1].legend()

        axs[nrows - 1][2].set_xlabel('epochs')
        axs[nrows - 1][2].set_title('izy')
        axs[nrows - 1][2].plot(Epochs, measures['izy_train'], label='izy_train')
        axs[nrows - 1][2].plot(Epochs, measures['izy_test_std'], label='izy_test_std')
        axs[nrows - 1][2].plot(Epochs, measures['izy_test_adv'], label='izy_test_adv')
        axs[nrows - 1][2].legend()

        axs[nrows - 1][3].set_xlabel('epochs')
        axs[nrows - 1][3].set_title('izx')
        axs[nrows - 1][3].plot(Epochs, measures['izx_train'], label='izx_train')
        axs[nrows - 1][3].plot(Epochs, measures['izx_test_std'], label='izx_test_std')
        axs[nrows - 1][3].plot(Epochs, measures['izx_test_adv'], label='izx_test_adv')
        axs[nrows - 1][3].legend()

        # plt.scatter(epoch_MI_hM_X_upper[0], epoch_MI_hM_Y_upper[0])
        # plt.legend()

        fig.suptitle(title)
        # fig.colorbar(sm, ax=axs, label='Epoch')

        # fig = plt.gcf()
        # if Enable_Show:
        plt.show()
        fig.savefig('mutual_info_%s_%s_%s.pdf' % (
            self.Model_Name, Is_Adv_Training,
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))


if __name__ == '__main__':
    train_0 = Train_VIB()
    # train_0.train_vib(Enable_Adv_Training=False)
    # train_0.train_vib(Enable_Adv_Training=True)
    train_0.plot_data(Enable_Adv_Training=False)
    train_0.plot_data(Enable_Adv_Training=True)
    # import torch
    #
    # from torch import nn
    # from torch import optim
    #
    # from torchvision.datasets import MNIST
    #
    # from torch.utils.data import TensorDataset, Dataset, DataLoader
    #
    # import numpy as np
    # import tqdm
    #
    # beta = 1e-3
    # batch_size = 100
    # samples_amount = 10
    # num_epochs = 10
    #
    # train_data = MNIST('mnist', download=True, train=True)
    # train_dataset = TensorDataset(train_data.train_data.view(-1, 28 * 28).float() / 255, train_data.train_labels)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)
    #
    # test_data = MNIST('mnist', download=True, train=False)
    # test_dataset = TensorDataset(test_data.test_data.view(-1, 28 * 28).float() / 255, test_data.test_labels)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    #
    #
    # def KL_between_normals(q_distr, p_distr):
    #     mu_q, sigma_q = q_distr
    #     mu_p, sigma_p = p_distr
    #     k = mu_q.size(1)
    #
    #     mu_diff = mu_p - mu_q
    #     mu_diff_sq = torch.mul(mu_diff, mu_diff)
    #     logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    #     logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)
    #
    #     fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2),
    #                                                                              dim=1)
    #     two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    #     return two_kl * 0.5
    #
    #
    # class VIB(nn.Module):
    #     def __init__(self, X_dim, y_dim, dimZ=256, beta=1e-3, num_samples=10):
    #         # the dimension of Z
    #         super().__init__()
    #
    #         self.beta = beta
    #         self.dimZ = dimZ
    #         self.num_samples = num_samples
    #
    #         self.encoder = nn.Sequential(nn.Linear(in_features=X_dim, out_features=1024),
    #                                      nn.ReLU(),
    #                                      nn.Linear(in_features=1024, out_features=1024),
    #                                      nn.ReLU(),
    #                                      nn.Linear(in_features=1024, out_features=2 * self.dimZ))
    #
    #         #   try heads
    #         #         self.encoder_sigma_head = nn.Linear()
    #         #         self.encoder_mu_head = ...
    #
    #         # decoder a simple logistic regression as in the paper
    #         self.decoder_logits = nn.Linear(in_features=self.dimZ, out_features=y_dim)
    #
    #     def gaussian_noise(self, num_samples, K):
    #         # works with integers as well as tuples
    #         return torch.normal(torch.zeros(*num_samples, K), torch.ones(*num_samples, K)).cuda()
    #
    #     def sample_prior_Z(self, num_samples):
    #         return self.gaussian_noise(num_samples=num_samples, K=self.dimZ)
    #
    #     def encoder_result(self, batch):
    #         encoder_output = self.encoder(batch)
    #
    #         mu = encoder_output[:, :self.dimZ]
    #         sigma = torch.nn.functional.softplus(encoder_output[:, self.dimZ:])
    #
    #         return mu, sigma
    #
    #     def sample_encoder_Z(self, num_samples, batch):
    #         batch_size = batch.size()[0]
    #         mu, sigma = self.encoder_result(batch)
    #
    #         return mu + sigma * self.gaussian_noise(num_samples=(num_samples, batch_size), K=self.dimZ)
    #
    #     def forward(self, batch_x):
    #         batch_size = batch_x.size()[0]
    #
    #         # sample from encoder
    #         encoder_Z_distr = self.encoder_result(batch_x)
    #         to_decoder = self.sample_encoder_Z(num_samples=self.num_samples, batch=batch_x)
    #
    #         decoder_logits_mean = torch.mean(self.decoder_logits(to_decoder), dim=0)
    #
    #         return decoder_logits_mean
    #
    #     def batch_loss(self, num_samples, batch_x, batch_y):
    #         batch_size = batch_x.size()[0]
    #
    #         prior_Z_distr = torch.zeros(batch_size, self.dimZ).cuda(), torch.ones(batch_size, self.dimZ).cuda()
    #         encoder_Z_distr = self.encoder_result(batch_x)
    #
    #         I_ZX_bound = torch.mean(KL_between_normals(encoder_Z_distr, prior_Z_distr))
    #
    #         to_decoder = self.sample_encoder_Z(num_samples=self.num_samples, batch=batch_x)
    #
    #         decoder_logits = self.decoder_logits(to_decoder)
    #         # batch should go first
    #         decoder_logits = decoder_logits.permute(1, 2, 0)
    #
    #         loss = nn.CrossEntropyLoss(reduce=False)
    #         cross_entropy_loss = loss(decoder_logits, batch_y[:, None].expand(-1, num_samples))
    #
    #         # estimate E_{eps in N(0, 1)} [log q(y | z)]
    #         cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)
    #
    #         minusI_ZY_bound = torch.mean(cross_entropy_loss_montecarlo, dim=0)
    #
    #         return torch.mean(minusI_ZY_bound + self.beta * I_ZX_bound), -minusI_ZY_bound, I_ZX_bound
    #
    #
    # # beta = 1e-3
    # batch_size = 100
    # # samples_amount = 30
    # # num_epochs = 200
    #
    # model = VIB(X_dim=784, y_dim=10, beta=beta, num_samples=samples_amount).cuda()
    #
    # opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    #
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.97)
    #
    #
    # class EMA(nn.Module):
    #     def __init__(self, mu):
    #         super(EMA, self).__init__()
    #         self.mu = mu
    #         self.shadow = {}
    #
    #     def register(self, name, val):
    #         self.shadow[name] = val.clone()
    #
    #     def forward(self, name, x):
    #         assert name in self.shadow
    #         new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
    #         self.shadow[name] = new_average.clone()
    #         return new_average
    #
    #
    # ema = EMA(0.999)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         ema.register(name, param.data)
    # import time
    #
    # seed = time.strftime("%Y-%m-%d %H:%M")
    # # from tensorboardX import SummaryWriter
    #
    # # writer = SummaryWriter(log_dir="tensor_logs/" + seed)
    #
    # for epoch in range(num_epochs):
    #     loss_by_epoch = []
    #     accuracy_by_epoch_train = []
    #     I_ZX_bound_by_epoch = []
    #     I_ZY_bound_by_epoch = []
    #
    #     loss_by_epoch_test = []
    #     accuracy_by_epoch_test = []
    #     I_ZX_bound_by_epoch_test = []
    #     I_ZY_bound_by_epoch_test = []
    #
    #     if epoch % 2 == 0 and epoch > 0:
    #         scheduler.step()
    #
    #     for x_batch, y_batch in tqdm.tqdm(train_loader):
    #         x_batch = x_batch.cuda()
    #         y_batch = y_batch.cuda()
    #
    #         loss, I_ZY_bound, I_ZX_bound = model.batch_loss(samples_amount, x_batch, y_batch)
    #
    #         logits = model.forward(x_batch)
    #         prediction = torch.max(logits, dim=1)[1]
    #         accuracy = torch.mean((prediction == y_batch).float())
    #
    #         loss.backward()
    #         opt.step()
    #         opt.zero_grad()
    #
    #         # compute exponential moving average
    #         # for name, param in model.named_parameters():
    #         #     if param.requires_grad:
    #         #         ema(name, param.data)
    #
    #         I_ZX_bound_by_epoch.append(I_ZX_bound.item())
    #         I_ZY_bound_by_epoch.append(I_ZY_bound.item())
    #
    #         loss_by_epoch.append(loss.item())
    #         accuracy_by_epoch_train.append(accuracy.item())
    #
    #     for x_batch, y_batch in tqdm.tqdm(test_loader):
    #         x_batch = x_batch.cuda()
    #         y_batch = y_batch.cuda()
    #
    #         loss, I_ZY_bound, I_ZX_bound = model.batch_loss(samples_amount, x_batch, y_batch)
    #
    #         logits = model.forward(x_batch)
    #         prediction = torch.max(logits, dim=1)[1]
    #         accuracy = torch.mean((prediction == y_batch).float())
    #
    #         I_ZX_bound_by_epoch_test.append(I_ZX_bound.item())
    #         I_ZY_bound_by_epoch_test.append(I_ZY_bound.item())
    #
    #         loss_by_epoch_test.append(loss.item())
    #         accuracy_by_epoch_test.append(accuracy.item())
    #
    #     # writer.add_scalar("accuracy", np.mean(accuracy_by_epoch_test), global_step=epoch)
    #     # writer.add_scalar("loss", np.mean(loss_by_epoch_test), global_step=epoch)
    #     # writer.add_scalar("I_ZX", np.mean(I_ZX_bound_by_epoch_test), global_step=epoch)
    #     # writer.add_scalar("I_ZY", np.mean(I_ZY_bound_by_epoch_test), global_step=epoch)
    #
    #     print('epoch', epoch, 'loss', np.mean(loss_by_epoch_test),
    #           'prediction_train', np.mean(accuracy_by_epoch_train),
    #           'prediction_test', np.mean(accuracy_by_epoch_test))
    #
    #     print('I_ZX_bound', np.mean(I_ZX_bound_by_epoch_test),
    #           'I_ZY_bound', np.mean(I_ZY_bound_by_epoch_test))

    # from torch.autograd import Variable
'''
# Fix random seeds for reproducibility
seed = 73
torch.manual_seed(seed)
np.random.seed(seed)
# Load
# MNIST
# Dataset
# import torchvision
# from torchvision import transforms
# from torchvision.datasets import MNIST
# # 60000 tuples with 1x28x28 image and corresponding label
# data = MNIST('data',
#              train=True,
#              download=True,
#              transform = transforms.Compose([transforms.ToTensor()]))
# # Split data into images and labels
# x_train = data.train_data
# y_train = data.train_labels
# # Scale images from [0,255] to [0,+1]
# x_train = x_train.float() / 255
# # Save as .npz
# np.savez_compressed('data/mnist_train',
#                     a=x_train,
#                     b=y_train)

# # 10000 tuples with 1x28x28 image and corresponding label
# data = MNIST('data',
#              train=False,
#              download=True,
#              transform = transforms.Compose([transforms.ToTensor()]))
# # Split data into images and labels
# x_test = data.test_data
# y_test = data.test_labels
# # Scale images from [0,255] to [0,+1]
# x_test = x_test.float() / 255
# # Save as .npz
# np.savez_compressed('data/mnist_test',
#                     a=x_test,
#                     b=y_test)
# Load MNIST data locally
train_data = np.load('data/mnist_train.npz')
x_train = torch.Tensor(train_data['a'])
y_train = torch.Tensor(train_data['b'])
n_classes = len(np.unique(y_train))

test_data = np.load('data/mnist_test.npz')
x_test = torch.Tensor(test_data['a'])
y_test = torch.Tensor(test_data['b'])
# Visualise data
plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 4, figsize=(35, 35))
imx, imy = (28, 28)
labels = [0, 1, 2, 3]
for i, ax in enumerate(axes):
    visual = np.reshape(x_train[labels[i]], (imx, imy))
    ax.set_title("Example Data Image, y=" + str(int(y_train[labels[i]])))
    ax.imshow(visual, vmin=0, vmax=1)
plt.show()
'''
