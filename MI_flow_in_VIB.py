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
        # self.fc_mu = nn.Linear(1024, self.z_dim)
        # self.fc_std = nn.Linear(1024, self.z_dim)

        # build decoder
        self.decoder = nn.Linear(self.z_dim, output_shape)
        # self.weight_init()

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        encoder_ouput = self.encoder(x)
        mu = encoder_ouput[:, :self.z_dim]
        sigma = F.softplus(encoder_ouput[:, self.z_dim:] - 5, beta=1)
        # return self.fc_mu(x), F.softplus(self.fc_std(x) - 5, beta=1)
        return mu, sigma

    def decode(self, z):
        """
        z : [batch_size,z_dim]
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
        # flattent image
        x_flat = x.view(x.size(0), -1)
        mu, std = self.encode(x_flat)
        z = self.reparameterize(mu, std)
        return self.decode(z), mu, std

    # def xavier_init(self, ms):
    #     for m in self.named_modules():
    #         if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #             nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
    #             m.bias.data.zero_()
    #
    # def weight_init(self):
    #     print('weight_init')
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #             nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    #             m.bias.data.zero_()


class Train_VIB():
    # Device Config
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = 1e-3
        self.z_dim = 256
        self.epochs = 20
        self.batch_size = 128
        self.data_set_name = 'MNIST'
        self.learning_rate = 1e-4
        self.decay_rate = 0.97
        self.n_classes = 10

        self.Model_Name = 'VIB'
        self.Forward_Size = 128
        self.Forward_Repeat = 1

    def get_train_data(self):
        import torchvision.transforms as transforms
        import torchvision
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set, test_set = None, None
        data_set_name = self.data_set_name
        if data_set_name == 'CIFAR10':
            train_set = torchvision.datasets.CIFAR10(root='./DataSet/' + data_set_name, train=True, download=True,
                                                     transform=transform_train)
            test_set = torchvision.datasets.CIFAR10(root='./DataSet/' + data_set_name, train=False, download=True,
                                                    transform=transform_test)
        if data_set_name == 'MNIST':
            train_set = torchvision.datasets.MNIST(root='./DataSet/' + data_set_name, train=True, download=True,
                                                   transform=transform_train)
            test_set = torchvision.datasets.MNIST(root='./DataSet/' + data_set_name, train=False, download=True,
                                                  transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, )
        return [train_loader, test_loader]

    # Models

    # Training
    # Hyper-parameters

    # Loss function: Cross Entropy Loss (CE) + beta*KL divergence
    def loss_function(self, y_pred, y, mu, std):
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
        return izy_lower_bound, izx_upper_bound, CE + self.beta * KL

    def train_vib(self, Enable_Adv_Training=False):

        # Create DatatLoader
        train_loader, test_loader = self.get_train_data()
        # Initialize Deep VIB
        vib = DeepVIB(1 * 28 * 28, self.n_classes, self.z_dim)
        # Optimizer
        optimizer = torch.optim.Adam(vib.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.decay_rate)
        # Send to GPU if available
        vib.to(self.device)
        print("Device: ", self.device)
        print(vib)

        # Training
        from collections import defaultdict
        measures = defaultdict(list)
        start_time = time.time()

        # put Deep VIB into train mode 
        vib.train()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            # exponential decay of learning rate every 2 epochs
            if epoch % 2 == 0 and epoch > 0:
                scheduler.step()

            loss_sum = 0
            acc_N = 0
            sample_N = 0
            izy_lower_bound_total, izx_upper_bound_total = 0., 0.
            for _, (X, y) in enumerate(train_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                # forward pass through Deep VIB
                y_pred, mu, std = vib(X)

                # Calculate loss
                izy_lower_bound, izx_upper_bound, loss = self.loss_function(y_pred, y, mu, std)
                # Backpropagation: calculating gradients
                loss.backward()
                # Update parameters of generator
                optimizer.step()
                # Zero accumulated gradients
                vib.zero_grad()

                # save mutual info per batch
                izy_lower_bound_total += izy_lower_bound
                izx_upper_bound_total += izx_upper_bound
                # Save loss per batch
                loss_sum += loss.item()
                # Save accuracy per batch
                y_pred = torch.argmax(y_pred, dim=1)
                acc_N += y_pred.eq(y.data).cpu().sum().item()
                sample_N += y.size(0)

            # Save average mutual info per epoch
            measures['izy_lower_bound'].append(izy_lower_bound_total / len(train_loader))
            # Save average loss per epoch
            measures['izx_upper_bound'].append(izx_upper_bound_total / len(train_loader))
            # Save accuracy per epoch
            measures['ave_loss'].append(loss_sum / len(train_loader))
            # print(acc_N, sample_N)
            measures['accuracy'].append(acc_N * 100. / sample_N)

            print("Epoch: [%d]/[%d] " % (epoch + 1, self.epochs),
                  "izy/izx: [%.2f]/[%.2f] " % (measures['izy_lower_bound'][-1], measures['izx_upper_bound'][-1]),
                  "Ave Loss: [%.2f] " % (measures['ave_loss'][-1]),
                  "Accuracy: [%.2f%%] " % (measures['accuracy'][-1]),
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


if __name__ == '__main__':
    train_0 = Train_VIB()
    train_0.train_vib()
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
