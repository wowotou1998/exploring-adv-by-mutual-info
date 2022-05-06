import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data_utils

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Fix random seeds for reproducibility
# seed = 73
# torch.manual_seed(seed)
# np.random.seed(seed)
# # Load
# # MNIST
# # Dataset
# # import torchvision
# # from torchvision import transforms
# # from torchvision.datasets import MNIST
# # # 60000 tuples with 1x28x28 image and corresponding label
# # data = MNIST('data',
# #              train=True,
# #              download=True,
# #              transform = transforms.Compose([transforms.ToTensor()]))
# # # Split data into images and labels
# # x_train = data.train_data
# # y_train = data.train_labels
# # # Scale images from [0,255] to [0,+1]
# # x_train = x_train.float() / 255
# # # Save as .npz
# # np.savez_compressed('data/mnist_train',
# #                     a=x_train,
# #                     b=y_train)
#
# # # 10000 tuples with 1x28x28 image and corresponding label
# # data = MNIST('data',
# #              train=False,
# #              download=True,
# #              transform = transforms.Compose([transforms.ToTensor()]))
# # # Split data into images and labels
# # x_test = data.test_data
# # y_test = data.test_labels
# # # Scale images from [0,255] to [0,+1]
# # x_test = x_test.float() / 255
# # # Save as .npz
# # np.savez_compressed('data/mnist_test',
# #                     a=x_test,
# #                     b=y_test)
# # Load MNIST data locally
# train_data = np.load('data/mnist_train.npz')
# x_train = torch.Tensor(train_data['a'])
# y_train = torch.Tensor(train_data['b'])
# n_classes = len(np.unique(y_train))
#
# test_data = np.load('data/mnist_test.npz')
# x_test = torch.Tensor(test_data['a'])
# y_test = torch.Tensor(test_data['b'])
# # Visualise data
# plt.rcParams.update({'font.size': 16})
# fig, axes = plt.subplots(1, 4, figsize=(35, 35))
# imx, imy = (28, 28)
# labels = [0, 1, 2, 3]
# for i, ax in enumerate(axes):
#     visual = np.reshape(x_train[labels[i]], (imx, imy))
#     ax.set_title("Example Data Image, y=" + str(int(y_train[labels[i]])))
#     ax.imshow(visual, vmin=0, vmax=1)
# plt.show()


def get_train_data(data_set_name, batch_size):
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
    if data_set_name == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(root='../DataSet/' + data_set_name, train=True, download=True,
                                                 transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='../DataSet/' + data_set_name, train=False, download=True,
                                                transform=transform_test)
    if data_set_name == 'MNIST':
        train_set = torchvision.datasets.MNIST(root='../DataSet/' + data_set_name, train=True, download=True,
                                               transform=transform_train)
        test_set = torchvision.datasets.MNIST(root='../DataSet/' + data_set_name, train=False, download=True,
                                              transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, )
    return [train_loader, test_loader]


# Models


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
                                     nn.ReLU(inplace=True))
        self.fc_mu = nn.Linear(1024, self.z_dim)
        self.fc_std = nn.Linear(1024, self.z_dim)

        # build decoder
        self.decoder = nn.Linear(self.z_dim, output_shape)

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x) - 5, beta=1)

    def decode(self, z):
        """
        z : [batch_size,z_dim]
        """
        return self.decoder(z)

    def reparameterise(self, mu, std):
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
        z = self.reparameterise(mu, std)
        return self.decode(z), mu, std


# Training
# Hyper-parameters

beta = 1e-3
z_dim = 256
epochs = 200
batch_size = 128
learning_rate = 1e-4
decay_rate = 0.97
n_classes = 10
# Create DatatLoader 
train_loader, test_loader = get_train_data(data_set_name='MNIST', batch_size=batch_size)


# Loss function: Cross Entropy Loss (CE) + beta*KL divergence
def loss_function(y_pred, y, mu, std):
    """    
    y_pred : [batch_size,10]
    y : [batch_size,10]    
    mu : [batch_size,z_dim]  
    std: [batch_size,z_dim] 
    """
    CE = F.cross_entropy(y_pred, y, reduction='sum')
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2 * std.log() - 1)
    return (beta * KL + CE) / y.size(0)


# Initialize Deep VIB
vib = DeepVIB(1 * 28 * 28, n_classes, z_dim)

# Optimiser
optimiser = torch.optim.Adam(vib.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=decay_rate)

# Send to GPU if available
vib.to(device)

print("Device: ", device)
print(vib)

# Training
measures = defaultdict(list)
start_time = time.time()

# put Deep VIB into train mode 
vib.train()

for epoch in range(epochs):
    epoch_start_time = time.time()

    # exponential decay of learning rate every 2 epochs
    if epoch % 2 == 0 and epoch > 0:
        scheduler.step()

    batch_loss = 0
    batch_accuracy = 0
    for _, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        # Zero accumulated gradients
        vib.zero_grad()

        # forward pass through Deep VIB
        y_pred, mu, std = vib(X)

        # Calculate loss
        loss = loss_function(y_pred, y, mu, std)
        # Backpropogation: calculating gradients
        loss.backward()
        # Update parameters of generator
        optimiser.step()

        # Save loss per batch
        batch_loss += loss.item() * X.size(0)
        # Save accuracy per batch
        y_pred = torch.argmax(y_pred, dim=1)
        batch_accuracy += y_pred.eq(y.data).cpu().sum().item()

        # Save losses per epoch
    measures['total_loss'].append(batch_loss / y.shape[0])
    # Save accuracy per epoch
    measures['accuracy'].append(batch_accuracy / y.shape[0])

    print("Epoch: {}/{}...".format(epoch + 1, epochs),
          "Loss: {:.4f}...".format(measures['total_loss'][-1]),
          "Accuracy: {:.4f}...".format(measures['accuracy'][-1]),
          "Time Taken: {:,.4f} seconds".format(time.time() - epoch_start_time))

print("Total Time Taken: {:,.4f} seconds".format(time.time() - start_time))

