import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt

import math
from collections import OrderedDict


class VGG_s(nn.Module):
    # implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3,
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        super(VGG_s, self).__init__()
        self.features = nn.Sequential(
            # Stage 1
            #  convolutional layer, input channels 3, output channels 8, filter size 3
            #  max-pooling layer, size 2
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 2
            #  convolutional layer, input channels 8, output channels 16, filter size 3
            #  max-pooling layer, size 2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 3
            #  convolutional layer, input channels 16, output channels 32, filter size 3
            #  convolutional layer, input channels 32, output channels 32, filter size 3
            #  max-pooling layer, size 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 4
            #  convolutional layer, input channels 32, output channels 64, filter size 3
            #  convolutional layer, input channels 64, output channels 64, filter size 3
            #  max-pooling layer, size 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 5
            #  convolutional layer, input channels 64, output channels 64, filter size 3
            #  convolutional layer, input channels 64, output channels 64, filter size 3
            #  max-pooling layer, size 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            #  fully-connected layer (64->64)
            #  fully-connected layer (64->10)
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64)
        x = self.classifier(x)
        return x


def train(trainloader, net, criterion, optimizer, device, epochs=5):
    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        try:
            histogram = np.zeros(np.shape(trainloader.dataset.dataset.classes))
        except AttributeError:
            histogram = np.zeros(np.shape(np.unique(trainloader.dataset.dataset.train_labels)))
        for i, (images, labels) in enumerate(trainloader):
            histogram += np.histogram(labels.numpy(), bins=np.shape(histogram)[0])[0]
            images = images.to(device)
            labels = labels.to(device)
            #  zero the parameter gradients
            #  forward pass
            #  backward pass
            #  optimize the network
            optimizer.zero_grad()
            scores = net.forward(images)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end - start))
                start = time.time()
                running_loss = 0.0
        print(histogram)
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    try:
        class_count = len(testloader.dataset.class_to_idx)
    except AttributeError:
        class_count = np.shape(np.unique(testloader.dataset.test_labels))[0]
    matrix = torch.zeros((class_count, class_count))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                matrix[labels[i], predicted[i]] += 1
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    print(matrix.numpy())
    return matrix.numpy()


def trainset_select(dataset, device, distribution=None):
    ''' select subset of training set by given distribution
        dafault:    all
        int:        equally distributed
        range:      same as input range
        np.array:   counting each class
    '''
    try:
        labels = dataset.targets
    except AttributeError:
        labels = dataset.train_labels

    if distribution is None:
        return range(np.shape(labels)[0])
    elif isinstance(distribution, range):
        return distribution
    elif isinstance(distribution, int):
        distribution *= np.ones(np.shape(np.unique(labels)))
    assert (np.shape(distribution) == np.shape(np.unique(labels)))
    assert (np.all(distribution <=
                   np.histogram(labels, bins=np.shape(distribution)[0])[0]))
    subset = []
    for i in range(np.shape(labels)[0]):
        if (distribution[labels[i]] > 0):
            subset.append(i)
            distribution[labels[i]] -= 1
    return subset


def main(argv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device.type)
    torch.nn.Module.dump_patches = True
    torch.manual_seed(0)
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False)

    if len(argv) == 1:
        net = torch.load('vgg.pt')
        print('load')
        filename = 'vgg'
    elif argv[1] == 'retrain':
        assert (isinstance(argv[2], str) and argv[2].isnumeric())
        if (len(argv) == 3):
            argv.append('5')
        print(argv[1], "subset=" + argv[2], "epoch=" + argv[3])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        trainset = torch.utils.data.Subset(trainset,
                                           trainset_select(trainset, device, distribution=
                                           50 * int(argv[2]) * np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=100, shuffle=True)

        net = VGG_s().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        train(trainloader, net, criterion, optimizer, device, epochs=int(argv[3]));
        torch.save(net, 'vgg.pt')
        filename = 'vgg-epoch={}-subset={}'.format(argv[3], argv[2])
    else:
        raise NameError('invalid mode')

    confusion_mtx = test(testloader, net, device)
    plt.imsave(filename + '.pdf', confusion_mtx)
    return confusion_mtx


if __name__ == "__main__":
    import sys

    # assert (sys.argv[0][-6:] == 'vgg.py')
    # main(sys.argv)
    vgg = VGG_s()
    x = torch.rand((2, 3, 32, 32))
    y = torch.nn.Softmax(vgg(x))
    print(y)
