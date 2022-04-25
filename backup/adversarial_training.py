import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets, transforms

import torchattacks
from torchattacks import PGD, FGSM

from std_training_mnist import evaluate_accuracy
from ModelSet import FC_Sigmoid

# print("PyTorch", torch.__version__)
# print("Torchvision", torchvision.__version__)
# print("Torchattacks", torchattacks.__version__)
# print("Numpy", np.__version__)
# load data
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize([0.5], [0.5])
     ]
)

train_dataset = datasets.MNIST(root='../DataSet/MNIST', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='../DataSet/MNIST', train=False, transform=data_tf)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=True)

# 2. Define Model
model = FC_Sigmoid(torch.nn.ReLU()).cuda()
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
atk = PGD(model, eps=0.3, alpha=0.1, steps=7)

# 3. Adversarial Training
num_epochs = 5
for epoch in range(num_epochs):
    total_batch = len(train_dataset) // batch_size
    for i, (batch_images, batch_labels) in enumerate(train_loader):
        X = atk(batch_images, batch_labels).cuda()
        Y = batch_labels.cuda()

        pre = model(X)
        cost = loss(pre, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))
    evaluate_accuracy(test_loader, model, device=torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu"))

# 4.Test Model
# 4.1 Standard Accuracy
model.eval()

correct = 0
total = 0

for images, labels in test_loader:
    images = images.cuda()
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))

# 4.2 Robust Accuracy
# model.eval()

correct = 0
total = 0

atk = FGSM(model, eps=0.3)

for images, labels in test_loader:
    images = atk(images, labels).cuda()
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
