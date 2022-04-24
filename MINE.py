import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, x_D, y_D, hidden_D):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_D, hidden_D)
        self.fc2 = nn.Linear(y_D, hidden_D)
        self.fc3 = nn.Linear(hidden_D, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2


@torch.enable_grad()
def calculate_MI_MINE(x_sample, y_sample, hidden_D=50, n_epoch=100, lr=0.1):
    x_sample = x_sample.view(x_sample.size(0), -1).float()
    y_sample = y_sample.view(y_sample.size(0), -1).float()
    x_sample.requires_grad = True  # 开启梯度
    y_sample.requires_grad = True  # 开启梯度
    model = Net(x_sample.size()[1], y_sample.size()[1], hidden_D)
    model.to(x_sample.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    plot_loss = []
    estimation = 0.
    for epoch in tqdm(range(n_epoch)):
        y_shuffle = y_sample[torch.randperm(y_sample.size()[0])]
        pred_xy = model(x_sample, y_sample)
        pred_x_y = model(x_sample, y_shuffle)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = - ret  # maximize
        # plot_loss.append(loss.data.numpy())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        estimation = loss.item()
    return estimation
