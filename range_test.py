for i in range(1, 10, 2):
    print(i)

import matplotlib.pyplot as plt
import datetime


def plot_performance(data, Enable_Adv_Training):
    # show two accuracy rate at the same figure
    # 想要绘制线条的画需要记号中带有‘-’
    fig, axs = plt.subplots(ncols=1, nrows=4,)
    # fig, axs = plt.subplots(1, 4, figsize=(9, 3), )
    # print(axs[0].set_xlabel('layers'))
    idx = 0
    for k, v in data.items():
        k, v = (k, v)
        # axs[idx].set_xlabel('epoch')
        # axs[idx].set_ylabel(str(k))
        axs[0].plot(v, linestyle='-', linewidth=1)
        idx += 1
    title = 'Adv Training' if Enable_Adv_Training else 'Std Training'
    fig.suptitle(title)
    fig = plt.gcf()
    plt.show()
    # fig.savefig('/%s.jpg' % ("fig_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
    fig.savefig('./results_pdf/performance_%s.pdf' % (
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
                )


# plot_performance({'a': [1], 'b': [1], 'c': [1], 'd': [1]}, True)
# import matplotlib.pyplot as plt
#
# data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}
# names = list(data.keys())
# values = list(data.values())
#
# fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
# axs[0].bar(names, values)
# axs[1].scatter(names, values)
# axs[2].plot(names, values)
# fig.suptitle('Categorical Plotting')
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize([0.5], [0.5])
     ]
)
train_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=True, transform=data_tf, download=True)
test_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=False, transform=data_tf)
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
img, label = next(iter(test_loader))
print(img.size())
print(label.size())