import matplotlib.pyplot as plt
import numpy
import torch
import torchattacks
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import copy
from torchviz import make_dot, make_dot_from_trace


def show_one_image(images, title):
    plt.figure()
    print(images.shape)
    images = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # print(images.detach().numpy()[0].shape)
    plt.imshow(images)
    plt.title(title)
    plt.show()



def forward_hook(module, data_input, data_output):
    data_flow.append(data_output)


def register_hook(net, hook_function):
    h1 = net.features.init_block.register_forward_hook(hook_function)
    h2 = net.features.stage1.register_forward_hook(hook_function)
    h3 = net.features.stage2.register_forward_hook(hook_function)
    h4 = net.features.stage3.register_forward_hook(hook_function)
    h5 = net.features.final_pool.register_forward_hook(hook_function)
    # net.features.output.register_forward_hook(hook_function)
    return [h1, h2, h3, h4, h5]


def cancel_hook(handles):
    for handle in handles:
        handle.remove()
    # net.features.output.register_forward_hook(hook_function)





def explain(L1, L2, L3, L4):
    norms = generate_norms(L1, L2)
    norms_2 = generate_norms(L3, L4)
    plt.figure()
    legend_set = list()
    # plt.ylim(0.99,1.01)
    plt.subplot(221)
    plt.plot(norms[:, 0])
    plt.plot(norms_2[:, 0])
    plt.legend(['0_norm', 'failed'])

    plt.subplot(222)
    plt.plot(norms[:, 1])
    plt.plot(norms_2[:, 1])
    plt.legend(['1_norm', 'failed'])

    plt.subplot(223)
    plt.plot(norms[:, 2])
    plt.plot(norms_2[:, 2])
    plt.legend(['inf_norm', 'failed'])

    plt.subplot(224)
    plt.plot(norms[:, 3])
    plt.plot(norms_2[:, 3])
    plt.legend(['cosine_similarity', 'failed'])
    plt.show()


net = ptcv_get_model("resnet20_cifar10", pretrained=True)
# print(net)
data_flow = list()
data_flow_original_success = list()
data_flow_attack_success = list()

data_flow_original_no_success = list()
data_flow_attack_no_success = list()

# resnet_20_handle = None


# data_flow_attack_no_succeed = list()
# data_flow_attack_succeed = list()
resnet_20_handle = register_hook(net, hook_function=forward_hook)

# L1, L2, L3, L4 = attack_one_model(net, Epsilon=5, Iterations=10, Momentum=0.9)
attack_one_model(net, Epsilon=5, Iterations=10, Momentum=0.9)

# explain(L1, L2, L3, L4)

# print(net.features.stage1.unit1)
# for layer in net.named_modules():
#     print(layer)
# 使用isinstance可以判断这个模块是不是所需要的类型实例


# summary(net, input_size=(3, 32, 32))
