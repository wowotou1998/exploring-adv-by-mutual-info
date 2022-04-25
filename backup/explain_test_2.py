import matplotlib.pyplot as plt
import numpy
import torch
import torchattacks
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import copy


def show_one_image(images, title):
    plt.figure()
    print(images.shape)
    images = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # print(images.detach().numpy()[0].shape)
    plt.imshow(images)
    plt.title(title)
    plt.show()


def attack_one_model(model, Epsilon, Iterations, Momentum):
    data_tf = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=False, transform=data_tf)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    test_count = 0
    sample_attacked = 0

    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    # every epoch has 64 images ,every images has 1 channel and the channel size is 28*28
    pbar = tqdm(total=10000)
    model.to(device)
    model.eval()

    epsilon = 5 / 255.

    for data in test_loader:
        test_count += 1
        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
        # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # 选择预测正确的images和labels，
        # 剔除预测不正确的images和labels
        _, predict = torch.max(model(images), 1)
        predict_answer = (labels == predict)
        # print('predict_answer', predict_answer)
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        selection_index = torch.flatten(torch.nonzero(predict_answer))
        # print('selection_index', selection_index)
        images = torch.index_select(images, 0, selection_index)
        labels = torch.index_select(labels, 0, selection_index)

        # 产生对抗样本
        atk = torchattacks.FGSM(model, eps=epsilon)
        images_adv = atk(images, labels)

        # 提取攻击成功的对抗样本
        _, predict = torch.max(model(images_adv), 1)
        predict_answer = (labels != predict)
        # print('predict_answer', predict_answer)
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        selection_index = torch.flatten(torch.nonzero(predict_answer))
        # print('selection_index', selection_index)
        images_adv_success = torch.index_select(images_adv, 0, selection_index)

        # 提取攻击成功的对抗样本的数据流 以及 与之对应的原始样本的数据流
        data_flow.clear()
        _ = model(images_adv_success)
        data_flow_attack_success = data_flow.copy()

        # 提取正常图片的数据流
        images_success = torch.index_select(images, 0, selection_index)
        data_flow.clear()
        _ = model(images_success)
        data_flow_original_success = data_flow.copy()

        # 提取攻击不成功的对抗样本
        _, predict = torch.max(model(images_adv), 1)
        predict_answer = (labels == predict)
        # print('predict_answer', predict_answer)
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        selection_index = torch.flatten(torch.nonzero(predict_answer))
        # print('selection_index', selection_index)
        images_adv_no_success = torch.index_select(images_adv, 0, selection_index)

        # 提取攻击不成功的对抗样本的数据流 以及 与之对应的原始样本的数据流
        data_flow.clear()
        _ = model(images_adv_no_success)
        data_flow_attack_no_success = data_flow.copy()

        # 提取正常图片的数据流
        images_no_success = torch.index_select(images, 0, selection_index)
        data_flow.clear()
        _ = model(images_no_success)
        data_flow_original_no_success = data_flow.copy()

        if test_count == 1:
            return data_flow_attack_success, data_flow_original_success, data_flow_attack_no_success, data_flow_original_no_success

        sample_attacked += labels.shape[0]
        pbar.update(labels.shape[0])
        # to quickly test
        # break

    pbar.close()


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


resnet_20_handle = register_hook(net, hook_function=forward_hook)

L1, L2, L3, L4 = attack_one_model(net, Epsilon=5, Iterations=10, Momentum=0.9)


def generate_norms(L1, L2):
    B, C, H, W = L1[0].shape
    norms = numpy.zeros(shape=(len(L1), 4))
    # perturbation = torch.abs(images - previous_images).view(B, -1)
    for i in range(len(L1)):
        norms[i][0] = torch.sum(L1[i] == L2[i]).item()

        perturbation = torch.abs(L1[i] - L2[i]).view(B, -1)
        norms[i][1] = torch.mean(perturbation).item()

        # calculate norm_inf
        value, indices = torch.max(perturbation, dim=1)
        norms[i][2] = torch.mean(value).item()
        # calculate similarity
        # 注意，这里的余弦相似度极有可能会数值溢出，因此一定要做好数值截断处理，将数据控制在合理的范围之内

        cosine_similarity = torch.cosine_similarity(x1=L1[i].view(B, -1) * 100,
                                                    x2=L2[i].view(B, -1) * 100,
                                                    dim=1,
                                                    eps=1e-11)
        cosine_similarity = torch.clamp(cosine_similarity, min=-1., max=1.)
        norms[i][3] = torch.mean(cosine_similarity).item()

    return norms


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


explain(L1, L2, L3, L4)

# print(net.features.stage1.unit1)
# for layer in net.named_modules():
#     print(layer)
# 使用isinstance可以判断这个模块是不是所需要的类型实例


# summary(net, input_size=(3, 32, 32))
