import torch
import numpy as np
import torch.nn.functional as F


def caculate_MI(layer_activations, Y):
    MI_hM_X_upper = []
    MI_hM_Y_upper = []
    label_num = 10
    # print("layer_activations len", len(layer_activations))

    nats2bits = 1.0 / np.log(2)
    Y_one_hot = F.one_hot(Y, num_classes=10).float().to(Y.device)
    Y_probs = torch.mean(Y_one_hot, dim=0)
    Y_i_idx = []
    for i in range(label_num):
        # 获取标签Y中等于label_i的下标集合
        Y_equal_label_i_index = torch.flatten(torch.nonzero(Y == i))
        Y_i_idx.append(Y_equal_label_i_index)

    noise_variance = 0.1
    for layer_i in range(len(layer_names)):
        # 最后一层输出\hat{y}也可以直接使用KDE来计算互信息,
        # 因为\hat{y}仅仅只是预测值,不是真实的标签y, 自然也可以当成隐藏层来计算互信息

        # 计算I(T;X) upper
        hM_upper = entropy_estimator_kl_simple(layer_activations[layer_i], noise_variance)
        hM_given_X = kde_multivariate_gauss_entropy(layer_activations[layer_i], noise_variance)

        MI_hM_X_upper.append(nats2bits * (hM_upper - hM_given_X))

        # 计算I(T;Y) upper
        hM_given_Y_upper = 0.
        for y_i in range(label_num):
            """
            依次选择激活层i中有关于标签j的激活值， 并计算这部分激活值的的互信息
            """
            activation_i_for_Y_i = torch.index_select(layer_activations[layer_i], dim=0, index=Y_i_idx[y_i])
            hM_given_Y_i_upper = entropy_estimator_kl_simple(activation_i_for_Y_i, noise_variance)
            hM_given_Y_upper += Y_probs[y_i].item() * hM_given_Y_i_upper

        MI_hM_Y_upper.append(nats2bits * (hM_upper - hM_given_Y_upper))

    return MI_hM_X_upper, MI_hM_Y_upper
