import torch
from torch import nn, Tensor
from pytorch_kde import *
import torch.nn.functional as F


class MI_estimator(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # 初始化时,会自动将模型的每一层加上hook函数
        self.model = model
        self.MI_hM_X_upper = []
        self.MI_hM_Y_upper = []
        self.activation_list = []
        self.layer_name = []

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            # layer.__name__ = name
            self.layer_name.append(name)
            layer.register_forward_hook(
                lambda layer_i, input, output:
                self.activation_list.append(output.view(output.size(0), -1))
            )

    def caculate_MI(self, Y):
        nats2bits = 1.0 / np.log(2)
        Y_one_hot = F.one_hot(Y, num_classes=10).float().to(Y.device)
        Y_probs = torch.mean(Y_one_hot, dim=0)
        Y_i_idx = []
        for i in range(10):
            # 获取标签Y中等于label_i的下标集合
            Y_equal_label_i_index = torch.flatten(torch.nonzero(Y == i))
            Y_i_idx.append(Y_equal_label_i_index)

        noise_variance = 0.1
        for activation_i in self.activation_list:
            # 最后一层输出\hat{y}也可以直接使用KDE来计算互信息,
            # 因为\hat{y}仅仅只是预测值,不是真实的标签y, 自然也可以当成隐藏层来计算互信息
            # 计算I(T;X) upper
            hM_upper = entropy_estimator_kl_simple(activation_i, noise_variance)
            hM_given_X = kde_multivariate_gauss_entropy(activation_i, noise_variance)
            self.MI_hM_X_upper.append(nats2bits * (hM_upper - hM_given_X))
            # 计算I(T;Y) upper
            hM_given_Y_upper = 0.
            for i in range(10):
                activation_i_for_Y_i = torch.index_select(activation_i, dim=0, index=Y_i_idx[i])
                hM_given_Y_i_upper = entropy_estimator_kl_simple(activation_i_for_Y_i, noise_variance)
                hM_given_Y_upper += Y_probs[i].item() * hM_given_Y_i_upper

            self.MI_hM_Y_upper.append(nats2bits * (hM_upper - hM_given_Y_upper))
            # H_Y_given_X = kde_multivariate_gauss_entropy(activation_i, noise_variance)
            # H_Y_upper = entropy_estimator_kl_simple(activation_i, noise_variance)
            # H_Y_lower = entropy_estimator_bd(activation_i, noise_variance)
        # after calculation , clear the activation list


    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
