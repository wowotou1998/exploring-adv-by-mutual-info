import numpy as np
import torch.nn.functional as F
import torch
from MI_estimators.simple_bin import bin_calc_information2
from MI_estimators.pytorch_kde import kde_multivariate_gauss_entropy, entropy_estimator_kl_simple, entropy_estimator_bd

"""
此处注意，
layer_names 和 layer_activations 不一定是一一对应的，
layer_activations 可能会包含 nn.sequential 中的子模块， 可能实际 layer_activations 的元素要远远超出模型中 nn.sequential 的大小
只要是 torch.nn 类， hook 函数都会钩住
"""


class mutual_info_estimator(object):
    def __init__(self, modules_to_hook,
                 By_Layer_Name=False,
                 Label_Num=10,
                 Enable_Detail=False):
        # 根据modules_to_hook中元素的类型是不是字符串对象来判断
        self.Label_Num = Label_Num
        self.By_Layer_Name = isinstance(modules_to_hook[0], str)

        self.DO_LOWER = True
        self.DO_UPPER = True
        self.DO_BIN = True
        self.Enable_Detail = Enable_Detail

        # self.DO_MINE = False

        self.modules_to_hook = modules_to_hook
        self.layer_names = []
        self.layer_activations = []
        self.handle_list = []

        self.epoch_MI_hM_X_lower = []
        self.epoch_MI_hM_Y_lower = []
        self.epoch_MI_hM_Y_lower_detail = []

        self.epoch_MI_hM_X_upper = []
        self.epoch_MI_hM_Y_upper = []

        self.epoch_MI_hM_X_bin = []
        self.epoch_MI_hM_Y_bin = []

        # self.epoch_MI_hM_X_mine = []
        # self.epoch_MI_hM_Y_mine = []

        # temp
        self.epoch_i_MI_hM_X_lower = []
        self.epoch_i_MI_hM_Y_lower = []
        self.epoch_i_MI_hM_Y_lower_detail = []

        self.epoch_i_MI_hM_X_upper = []
        self.epoch_i_MI_hM_Y_upper = []

        self.epoch_i_MI_hM_X_bin = []
        self.epoch_i_MI_hM_Y_bin = []

        # def hook(self, layer, input, output):
        """
        hook 函数钩住的对象一旦不是模型上有关系的对象，就会出很多问题
        """
        # print("before hook, ", len(layer_activations))
        # print(layer)

        # print("after hook, ", len(layer_activations))

    def clear_all(self):
        self.cancel_hook()
        self.clear_activations()

        self.epoch_MI_hM_X_lower.clear()
        self.epoch_MI_hM_Y_lower.clear()
        self.epoch_MI_hM_Y_lower_detail.clear()

        self.epoch_MI_hM_X_upper.clear()
        self.epoch_MI_hM_Y_upper.clear()

        self.epoch_MI_hM_X_bin.clear()
        self.epoch_MI_hM_Y_bin.clear()

        # temp variable
        self.epoch_i_MI_hM_X_lower.clear()
        self.epoch_i_MI_hM_Y_lower.clear()
        self.epoch_i_MI_hM_Y_lower_detail.clear()

        self.epoch_i_MI_hM_X_upper.clear()
        self.epoch_i_MI_hM_Y_upper.clear()

        self.epoch_i_MI_hM_X_bin.clear()
        self.epoch_i_MI_hM_Y_bin.clear()

    def cancel_hook(self):
        # print("handle list len", len(handle_list))
        # 执行完remove()操作后 清除handle_list列表

        for handle in self.handle_list:
            handle.remove()
        self.handle_list.clear()

    def clear_activations(self):
        self.layer_activations.clear()
        self.layer_names.clear()

    def do_forward_hook(self, model):
        if self.By_Layer_Name:
            for layer_name, layer in model.named_modules():
                if layer_name in self.modules_to_hook:
                    self.layer_names.append(layer_name)
                    # print('layer name: ', layer_name)
                    handle = layer.register_forward_hook(
                        lambda layer, input, output:
                        self.layer_activations.append(output.clone().detach().cpu().view(output.size(0), -1)))
                    # self.layer_activations.append(output.clone().detach().view(output.size(0), -1)))
                    self.handle_list.append(handle)

        else:
            for layer_name, layer in model.named_modules():
                if isinstance(layer, self.modules_to_hook):
                    self.layer_names.append(layer_name)
                    # print('layer name: ', layer_name)
                    handle = layer.register_forward_hook(
                        lambda layer, input, output:
                        self.layer_activations.append(output.clone().detach().cpu().view(output.size(0), -1)))
                    # self.layer_activations.append(output.clone().detach().view(output.size(0), -1)))
                    self.handle_list.append(handle)

        """
        named_children， hook的bug之源
        """
        # for name_i, layer_i in model.named_children():
        #     if isinstance(layer_i, nn.Sequential):
        #         for name_ii, layer_ii in layer_i.named_children():
        #             if isinstance(layer_ii, (nn.ReLU, nn.Tanh)):  # 是元组中的一个返回 True
        #                 layer_names.append(name_i + name_ii)
        #                 handle = layer_ii.register_forward_hook(hook)
        #                 # 保存handle对象， 方便随时取消hook
        #                 handle_list.append(handle)

    def caculate_MI(self, X, Y):
        layer_activations = self.layer_activations
        print("---> caculate_MI, layer activations size[%d],sample num[%d] <---" % (len(layer_activations),
                                                                                    layer_activations[0].size(0)))

        MI_hM_X_lower = []
        MI_hM_Y_lower = []
        """
        如果不做细致的分析，MI_hM_Y_upper 列表中每一个元素都是每一层神经网络的互信息值
        """
        MI_hM_Y_lower_detail = []

        MI_hM_X_upper = []
        MI_hM_Y_upper = []

        MI_hM_X_bin = []
        MI_hM_Y_bin = []

        # MI_hM_X_mine = []
        # MI_hM_Y_mine = []

        label_num = self.Label_Num
        noise_variance = 0.1
        nats2bits = 1.0 / np.log(2)
        # print("layer_activations len", len(layer_activations))

        """
        获取标签Y中label_i的经验概率
        """
        Y_one_hot = F.one_hot(Y, num_classes=label_num).float().to(Y.device)
        Y_probs = torch.mean(Y_one_hot, dim=0)

        """
        获取标签Y中等于label_i的下标集合, pytorch中的tensor可以使用布尔索引,布尔索引中的元素要为布尔值
        """
        Y_i_idx = []
        for i in range(label_num):
            Y_equal_label_i_index = torch.flatten(Y == i)
            Y_i_idx.append(Y_equal_label_i_index)

        saved_label_idx = {}
        for idx, value in enumerate(Y_i_idx):
            saved_label_idx[idx] = value.clone().detach().cpu().numpy()

        for layer_idx, layer_i_activations in enumerate(layer_activations):
            layer_i_lower_detail = []
            # -------- I(T;X), I(T;Y)  MINE --------
            """
            实践证明， MINE的效果非常不理想，超参数， 神经网络的设置是一个让人很头疼的问题
            """
            # if self.DO_MINE:
            #     MI_hM_X_mine_i = calculate_MI_MINE(layer_i_activations, X)
            #     MI_hM_Y_mine_i = calculate_MI_MINE(layer_i_activations, Y)
            #     MI_hM_X_mine.append(nats2bits * MI_hM_X_mine_i)
            #     MI_hM_Y_mine.append(nats2bits * MI_hM_Y_mine_i)

            """
            -------- I(T;X), I(T;Y)  binning --------
            """
            if self.DO_BIN:
                """
                为什么实验结果绘制出的各层的信息变化曲线是平行线且几乎重合在一起呢， 是因为
                几乎每一层的 H_LAYER =ln(batch_size) , ln(100) = 4.605..8809,即每一个批次中所有结果是等概率出现的， 概率为 1/batch_size
                另外 H_LAYER_GIVEN_OUTPUT 约等于 ln(batch_size/category_of_label) ,当 batch_size =100, category_of_label =10 时
                H_LAYER_GIVEN_OUTPUT = ln(10)=2.302...
                """
                MI_hM_X_bin_layer_i, MI_hM_Y_bin_layer_i = bin_calc_information2(saved_label_idx,
                                                                                 layer_i_activations.cpu().numpy(),
                                                                                 0.5)
                # MI_hM_X_bin.append(MI_hM_X_bin_layer_i)
                # MI_hM_Y_bin.append(MI_hM_Y_bin_layer_i)
                MI_hM_X_bin.append(nats2bits * MI_hM_X_bin_layer_i)
                MI_hM_Y_bin.append(nats2bits * MI_hM_Y_bin_layer_i)

            """
            -------- I(T;X), I(T;Y) lower  --------
            """
            # 最后一层输出 \hat{y} 也可以直接使用KDE来计算互信息, 因为 \hat{y} 仅仅只是预测值,不是真实的标签 y, 自然也可以当成隐藏层来计算互信息

            hM_given_X = kde_multivariate_gauss_entropy(layer_i_activations, noise_variance)

            if self.DO_LOWER:
                # -------- I(T;X) lower --------
                hM_lower = entropy_estimator_bd(layer_i_activations, noise_variance)
                MI_hM_X_lower.append(nats2bits * (hM_lower - hM_given_X))

                # -------- I(T;Y) lower --------
                hM_given_Y_lower = 0.
                for y_i in range(label_num):
                    """
                    依次选择激活层i中有关于标签j的激活值， 并计算这部分激活值的的互信息
                    """
                    # 获取第i层激活值关于标签i的部分， 使用bool索引

                    activation_i_for_Y_i = layer_i_activations[Y_i_idx[y_i], :]
                    hM_given_Y_i_lower = entropy_estimator_bd(activation_i_for_Y_i, noise_variance)
                    hM_given_Y_lower += Y_probs[y_i].item() * hM_given_Y_i_lower

                    # 存储 H(T|y) 的信息 以及 p(y) 的概率
                    # 这里感觉没必要把各个类别出现的经验概率值记录下来
                    # layer_i_lower_detail.append(Y_probs[y_i].item())
                    if self.Enable_Detail:
                        layer_i_lower_detail.append(nats2bits * hM_given_Y_i_lower)
                if self.Enable_Detail:
                    layer_i_lower_detail.append(nats2bits * hM_lower)
                MI_hM_Y_lower.append(nats2bits * (hM_lower - hM_given_Y_lower))
                if self.Enable_Detail:
                    MI_hM_Y_lower_detail.append(layer_i_lower_detail)

            # -------- I(T;X), I(T;Y)  upper  --------
            if self.DO_UPPER:
                # -------- I(T;X) upper --------
                hM_upper = entropy_estimator_kl_simple(layer_i_activations, noise_variance)
                MI_hM_X_upper.append(nats2bits * (hM_upper - hM_given_X))

                # -------- I(T;Y) upper --------
                hM_given_Y_upper = 0.
                for y_i in range(label_num):
                    """
                    依次选择激活层i中有关于标签j的激活值， 并计算这部分激活值的的互信息
                    """
                    # 获取第i层激活值关于标签i的部分， 使用bool索引
                    activation_i_for_Y_i = layer_i_activations[Y_i_idx[y_i], :]
                    hM_given_Y_i_upper = entropy_estimator_kl_simple(activation_i_for_Y_i, noise_variance)
                    hM_given_Y_upper += Y_probs[y_i].item() * hM_given_Y_i_upper

                MI_hM_Y_upper.append(nats2bits * (hM_upper - hM_given_Y_upper))
        # 在计算完所有层的互信息之后，临时存储所有结果
        if self.DO_BIN:
            self.epoch_i_MI_hM_X_bin = MI_hM_X_bin
            self.epoch_i_MI_hM_Y_bin = MI_hM_Y_bin
        if self.DO_UPPER:
            self.epoch_i_MI_hM_X_upper = MI_hM_X_upper
            self.epoch_i_MI_hM_Y_upper = MI_hM_Y_upper
        if self.DO_LOWER:
            self.epoch_i_MI_hM_X_lower = MI_hM_X_lower
            self.epoch_i_MI_hM_Y_lower = MI_hM_Y_lower

            self.epoch_i_MI_hM_Y_lower_detail = MI_hM_Y_lower_detail
        # if self.DO_MINE:
        #     self.epoch_i_MI_hM_X_mine = MI_hM_X_mine
        #     self.epoch_i_MI_hM_Y_mine = MI_hM_Y_mine

    def store_MI(self):
        if self.DO_BIN:
            self.epoch_MI_hM_X_bin.append(self.epoch_i_MI_hM_X_bin)
            self.epoch_MI_hM_Y_bin.append(self.epoch_i_MI_hM_Y_bin)
        if self.DO_UPPER:
            self.epoch_MI_hM_X_upper.append(self.epoch_i_MI_hM_X_upper)
            self.epoch_MI_hM_Y_upper.append(self.epoch_i_MI_hM_Y_upper)
        if self.DO_LOWER:
            self.epoch_MI_hM_X_lower.append(self.epoch_i_MI_hM_X_lower)
            self.epoch_MI_hM_Y_lower.append(self.epoch_i_MI_hM_Y_lower)
            self.epoch_MI_hM_Y_lower_detail.append(self.epoch_i_MI_hM_Y_lower_detail)
