# utf-8
"""
named_children()----返回的是子模块的迭代器还有名字
named_modules()----返回的是所有模块的迭代器还有名字
net.children()  返回网络模块的第一代子模块
net.modules()  返回网络模块的自己本身和所有后代模块
"""
import ModelSet
import torch.nn as nn

Activation_F = 'ReLU'
# model = ModelSet.FC_with_Sigmoid(Activation_F)
model = ModelSet.FC(Activation_F)

# print(len(list(model.children())))
# for i in model.children():
#     print(i)
#
# print("--------------------------")
#
# print(len(list(model.modules())))
# for i in model.modules():
#     if  isinstance(i, nn.Sequential):
#         print(i[len(i)-1])
#         # print(type(i))
#         # print(i)
#
# print("--------------------------")

# print(len(list(model.named_children())))
# for name,i in model.named_children():
#     # if  isinstance(i, nn.Sequential):
#     #     print(i[len(i)-1])
#         # print(type(i))
#         print(name,i)

# print(len(list(model.modules())))
# for name, i in model.named_modules():
#     if  isinstance(i, nn.Sequential):
#         print(i[len(i)-1])
#     print(type(i))
#     print(name, '___', i)

# for  i in model.modules():
#     print('------',i)
#     # print(type(i))

# print(len(list(model.named_children())))
# for name,i in model.named_modules():
#     # if  isinstance(i, nn.Sequential):
#     #     print(i[len(i)-1])
#         # print(type(i))
#         print(name,"-----",i)

# 直接使用这种方式来获取相应的layer
# print(model.fc2[1])

# for layer in model.modules():
#     if isinstance(layer, torch.nn.modules.conv.Conv2d):
#         handle = layer.register_forward_hook(save_output)
#         hook_handles.append(handle)

# for layer in model.children():
#     print('____')
#     print(layer)

import torch


def get_all_layers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []

    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """

        # get all layers of the model
        # layer_list 是一个列表， 列表里的每一个元素是一个元组，元组有两个元素， 第一个元素是名称， 第二个元素是对象
        layer_list = list(model.named_children())
        # print(layer_list)
        # print("---")
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)
            # if current layer contains sublayers, add current layer name on its sublayers
            # 如果模块i没有子模块， 则模块i加入大集合中
            if sublayer_num == 0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            # 如果模块i有子模块， 则则对子模块进行遍历
            elif isinstance(module, torch.nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers


layers = get_all_layers(model)
for i in layers:
    print(i)
