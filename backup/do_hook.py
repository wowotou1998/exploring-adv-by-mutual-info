def do_forward_hook(model):
    for name_i, layer_i in model.named_children():
        if isinstance(layer_i, nn.Sequential):
            for name_ii, layer_ii in layer_i.named_children():
                if isinstance(layer_ii, nn.ReLU):
                    layer_names.append(name_i+name_ii)
                    handle = layer_ii.register_forward_hook(
                        lambda torch_nn, input, output:
                        layer_activations.append(output.view(output.size(0), -1))
                    )
                    # 保存handle对象， 方便随时取消hook
                    handle_list.append(handle)

# def register_forward_hook(selected_layers):
#     print(selected_layers)
#     for layer_i in selected_layers:
#         layer_names.append('1')
#         handle = layer_i.register_forward_hook(
#             lambda torch_nn, input, output:
#             layer_activations.append(output.view(output.size(0), -1))
#         )
#         # 保存handle对象， 方便随时取消hook
#         handle_list.append(handle)

# for layer in model.modules():
#     if isinstance(layer, torch.nn.modules.conv.Conv2d):
#         handle = layer.register_forward_hook(save_output)
#         hook_handles.append(handle)

def hook(layer, input, output):
    print("before hook, ", len(layer_activations))
    print(layer)
    layer_activations.append(output.detach().clone().view(output.size(0), -1))
    print("after hook, ", len(layer_activations))


# def do_forward_hook(model):
#     selected_layers = model.get_seleted_layers()
#     print(selected_layers)
#     for layer_i in selected_layers:
#         layer_names.append('1')
#         # hook 机制在 sequential中有很大的问题
#         handle = layer_i.register_forward_hook(hook)
#         # 保存handle对象， 方便随时取消hook
#         handle_list.append(handle)


# def do_forward_hook(model):
#     selected_layers = model.get_selected_layers()
#     print("selected_layers", selected_layers)
#     for layer_i in selected_layers:
#         layer_names.append(layer_i.__class__)
#         # hook 机制在 sequential中有很大的问题
#         handle = layer_i.register_forward_hook(hook)
#         # 保存handle对象， 方便随时取消hook
#         handle_list.append(handle)

from model_test import get_all_layers

"""
hook 函数钩住的对象一旦不是模型上有关系的对象，就会出很多问题
"""


# def do_forward_hook(model):
#     """
#         get each layer's name and its module
#         :param model:
#         :return: each layer's name and its module
#         """
#
#     def unfoldLayer(model):
#         """
#         unfold each layer
#         :param model: the given model or a single layer
#         :param root: root name
#         :return:
#         """
#
#         # get all layers of the model
#         # layer_list 是一个列表， 列表里的每一个元素是一个元组，元组有两个元素， 第一个元素是名称， 第二个元素是对象
#         layer_list = list(model.named_children())
#         # print(layer_list)
#         # print("---")
#         for item in layer_list:
#             module = item[1]
#             sublayer = list(module.named_children())
#             sublayer_num = len(sublayer)
#             # if current layer contains sublayers, add current layer name on its sublayers
#             # 如果模块i没有子模块， 则模块i加入大集合中
#             if sublayer_num == 0:
#                 if isinstance(module, nn.ReLU):
#                     handle = module.register_forward_hook(
#                         lambda torch_nn, input, output:
#                         layer_activations.append(output.view(output.size(0), -1))
#                     )
#                     # 保存handle对象， 方便随时取消hook
#                     handle_list.append(handle)
#             # if current layer contains sublayers, unfold them
#             # 如果模块i有子模块， 则则对子模块进行遍历
#             elif isinstance(module, torch.nn.Module):
#                 unfoldLayer(module)
#
#     unfoldLayer(model)