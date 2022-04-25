import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

net = ptcv_get_model("resnet20_cifar10", pretrained=True)

# print(net)
model_output_list = list()
model_input_list = list()


def forward_hook(module, data_input, data_output):
    model_input_list.append(data_input)
    model_output_list.append(data_output)


net.features.init_block.register_forward_hook(forward_hook)
fake_img = torch.ones((1, 3, 32, 32))  # batch size * channel * H * W
output = net(fake_img)

print(net)
# print(model_input_list[0])
# print(model_output_list[0])

# print(net.features.stage1.unit1)
# for layer in net.named_modules():
#     print(layer)
# 使用isinstance可以判断这个模块是不是所需要的类型实例


# summary(net, input_size=(3, 32, 32))
