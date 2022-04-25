import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
from ModelSet import FC_3
import matplotlib.pyplot as plt

L = []


def viz(module, input, output):
    # plt.show()
    L.append(output)


import cv2
import numpy as np

handle_L = []


def main():
    t = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FC_3(torch.nn.ReLU()).to(device)
    print(model)
    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and
        #         not isinstance(m, torch.nn.Sequential) and
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.ReLU):
            handle = m.register_forward_hook(viz)
            handle_L.append(handle)

    img = torch.rand(1,1, 28, 28).to(device)
    with torch.no_grad():
        model(img)
    print("uu")


if __name__ == '__main__':
    main()
