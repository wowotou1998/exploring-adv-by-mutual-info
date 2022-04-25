import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt

L = []


def viz(module, input, output):
    x = input[0][0]
    # 最多显示4张图
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i + 1)
        plt.imshow(x[i].cpu())
    # plt.show()
    L.append(output)


import cv2
import numpy as np

handle_L = []


def main():
    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet18(pretrained=True).to(device)
    print(model)
    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and
        #         not isinstance(m, torch.nn.Sequential) and
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.Conv2d):
            handle = m.register_forward_hook(viz)
            handle_L.append(handle)

    img = cv2.imread('cat.png')
    img = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(img)
    print("uu")


if __name__ == '__main__':
    main()
