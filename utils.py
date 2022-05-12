import os
import torch
import numpy as np


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_model(model, file_name):
    model.load_state_dict(
        torch.load(file_name, map_location=lambda storage, loc: storage))


def save_model(model, file_name):
    # 如果文件路径不存在则创建
    # if not os.path.exists(file_name):
    #     os.makedirs(file_name)
    torch.save(model.state_dict(), file_name)


# def save_array_dict(data, file_name):
#     # 如果文件路径不存在则创建
#     # if not os.path.exists(file_name):
#     #     os.makedirs(file_name)
#     np.save("./Checkpoint/%s.npy" % str(file_name), data)
#
#
# def load_array_dict(file_name, Type='Array'):
#     # 如果文件路径不存在则创建
#     if not os.path.exists(file_name):
#         os.makedirs(file_name)
#     if Type == 'Array':
#         k = np.load("./Checkpoint/%s.npy" % str(file_name))
#     else:
#         k = np.load('./Checkpoint/%s.npy' % str(file_name)).item()
#     return k
