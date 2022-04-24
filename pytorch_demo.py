# Simple example of how to estimate MI between X and Y, where Y = f(X) + Noise(0, noise_variance)
from __future__ import print_function
import torch
import pytorch_kde
import numpy as np

noise_variance = 0.1

entropy_func_upper = pytorch_kde.entropy_estimator_kl_simple
entropy_func_lower = pytorch_kde.entropy_estimator_bd
# 设定好随机数种子, 保证每次实验结果都相同
np.random.seed(1)
data = np.random.random(size=(100, 30))  # N x dims
data = torch.from_numpy(data)
# 由于Y=f(X)+noise, H(Y|X)的不确定性由noise的分布来决定, 则H(Y|X)的熵=H(noise), noise的分布为多元高斯分布
H_Y_given_X = pytorch_kde.kde_multi_gauss_entropy(data, noise_variance)
H_Y_upper = entropy_func_upper(data, noise_variance)
H_Y_lower = entropy_func_lower(data, noise_variance)

print("Upper bound: %0.3f nats" % (H_Y_upper - H_Y_given_X))
print("Lower bound: %0.3f nats" % (H_Y_lower - H_Y_given_X))



'''
# Alternative calculation, direct from distance matrices
dims, N = pytorch_kde.get_shape(K.variable(data))
dists = pytorch_kde.Kget_dists(K.variable(data))
dists2 = dists / (2 * noise_variance)
mi2 = K.eval(-K.mean(K.logsumexp(-dists2, axis=1) - K.log(N)))
print("Upper bound2: %0.3f nats" % mi2)

dims, N = pytorch_kde.get_shape(K.variable(data))
dists = pytorch_kde.Kget_dists(K.variable(data))
dists2 = dists / (2 * 4 * noise_variance)
mi2 = K.eval(-K.mean(K.logsumexp(-dists2, axis=1) - K.log(N)))
print("Lower bound2: %0.3f nats" % mi2)
'''
