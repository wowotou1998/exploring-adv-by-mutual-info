import torch
import numpy as np


def calculate_dists_matrix(x):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    #  The 0 axis in tensorflow is the rows, 1 axis is the columns.
    #  By doing tf.reduce_sum(x, 0) the tensor is reduced along the first dimension
    #  (rows), so the result is [1, 2, 4] + [8, 16, 32] = [9, 18, 32].
    #
    #  By doing tf.reduce_sum(x, 1) the tensor is reduced along the second dimension
    #  (columns), so the result is [1, 9] + [2, 16] + [4, 32] = [7, 56].

    # calculate_dists_matrix 是在一次性计算n个向量的相互之间的2范数的平方, |X_i-X_j|_2^2

    x_square = torch.unsqueeze(torch.sum(x.pow(2), dim=1), dim=1)
    # x2.shape = batch_size * 1, // x2.t().shape = 1 * batch_size
    # x2 + x2.t() 会根据广播机制自动扩展shape= batch_size * batch_size 大小的矩阵
    dists = x_square + x_square.t() - 2 * torch.matmul(x, x.t())
    return dists





def entropy_estimator_kl_simple(x, var):
    # 这个值的计算结果和上式得出的结果一致, 可以说是上式entropy_estimator_kl的简化版本
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    # 返回维度和批次当做的采样个数
    N, dims = x.size(0) * 1.0, x.size(1) * 1.0
    dists = calculate_dists_matrix(x)
    dists2 = dists / (2 * var)
    const = (dims / 2.0) * np.log(2 * np.pi * var)
    log_sum_exp = torch.logsumexp(-dists2, dim=1)
    h = torch.mean(log_sum_exp)
    result = dims / 2 + const - h.item() + np.log(N)
    return result


def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    # see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    N, dims = x.size(0) * 1.0, x.size(1) * 1.0
    val = entropy_estimator_kl_simple(x, 4 * var)
    return val + np.log(0.25) * dims / 2


def kde_multivariate_gauss_entropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.size(1)
    return (dims / 2.0) * (np.log(2 * np.pi * var) + 1)

# def get_shape(x):
#     N, dims = x.size(0), x.size(1)
#     return dims / 1.0, N / 1.0