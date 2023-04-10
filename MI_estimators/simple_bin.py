# Simplified MI computation code from https://github.com/ravidziv/IDNNs
import numpy as np

# get_unique_probs 这个函数可以处理高维随机变量的熵的计算问题.
# 无论x中每一个元素的维数是多少, 我们只需要计算相应的频次就可以得出每一个元素的概率值,
def get_unique_probs(x):
    # np.ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    # 创建一个新的数据类型, 大小为x的dimension * x.dtype.itemsize, (x的大小为batchsize * dimension)
    uniqueids = np.ascontiguousarray(x).view(
        np.dtype(
            (np.void, x.dtype.itemsize * x.shape[1])
        )
    )
    # np.unique去除数组中的重复数字，并进行排序之后输出
    # return_index=True表示返回新列表元素在旧列表中的位置，并以列表形式储存在返回值中
    # return_inverse = True 表示返回旧列表元素在新列表中的位置，并以列表形式储存在unique_inverse中
    # return_counts= True, 返回新列表元素在求列表中出现的次数

    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    # 当数据源是ndarray时，np.array仍然会copy出一个副本，占用新的内存，但np.asarray不会.
    # unique_counts / float(sum(unique_counts)) 返回了各数据出现的概率值
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


def bin_calc_information(inputdata, layerdata, num_of_bins):
    p_xs, unique_inverse_x = get_unique_probs(inputdata)

    bins = np.linspace(-1, 1, num_of_bins, dtype='float32')
    digitized = bins[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins) - 1].reshape(len(layerdata), -1)
    p_ts, _ = get_unique_probs(digitized)

    H_LAYER = -np.sum(p_ts * np.log(p_ts))
    H_LAYER_GIVEN_INPUT = 0.
    for xval in unique_inverse_x:
        p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
        H_LAYER_GIVEN_INPUT += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))
    return H_LAYER - H_LAYER_GIVEN_INPUT


def bin_calc_information2(labelixs, layerdata, binsize):
    # This is even further simplified, where we use np.floor instead of digitize
    def get_h(d):
        digitized = np.floor(d / binsize).astype('int')
        p_ts, _ = get_unique_probs(digitized)
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(layerdata)
    H_LAYER_GIVEN_OUTPUT = 0.
    # labelixs 是一个字典, k为标签id, v存储与标签id相关的数据在batch中的索引
    # ixs.mean() 表示标签i发生的概率
    """
    ixs存储与标签i相关的数据在batch中的布尔索引, pytorch中应该可以使用select_nonzero函数, 也可以使用bool索引
    saved_labelixs[i] 为一维数组, 一共有batchsize个元素, 每一个元素为布尔值要么为False, 要么是True
    我们可以通过一个布尔数组来索引目标数组，以此找出与布尔数组中值为True的对应的目标数组中的数据（后面通过实例可清晰的观察).
    需要注意的是，布尔数组的长度必须与目标数组对应的轴的长度一致.
    """
    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(layerdata[ixs, :])
    return H_LAYER, H_LAYER - H_LAYER_GIVEN_OUTPUT
    # H_LAYER_GIVEN_INPUT = 0.
    # for xval in unique_inverse_x:
    #     p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
    #     H_LAYER_GIVEN_INPUT += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))
    # print('here', H_LAYER_GIVEN_INPUT)
    # return H_LAYER - H_LAYER_GIVEN_INPUT
