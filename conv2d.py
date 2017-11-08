import numpy as np
import tensorflow as tf

# <二维卷积的计算复杂度分析>
# - 输入图像 (H, W, C) （单张输入图像）
# - 卷积核尺寸 (KH, KW, C) （单个卷积核）
'''
Time - O(H * W * KH * KW * C)
'''
# 如果使用KC个卷积核，针对N张输出图像，那么时间复杂度就是
'''
Time - O(H * W * KH * KW * C * KC * N)
'''


# Assume stride = 1, padding = 0
# output = (input - kernel + 2 * padding) / stride + 1
def conv2d(img, kernel):
    height, width, in_channels = img.shape
    kernel_height, kernel_width, in_channels, out_channels = kernel.shape
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1
    feature_maps = np.zeros(shape=(out_height, out_width, out_channels))
    for oc in range(out_channels):              # Conv with each kernel
        for h in range(out_height):             # Scan height
            for w in range(out_width):          # Scan width
                for ic in range(in_channels):   # sum all input channel.
                    patch = img[h: h + kernel_height, w: w + kernel_width, ic]
                    feature_maps[h, w, oc] += np.sum(patch * kernel[:, :, ic, oc])

    return feature_maps


# 简化版，用slicing代替遍历所有ic通道的过程
def conv2d2(img, kernel):
    height, width, in_channels = img.shape
    kernel_height, kernel_width, in_channels, out_channels = kernel.shape
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1
    feature_maps = np.zeros(shape=(out_height, out_width, out_channels))
    for oc in range(out_channels):              # Conv with each kernel
        for h in range(out_height):             # Scan height
            for w in range(out_width):          # Scan width
                patch = img[h: h + kernel_height, w: w + kernel_width, :]
                feature_maps[h, w, oc] = np.sum(patch * kernel[:, :, :, oc])

    return feature_maps


# 支持 Stride 的二维卷积实现。注意for循环遍历的是输出索引，输入索引需要乘以stride的倍数。
def conv2d_with_stride(img, kernel, stride):
    height, width, in_channels = img.shape
    kernel_height, kernel_width, in_channels, out_channels = kernel.shape
    out_height = (height - kernel_height)//stride + 1
    out_width = (width - kernel_width)//stride + 1
    feature_maps = np.zeros(shape=(out_height, out_width, out_channels))
    print('Output Shape = ', feature_maps.shape)
    for oc in range(out_channels):              # Conv with each kernel
        for h in range(out_height):             # Scan height
            for w in range(out_width):          # Scan width
                h_stride = h * stride           # h 作为输出索引，h_stride 作为输入索引
                w_stride = w * stride           # w 作为输出索引，w_stride 作为输入索引
                patch = img[h_stride: h_stride + kernel_height, w_stride: w_stride + kernel_width, :]
                feature_maps[h, w, oc] = np.sum(patch * kernel[:, :, :, oc])

    return feature_maps


def test1():
    img = np.random.randint(0, 9, size=(3, 3, 3)).astype(np.float32)
    kernel = np.random.randint(0, 9, size=(2, 2, 3, 1)).astype(np.float32)

    A = tf.Variable(np.expand_dims(img, axis=0))
    C = tf.nn.conv2d(input=A, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_result = sess.run(C)
        tf_result = tf_result[0]

    my_result = conv2d(img, kernel)
    print(tf_result)
    print(my_result)
    print('\nResult Matched with TF:', np.array_equal(tf_result, my_result))


def test2():
    img = np.random.randint(0, 9, size=(10, 10, 3)).astype(np.float32)
    kernel = np.random.randint(0, 9, size=(3, 3, 3, 1)).astype(np.float32)

    A = tf.Variable(np.expand_dims(img, axis=0))
    C = tf.nn.conv2d(input=A, filter=kernel, strides=[1, 3, 3, 1], padding='VALID')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_result = sess.run(C)
        tf_result = tf_result[0]

    my_result = conv2d_with_stride(img, kernel, stride=3)
    print(tf_result)
    print(my_result)
    print('\nResult Matched with TF:', np.array_equal(tf_result, my_result))


test1()
test2()
