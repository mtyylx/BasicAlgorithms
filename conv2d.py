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


img = np.random.randint(0, 9, size=(3, 3, 3)).astype(np.float32)
kernel = np.random.randint(0, 9, size=(2, 2, 3, 1)).astype(np.float32)

A = tf.Variable(np.expand_dims(img, axis=0))
C = tf.nn.conv2d(input=A, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(C))

print(conv2d(img, kernel))
print(conv2d2(img, kernel))