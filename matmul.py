import numpy as np

# 实现二维矩阵乘法
#
#          N = 2, M = 3, P = 4
#                        |  |  |  |
#    ──*──*──*──         *  *  *  *         *  *  *  *
#                        |  |  |  |
#    ──*──*──*──         *  *  *  *         *  *  *  *
#       (2x3)            |  |  |  |
#                        *  *  *  *
#                        |  |  |  |
#                           (3x4)
# for i in range(N):
#     for j in range(P):
#         for k in range(M):
#             res[i, j] += a[i, k] * b[k, j]
#
# or
#
# for i in range(N):
#     for j in range(P):
#         res[i, j] = a[i, :] * b[:, j]

def matmul(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise Exception("Input must be numpy array!")

    if a.ndim > 2 or b.ndim > 2:
        raise Exception("Input dimension must not exceed 2!")

    if a.ndim < 2 or b.ndim < 2:
        return a * b

    if a.shape[1] != b.shape[0]:
        raise Exception("Input dimensions does not match!")

    N, M, M, P = a.shape[0], a.shape[1], b.shape[0], b.shape[1]
    res = np.zeros((N, P))

    for i in range(N):
        for j in range(P):
            for k in range(M):
                res[i, j] += a[i, k] * b[k, j]
    return res


a = np.random.randint(0, 9, size=(2, 3))
b = np.random.randint(0, 9, size=(3, 4))
print(matmul(a, b))
print(np.dot(a, b))
