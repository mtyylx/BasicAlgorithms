import numpy as np

# 实现二维矩阵乘法 A(N, P) x B(P, M) = C(N, M)
#
#                    N = 2, P = 3, M = 4
#
#                        |  |  |  |
#    ──*──*──*──         *  *  *  *         *  *  *  *
#                        |  |  |  |
#    ──*──*──*──         *  *  *  *         *  *  *  *
#      A(2x3)            |  |  |  |           C(2x4)
#                        *  *  *  *
#                        |  |  |  |
#                          B(3x4)
#
# 矩阵 A(N,P) 与矩阵 B(P,M) 相乘得到运算结果矩阵 C(N,M)，其本质是：
# 1. 将矩阵A拆分为 N 个<行向量>，以 i 作为矩阵A<行向量>的索引，i 取值范围 [0, N)
# 2. 将矩阵B拆分为 M 个<列向量>，以 j 作为矩阵B<列向量>的索引，j 取值范围 [0, M)
# 3. 两组向量的长度相等，都等于P，以 k 作为<向量元素>的索引，k 取值范围 [0, P)
# 4. 以 (i, j) 作为矩阵C的元素索引，以第 i 个行向量与第 j 个列向量的点积结果作为矩阵C的元素内容，构造矩阵C。
#
# for i in range(N):           <--- 矩阵C中元素的横坐标（行号）
#     for j in range(M):       <--- 矩阵C中元素的纵坐标（列号）
#         for k in range(P):   <--- 每个向量内元素的索引
#             res[i, j] += A[i, k] * B[k, j]     <--- 矩阵A按行访问，矩阵B按列访问
#


def matmul(A, B):
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise Exception("Input must be numpy array!")

    if A.ndim > 2 or B.ndim > 2:
        raise Exception("Input dimension must not exceed 2!")

    if A.ndim < 2 or B.ndim < 2:
        return A * B

    if A.shape[1] != B.shape[0]:
        raise Exception("Input dimensions does not match!")

    N, P, P, M = A.shape[0], A.shape[1], B.shape[0], B.shape[1]
    res = np.zeros((N, M), dtype=np.int32)

    for i in range(N):
        for j in range(M):
            for k in range(P):
                res[i, j] += A[i, k] * B[k, j]
    return res


def matmul2(A, B):
    N, P, M = A.shape[0], A.shape[1], B.shape[1]
    res = np.zeros((N, M), dtype=np.int32)

    for i in range(N):
        for j in range(M):
            res[i, j] = np.dot(a[i, :], b[:, j])        # 简化示意：本质上<矩阵的乘法>就是由<向量的点积>所构成的。
    return res


a = np.random.randint(0, 9, size=(2, 3))
b = np.random.randint(0, 9, size=(3, 4))
print(matmul(a, b))
print(matmul2(a, b))
print(np.dot(a, b))
