# Basic Binary Search
# 返回与target最相近的较小元素的index


def find_closest(a, target):
    if a is None or not isinstance(a, list) or len(a) == 0:
        return -1
    left = 0
    right = len(a) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if target > a[mid]:
            left = mid + 1
        elif target < a[mid]:
            right = mid - 1
        else:
            return mid
    return max(0, left - 1)     # 确保返回的索引不为负


print(find_closest('sssss', 1))
print(find_closest([], 111))
print(find_closest([2, 3, 5, 7, 11, 13, 17], 2))
print(find_closest([2, 3, 5, 7, 11, 13, 17], 1))
print(find_closest([2, 3, 5, 7, 11, 13, 17], 20))

