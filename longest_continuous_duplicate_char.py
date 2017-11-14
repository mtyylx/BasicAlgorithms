
# 需求
# 找到字符串序列中连续重复出现次数最多的字符和长度，如果出现多次相同长度的字符串，只计算第一个
# 注意的点：
# - 双指针 + 滑动窗：单循环而不是双循环
# - 在 max_len == curr_len 时不更新 max_len 和 具体字符内容
# - 在完整扫描数组后，要额外判断结尾字符串是否会刷新结果


def get_max_duplicate(s):
    if s is None or len(s) == 0:
        return None, 0
    i = 0
    j = 1
    res = s[i]
    curr_len = 1
    max_len = 1
    while i < len(s) and j < len(s):
        if s[j] == s[j - 1]:
            j += 1
            curr_len += 1
        else:
            if max_len < curr_len:
                max_len = curr_len
                res = s[i]
            curr_len = 1
            i = j
            j += 1
    if j == len(s):
        if max_len < curr_len:
            return s[i], curr_len
    return res, max_len


print(get_max_duplicate(None))
print(get_max_duplicate(''))
print(get_max_duplicate('a'))
print(get_max_duplicate('ab'))
print(get_max_duplicate('aaabbbccc'))
print(get_max_duplicate('abcabcabccc'))