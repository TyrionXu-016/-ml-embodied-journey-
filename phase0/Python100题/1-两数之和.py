# https://leetcode.cn/problems/two-sum/description/
from typing import List


# O(n) 哈希表解法
def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}  # 数值 -> 下标
    for i, num in enumerate(nums):
        need = target - num
        if need in seen:  # 之前已经出现过的数
            return [seen[need], i]  # 先出现的在前，后出现的在后
        seen[num] = i  # 存入当前数及其下标
    raise ValueError("No two sum solution")  # 按题意这行不会走到


# --- 测试 ---
if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
    print(two_sum([3, 2, 4], 6))  # [1, 2]
    print(two_sum([3, 3], 6))  # [0, 1]
