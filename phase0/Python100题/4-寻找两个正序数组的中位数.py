# https://leetcode.cn/problems/median-of-two-sorted-arrays/description/
from typing import List

class Solution:
    @staticmethod
    def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
        # 保证 nums1 是较短数组，二分只在短数组上做
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)

        left, right = 0, m
        half = (m + n + 1) // 2          # 左边需要多少元素

        while left <= right:
            i = (left + right) // 2      # nums1 的分割线
            j = half - i                 # nums2 的分割线

            # 边界值处理：若分割到数组外，用 ±inf 占位
            nums1_left_max  = float('-inf') if i == 0 else nums1[i-1]
            nums1_right_min = float('inf')  if i == m else nums1[i]
            nums2_left_max  = float('-inf') if j == 0 else nums2[j-1]
            nums2_right_min = float('inf')  if j == n else nums2[j]

            # 满足左边整体 ≤ 右边整体
            if nums1_left_max <= nums2_right_min and nums2_left_max <= nums1_right_min:
                # 总长度奇偶决定中位数
                if (m + n) % 2 == 1:
                    return max(nums1_left_max, nums2_left_max)
                else:
                    return (max(nums1_left_max, nums2_left_max) +
                            min(nums1_right_min, nums2_right_min)) / 2.0
            # 二分调整
            elif nums1_left_max > nums2_right_min:
                right = i - 1   # nums1 割太靠右，左移
            else:
                left = i + 1    # nums1 割太靠左，右移

        # 理论上一定能找到，不会走到这里
        raise ValueError("No median found")


# ------------------ 测试 ------------------
if __name__ == "__main__":
    sol = Solution()
    print(sol.findMedianSortedArrays([1, 3], [2]))        # 2.0
    print(sol.findMedianSortedArrays([1, 2], [3, 4]))     # 2.5
    print(sol.findMedianSortedArrays([], [1]))            # 1.0
    print(sol.findMedianSortedArrays([2], []))            # 2.0
