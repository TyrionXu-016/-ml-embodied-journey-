# https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/
# 滑动窗口解题
class Solution:
    @staticmethod
    def length_of_longest_substring(s: str) -> int:
        left = 0  # 窗口左端（含）
        last = {}  # 字符 -> 最近一次下标
        max_len = 0

        for right, ch in enumerate(s):  # right 是窗口右端（含）
            if ch in last and last[ch] >= left:  # 重复且在窗口内
                left = last[ch] + 1  # 左边界跳到重复位之后
            last[ch] = right  # 更新/记录当前字符位置
            max_len = max(max_len, right - left + 1)

        return max_len


# --- 测试 ---
if __name__ == "__main__":
    sol = Solution()
    print(sol.length_of_longest_substring("abcabcbb"))  # 3
    print(sol.length_of_longest_substring("bbbbb"))  # 1
    print(sol.length_of_longest_substring("pwwkew"))  # 3
    print(sol.length_of_longest_substring(""))  # 0
