# https://leetcode.cn/problems/add-two-numbers/description/
from typing import Optional


# 定义单链表节点
class ListNode:
    def __init__(self, val: int = 0, nxt: Optional["ListNode"] = None):
        self.val = val
        self.next = nxt

    # 辅助：把链表打印成 Python list，方便调试
    def to_list(self):
        out, cur = [], self
        while cur:
            out.append(cur.val)
            cur = cur.next
        return out


# 迭代写法
def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()  # 哨兵头节点，简化边界
    cur = dummy
    carry = 0  # 进位

    while l1 or l2 or carry:  # 只要还有节点或进位就继续
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        total = v1 + v2 + carry
        carry, digit = divmod(total, 10)  # 同时拿到商(进位)和余数(当前位)

        cur.next = ListNode(digit)  # 新建节点接到结果链
        cur = cur.next

        # 指针前移
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next

    return dummy.next


# ------------------ 测试 ------------------
if __name__ == "__main__":
    def build_linked_list(arr):
        dummy = ListNode()
        cur = dummy
        for v in arr:
            cur.next = ListNode(v)
            cur = cur.next
        return dummy.next


    l1 = build_linked_list([2, 4, 3])
    l2 = build_linked_list([5, 6, 4])
    ans = add_two_numbers(l1, l2)
    print(ans.to_list())  # [7, 0, 8]

    l1 = build_linked_list([0])
    l2 = build_linked_list([0])
    print(add_two_numbers(l1, l2).to_list())  # [0]

    l1 = build_linked_list([9, 9, 9, 9, 9, 9, 9])
    l2 = build_linked_list([9, 9, 9, 9])
    print(add_two_numbers(l1, l2).to_list())  # [8, 9, 9, 9, 0, 0, 0, 1]
