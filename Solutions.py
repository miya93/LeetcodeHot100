

class Solution:
    """
    无重复字符的最长子串
    给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度。
    示例 1:
        输入: s = "abcabcbb"
        输出: 3
        解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3
    Solution：滑动窗口 + 哈希集合 （判断是否有重复的字符）
    时间复杂度：O(n)：只需遍历字符串一次，n 为字符串长度
    空间复杂度：O(∣Σ∣)，其中 Σ 表示字符集（即字符串中可以出现的字符），∣Σ∣ 表示字符集的大小

    """
    def lengthOfLongestSubstring(self, s: str) -> int:
        #哈希集合，记录每个字符是否出现过
        occ = set()
        n = len(s)
        #右指针，以及存放最大长度变量
        rk, ans = 0, 0
        #i可作为左指针
        for i in range(n):
            if i != 0:
                # 左指针向右移动一格，移除一个字符
                occ.remove(s[i-1])
            while rk < n and s[rk] not in occ:
                occ.add(s[rk])
                # 不断地移动右指针
                rk += 1
            ans = max(ans, rk - i)
        return ans

    """
        变形：寻找字符串中不包含重复字符的最长子串及其长度

        参数:
            s (str): 输入字符串

        返回:
            tuple: (最长无重复字符子串, 子串长度)
        """
    def lengthAndStrOfLongestSubstring(self, s: str) -> tuple:
        #哈希集合，记录每个字符是否出现过
        occ = set()
        n = len(s)
        #右指针
        rk = 0
        max_substring = ""  # 记录最长子串
        max_length = 0  # 记录最大长度
        #i可作为左指针
        for i in range(n):
            if i != 0:
                # 左指针向右移动一格，移除一个字符
                occ.remove(s[i-1])
            while rk < n and s[rk] not in occ:
                occ.add(s[rk])
                # 不断地移动右指针
                rk += 1
            cur_length = rk - i
            if cur_length > max_length:
                max_length = cur_length
                max_substring = s[i : i + max_length]
        return max_substring, max_length

if __name__ == '__main__':
    sol = Solution()
    print(sol.lengthOfLongestSubstring("abcabcbb"))
    print(sol.lengthAndStrOfLongestSubstring("abcabcbb"))
    print(sol.lengthOfLongestSubstring("pwwkeww"))
    print(sol.lengthAndStrOfLongestSubstring("pwwkeww"))
