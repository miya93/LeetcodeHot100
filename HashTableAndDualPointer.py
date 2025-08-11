import collections
from typing import List


#题目主要涉及哈希表与双指针
class HashTableAndDualPointer:
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

    """
    两数之和
    给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标
    示例 1：
    输入：nums = [2,7,11,15], target = 9
    输出：[0,1]
    解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1]
    Solution：哈希表
    时间复杂度：O(N)，其中 N 是数组中的元素数量。对于每一个元素 x，我们可以 O(1) 地寻找 target - x
    空间复杂度：O(N)，其中 N 是数组中的元素数量。主要为哈希表的开销
    """
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numsDict = {}
        for i, num in enumerate(nums):
            # 我们首先查询哈希表中是否存在 target - num，然后将 num 插入到哈希表中，即可保证不会让 num 和自己匹配
            if target - num in numsDict:
                return [numsDict[target - num], i]
            numsDict[num] = i
        return []

    """
    字母异位词分组
    给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
    示例 1:
    输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
    Solution：哈希表
    时间复杂度：O(nklogk)，其中 n 是 strs 中的字符串的数量，k 是 strs 中的字符串的的最大长度。需要遍历 n 个字符串，对于每个字符串，需要 O(klogk) 的时间进行排序以及 O(1) 的时间更新哈希表，因此总时间复杂度是 O(nklogk)
    空间复杂度：O(nk)，其中 n 是 strs 中的字符串的数量，k 是 strs 中的字符串的的最大长度。需要用哈希表存储全部字符串
    """
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        strsDict = {}
        for s in strs:
            #这样得到的key是一个字符串（string），它是不可变的，因此可以作为字典的键。而列表（list）是可变（mutable）的，因此不可哈希
            key = ''.join(sorted(s))
            cur = strsDict.get(key, [])
            cur.append(s)
            strsDict[key] = cur #可改写成一行 strsDict.setdefault(key, []).append(s)， 与get(key, [])的区别是后者默认值不会修改原来的字典
        return list(strsDict.values())

    # 更优雅的写法：使用 collections.defaultdict, 自动处理键不存在的情况
    def groupAnagrams2(self, strs: List[str]) -> List[List[str]]:
        # 参数 list 指定了默认值工厂函数（factory function）当key不存在，自动传入一个空的list，比setdefault()方法效率高（因为 `setdefault` 每次调用都需要创建一个新的默认对象，即使键存在也会创建）
        # setdefault()更适用于需要动态确定默认值的场景
        strsDict = collections.defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            strsDict[key].append(s)
        return list(strsDict.values())

    """
    最长连续序列
    给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
    示例 1：
    输入：nums = [100,4,200,1,3,2]
    输出：4
    解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
    Solution：哈希表
    时间复杂度：O(n)，其中 n 为数组的长度
    空间复杂度：O(n)。哈希表存储数组中所有的数需要 O(n) 的空间。
    """
    def longestConsecutive(self, nums: List[int]) -> int:
        numsSet = set(nums)
        ans = 0
        if len(numsSet) <= 1:
            return len(numsSet)
        for num in numsSet:
            #序列长度
            length = 1
            #存在比当前数小的数，可直接跳过
            if num - 1 in numsSet:
                continue
            while (num + length) in numsSet:
                length += 1
            ans = max(length, ans)
        return ans

    """
    移动零
    给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序
    示例 1：
    输入: nums = [0,1,0,3,12]
    输出: [1,3,12,0,0]
    Solution：双指针
    使用双指针，左指针指向当前已经处理好的序列的尾部，右指针指向待处理序列的头部。
    右指针不断向右移动，每次右指针指向非零数，则将左右指针对应的数交换，同时左指针右移。
    注意到以下性质：
    左指针左边均为非零数；
    右指针左边直到左指针处均为零
    时间复杂度：O(n)，其中 n 为序列长度。每个位置至多被遍历两次
    空间复杂度：O(1)。只需要常数的空间存放若干变量。
    """
    def moveZeroes(self, nums: List[int]) -> None:
        n = len(nums)
        left = 0
        right = 0
        while right < n:
            #当当前数字为0，left不移动，使得left左边都是非0数，右边到右指针是0
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1
        print(nums)

    """
    盛最多水的容器
    给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
    找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
    返回容器可以储存的最大水量
    示例 1：
    输入: [1,8,6,2,5,4,8,3,7]
    输出: 49 
    Solution：双指针
    时间复杂度：O(N)，双指针总计最多遍历整个数组一次
    空间复杂度：O(1)，只需要额外的常数级别的空间
    """
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        max_area = 0 #最大的储水量
        while left < right:
            max_area = max(max_area, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area


    """
    三数之和
    给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，
    同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组
    示例 1：
    输入：nums = [-1,0,1,2,-1,-4]  [-4,-1,-1,0,1,2]
    输出：[[-1,-1,2],[-1,0,1]] 
    Solution：排序 + 双指针
    时间复杂度：O(N2)，其中 N 是数组 nums 的长度。
    空间复杂度：O(logN)。我们忽略存储答案的空间，额外的排序的空间复杂度为 O(logN)
    """
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n= len(nums)
        nums.sort()
        ans = list()
        if nums[0] > 0 or nums[-1] < 0:
            return []
        if len(nums) < 3:
            return []
        if len(nums) == 3:
            if sum(nums) == 0:
                return [nums]
            return []
        for first in range(n):
            target = -nums[first]
            third = n - 1
            if first > 0 and nums[first-1] == nums[first]:
                continue
            if nums[first] > 0 :
                break
            for second in range(first+1, n):
                if second > first + 1 and nums[second - 1] == nums[second]:
                    continue
                while third > second and nums[second] + nums[third] > target:
                    third -= 1
                if third == second:
                    break
                if nums[second] + nums[third] == target:
                    ans.append([nums[first], nums[second], nums[third]])
        return ans

if __name__ == '__main__':
    sol = HashTableAndDualPointer()
    # print(sol.lengthOfLongestSubstring("abcabcbb"))
    # print(sol.lengthAndStrOfLongestSubstring("abcabcbb"))
    # print(sol.lengthOfLongestSubstring("pwwkeww"))
    # print(sol.lengthAndStrOfLongestSubstring("pwwkeww"))
    # print(sol.twoSum([2,7,11,15],9))
    # print(sol.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    # print(sol.longestConsecutive([9,1,4,7,3,-1,0,5,8,-1,6]))
    # print(sol.moveZeroes([0,1,0,3,12]))
    print(sol.threeSum([-1,0,1,2,-1,-4] ))
