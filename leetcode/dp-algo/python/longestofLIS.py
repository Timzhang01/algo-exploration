from ast import List
from os import name

class LongestOfLIS:
    # 最长递增子序列
    def LIS(self, nums: List[int]) -> int:
        nums_len = len(nums)
        dp = [1] * nums_len
        max_len = 0
        for i in range(nums_len):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
            max_len = max(max_len, dp[i])
        return max_len
    
    # 最长递增子序列2
    def LIS2(self, nums: List[int], k: int) -> int:
        nums_len = len(nums)
        dp = [1] * nums_len
        max_len = 0
        for i in range(nums_len):
            for j in range(i):
                if nums[i] > nums[j] and nums[i] - nums[j] <= k:
                    dp[i] = max(dp[i], dp[j] + 1)
            max_len = max(max_len, dp[i])
        return max_len
    
    # 线段树
    # f[i][j] 表示前i个元素，以元素为j结尾的的最长子序列的长度
    # j != nums[i], f[i][j] = f[i-1][j]
    # j == nums[i], f[i][j] = max(f[i-1][j']) + 1, j-k<=j'<=j-1
    # 上述公式去掉第一个维度
    # j == nums[i]， f[j] = max(f[j']) + 1, j-k<=j'<=j-1
    def lengthOfLIS2(self, nums: List[int], k: int) -> int:
        max_value = max(nums)
        mx = [0] * (4 * max_value)
    

        # mx[i] = val
        def modify(o, l, r, i, val):
            if l == r:
                mx[o] = val
                return
            
            m = (l + r) // 2
            if i <= m: modify(o * 2, l, m, i, val)
            else: modify(o * 2 + 1, m+1, r, i, val)
            mx[o] = max(mx[o * 2], mx[o * 2 + 1])

        # 查询L,R区间的值
        def query(o, l, r, L, R):
            if L <= l and r <= R:
                return mx[o]
            res = 0
            m = (l + r) // 2
            # 注意这块不是if else 因为左右区间都需要查询
            if L <= m: res = query(o * 2, l, m, L, R)
            if R > m: res = max(res, query(o * 2 + 1, m + 1, r, L, R))
            return res
        
        for x in nums:
            if x == 1:
                modify(1, 1, max_value, 1, 1)
            else:
                res = 1 + query(1, 1, max_value, max(x - k, 1), x - 1)
                modify(1, 1, max_value, x, res)
        
        return mx[1]
            
    

if __name__ == "__main__":
    lofLIS = LongestOfLIS()
    nums = [4,2,1,4,3,4,5,8,15]
    print(f"test LIS ans is {lofLIS.lengthOfLIS2(nums=nums, k=3   )}")
        
