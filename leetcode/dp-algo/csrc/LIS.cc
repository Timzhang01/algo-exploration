#include <stdio.h>
#include <iostream>
#include <vector>

using namespace std;

class LIS{
 public:
    int longestIncSeq(vector<int>& nums) {
        // dp[i] = max(dp[j] + 1, dp[i]) if nums[i] > nums[j]
        if (nums.empty() || nums.size() <= 1) {
            return nums.size();
        }
        int len = nums.size();
        vector<int> dp(len, 1);
        int maxLen = 1;
        for (int i = 1; i < len; i ++) {
            for (int j = 0; j < i; j ++) {
                if (nums[i] > nums[j]) {
                    dp[i] = max(dp[j] + 1, dp[i]);
                }
            }
            maxLen = max(maxLen, dp[i]);
        }
        return maxLen;
    }

    int longestIncSeqNum(vector<int>& nums) {
        if (nums.empty() || nums.size() <= 1) {
            return nums.size();
        }
        int ans = 0;
        int max_len = 1;
        vector<int> dp(nums.size(), 1);
        vector<int> cnt(nums.size(), 1);
        for (int i = 1; i < nums.size(); i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        cnt[i] = cnt[j];
                    } else if (dp[j] + 1 == dp[i]) {
                        cnt[i] += cnt[j];
                    }
                }
            }
            if (dp[i] > max_len) {
                max_len = dp[i];
                ans = cnt[i];
            } else if (dp[i] == max_len) {
                ans += cnt[i];
            }
        }
        return ans;
    }

    vector<int> mx;

    void modify(int o, int l, int r, int i, int val) {
        if (l == r) {
            mx[o] = val;
            return;
        }

        int mid = l + (r - l) / 2;
        // 这里是dp[i] = val
        if (i <= mid) {
            modify(o * 2, l, mid, i, val);
        } else {
            modify(o * 2 + 1, mid + 1, r, i, val);
        }
        mx[o] = max(mx[o * 2], mx[o * 2 + 1]);
    }

    int query(int o, int l, int r, int L, int R) {
        // 这个范围也要注意，左闭右开
        if (L <= l && r <= R) {
            return mx[o];
        }
        int res = 0;
        int mid = l + (r - l) / 2;
        if (L <= mid) {
            res = query(o * 2, l, mid, L, R);
        } 
        if (R > mid) {
            res = max(res, query(o * 2 + 1, mid + 1, r, L, R));
        }
        return res;
    }
    
     int longestIncSeq2(vector<int>& nums, int k) {
        int max_val = 0;
        for (int i = 0; i < nums.size(); i ++) {
            max_val = max(max_val, nums[i]);
        }
        mx.resize(max_val * 4);
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == 1) {
                modify(1, 1, max_val, 1, 1);
            } else {
                int res = 1 + query(1, 1, max_val, max(nums[i] - k, 1), nums[i] - 1);
                modify(1, 1, max_val, nums[i], res);
            }
        }
        return mx[1];
     }


};



int main() {
    LIS test;
    vector<int> array{1,4,3,50,3,3};
    int ans = test.longestIncSeq2(array, 2);
    std::cout << "LIS ans:" << ans << std::endl;
}