from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        chars = []
        
        for i in range(len(corpus)):
            chars.append(corpus[i])
        
        ans   = []
        
        for i in range(num_merges):
            freq = {}
            mx_f = 0
            s = ''
            first = ''
            second = ''
            print(chars)

            for j in range(len(chars) - 1):
                pair = (chars[j],  chars[j+1])
                if pair in freq:
                    freq[pair] += 1
                else:
                    freq[pair] = 1
                mx_f = max(mx_f, freq[pair])

                if freq[pair] == mx_f:
                    s = pair

            # print(freq, s)

            for k, v in freq.items():
                if v == mx_f:
                    # print("Here ", k, s, (k < s))
                    if (k < s):
                        s = k
                        
            
            # merge every occurrence of s
            prev = False
            new_chars = []
            for j in range(len(chars)):
                if prev:
                    prev = False
                    continue

                if j != len(chars) - 1 and (chars[j], chars[j+1]) == s:
                    prev = True
                    new_chars.append(chars[j] + chars[j+1])
                    first = chars[j]
                    second = chars[j+1]

                else:
                    new_chars.append(chars[j])

            chars = []
            for c in new_chars:
                chars.append(c)
            current_merge = [first, second]
            ans.append(current_merge)
            

        return ans
