from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        ans = []

        for n in numbers:
            loc = []

            s = str(n)
            j = 0

            while j < len(s):
                jmp = -1
                for k in range(len(s) - 1, j, -1):
                    if (s[j:k+1] in vocab):
                        jmp = k
                        break
                if jmp != -1:
                    loc.append(s[j:(jmp + 1)])
                    j = jmp + 1
                else:
                    loc.append(s[j])
                    j += 1

            ans.append(loc)

        return ans

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        s = text
        loc = []

        
        j = 0

        while j < len(s):
            jmp = -1
            for k in range(len(s) - 1, j, -1):
                if (s[j:k+1] in vocab):
                    jmp = k
                    break
            if jmp != -1:
                loc.append(s[j:(jmp + 1)])
                j = jmp + 1
            else:
                loc.append(s[j])
                j += 1

        return len(loc)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        sl = text.split()
        return round(self.count_tokens(text, vocab) / len(sl), 4)
        
