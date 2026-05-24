from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = []
        n = len(text)

        for i in range(n):
            chars.append(text[i])

        chars.sort()

        self.stoi = {}
        self.itos = {}
        ctr = 0

        for i in range(len(chars)):
            if i == 0 or chars[i] != chars[i-1]:
                self.stoi[chars[i]] = ctr
                self.itos[ctr] = chars[i]
                ctr += 1


        return (self.stoi, self.itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        ans = []
        
        for i in range(len(text)):
            ans.append(stoi[text[i]])

        return ans

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        ans = ''

        for i in range(len(ids)):
            ans += itos[ids[i]]

        return ans
