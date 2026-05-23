import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        net = set()
        mx = 0

        for s in positive:
            loc = 0
            for w in s.split():
                net.add(w)
                loc += 1
            mx = max(mx, loc)
        
        for s in negative:
            loc = 0
            for w in s.split():
                net.add(w)
                loc += 1
            mx = max(mx, loc)

        tot = []
        for item in net:
            tot.append(item)
        tot.sort()

        idx = {}

        for i,w in enumerate(tot):
            idx[w] = i+1
        
        N = len(positive)
        
        answer = torch.zeros((2*N, mx))

        for i in range(N):
            ct = 0
            for j, w in enumerate(positive[i].split()):
                ct += 1
                answer[i][j] = idx[w]

            for j in range(mx - ct):
                answer[i][ct + j] = 0

        for i in range(N):
            ct = 0

            for j, w in enumerate(negative[i].split()):
                ct += 1
                answer[i+N][j] = idx[w]
            
            for j in range(mx - ct):
                answer[i+N][ct + j] = 0

        return answer
