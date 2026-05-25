import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        D = len(data)
        op = torch.optim.AdamW(model.parameters(), lr = lr)

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            X = torch.zeros((batch_size, context_length), dtype = torch.int64)
            Y = torch.zeros((batch_size, context_length), dtype = torch.int64)

            rand_ints = torch.randint(0, D - context_length, (batch_size,))

            for j in range(batch_size):
                X[j] = data[rand_ints[j] : rand_ints[j] + context_length]
                Y[j] = data[rand_ints[j] + 1 : rand_ints[j] + 1 + context_length]
            

            logits = model(X)
            
            logits = logits.reshape((batch_size * context_length,-1))
            Y = Y.reshape((batch_size * context_length, ))

            loss = F.cross_entropy(logits, Y)

            op.zero_grad()
            loss.backward()
            op.step()


        return round(loss.item(), 4)