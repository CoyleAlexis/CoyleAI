import torch.nn as nn
from torch.nn import functional as F
import torch
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #Each token reads off the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        logits = self.token_embedding_table(inputs)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_token):
        for _ in range(max_new_token):
            #Prediction
            logits, loss = self(index)
            logits = logits[:, -1, :]

            #Softmax for probabilities
            probs = F.softmax(logits, dim=-1)

            #Sample from distribution
            index_next = torch.multinomial(probs, num_samples=1)

            #Append sampled to current sequence
            index = torch.cat((index, index_next), dim=1)
        return index
    
