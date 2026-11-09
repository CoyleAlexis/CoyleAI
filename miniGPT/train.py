import torch
import torch.nn as nn
from torch.nn import functional as F
import os

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {char:integer for integer,char in enumerate(chars)}
itos = {integer:char for integer,char in enumerate(chars)}
encode = lambda text: [stoi[char] for char in text] #In: string -> Out: list of int
decode = lambda list_int: ''.join([itos[integer] for integer in list_int]) #In: list of int -> Out:

data = torch.tensor(encode(text), dtype = torch.long)

train_quantity = int(0.9 * len(data))
train_data = data[:train_quantity]
val_data = data[train_quantity:]

learning_rate = 3e-4
block_size = 256
batch_size = 64
train_model = True
nb_train_iter = 1000
eval_interval = 200
eval_iter = 50
num_embeddings = 384 #Needs to be divisible by num_head
num_head = 6
num_layer = 6
dropout = 0.2
#Check if GPU is available and supported otherwise uses cpu
device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability(0) == 1) else 'cpu'

class Head(nn.Module):
    #Single head attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        B,T,C = input.shape
        key = self.key(input)
        query = self.query(input)

        #Attention scores
        weight = query @ key.transpose(-2, -1) * key.shape[1]**-0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        #Weights aggregation of values
        value = self.value(input)
        out = weight @ value
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_embeddings, num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        out = torch.cat([h(input) for h in self.heads], dim=-1)
        out = self.projection(out)
        return(out)
    
class FeedForward(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings), 
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self , num_embeddings, num_head):
        super().__init__()
        head_size = num_embeddings // num_head
        self.self_attention = MultiHeadAttention(num_head, head_size)
        self.feed_forward = FeedForward(num_embeddings)
        self.layer_norm1 = nn.LayerNorm(num_embeddings)
        self.layer_norm2 = nn.LayerNorm(num_embeddings)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Each token reads off the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
        self.blocks = nn.Sequential(*[Block(num_embeddings, num_head=num_head) for _ in range(num_layer)])
        self.layer_norm_finish = nn.LayerNorm(num_embeddings)
        self.lang_model_head = nn.Linear(num_embeddings, vocab_size)

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        token_embeddings = self.token_embedding_table(inputs)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        head_input = token_embeddings + position_embeddings #(B, T, C)
        head_input = self.blocks(head_input)
        head_input = self.layer_norm_finish(head_input)
        logits = self.lang_model_head(head_input)


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
            #Crop index so it fits block_size
            index_cropped = index[:, -block_size:]

            #Prediction
            logits, loss = self(index_cropped)
            logits = logits[:, -1, :]

            #Softmax for probabilities
            probs = F.softmax(logits, dim=-1)

            #Sample from distribution
            index_next = torch.multinomial(probs, num_samples=1)

            #Append sampled to current sequence
            index = torch.cat((index, index_next), dim=1)
        return index


model = BigramLanguageModel()
model = model.to(device)

def get_batch(split):
    #Creates small batch of data with inputs (x) and targets (y)
    data = train_data if split == "train" else val_data
    index = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in index])
    y = torch.stack([data[i+1:i+block_size+1] for i in index])
    x, y = x.to(device), y.to(device)
    return x,y

xb, yb = get_batch("train")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#Load saved params if there is any
if os.path.exists('miniGPT.pth'):
    model.load_state_dict(torch.load('miniGPT.pth', weights_only=True))
    print("Saved model loaded!")
else:
    print("No saved model found. Starting from scratch.")
    train_model=True

logits, loss = model(xb,yb)


#Optimize
if train_model:
    print(f"Used device: {device}") #Prints used device for training
    model.train()
 
    #Create PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(nb_train_iter):

        if iter % eval_interval == 0 :
            losses = estimate_loss()
            print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}" )

        xb, yb = get_batch("train")
        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    #Save after training
    torch.save(model.state_dict(), 'miniGPT.pth')
    print(f"Final training loss: {loss.item():.4f}")

#Generate text
model.eval()
with torch.no_grad():
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_token=1000)

    print(decode(generated[0].tolist()))
