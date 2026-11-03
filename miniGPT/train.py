import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from model import BigramLanguageModel

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

block_size = 8
batch_size = 32
train_model = True
nb_train_iter = 1000
eval_interval = 100
eval_iter = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
model = BigramLanguageModel(vocab_size)
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

@torch.no_grad
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
    model.train()
 
    #Create PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for iter in range(nb_train_iter):
        if iter % nb_train_iter == 0 :
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
    generated = model.generate(context, max_new_token=100)

    print(decode(generated[0].tolist()))
