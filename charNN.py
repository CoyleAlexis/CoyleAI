import torch
import torch.nn.functional as F
import random

words = open('names.txt').read().splitlines()

#Create a lookup table for letters to int value 
chars = sorted(list(set(''.join(words)))) #Removes duplicate and makes all the words into a single array
stoi = {s:i+1 for i,s in enumerate(chars)} #string to int 
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} #reverse mapping of stoi (int to string)

#Settings
block_size = 4
embedding_dim = 35
hidden_neurons = 350
vocab_size = len(stoi)

def build_dataset(words):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for char in word + '.':
            index = stoi[char]
            X.append(context)
            Y.append(index)
            context = context[1:] + [index] #crop and append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
nbWords1 = int(0.8*len(words))
nbWords2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:nbWords1]) #80% of words
Xdev, Ydev = build_dataset(words[nbWords1:nbWords2]) #10% of words
Xtest, Ytest = build_dataset(words[nbWords2:]) #Other 10% of words

#Hidden layer
char_embeddings = torch.randn((vocab_size, embedding_dim))
weight1 = torch.randn((block_size * embedding_dim, hidden_neurons)) *0.2
bias1 = torch.randn(hidden_neurons) * 0.01

#Output layer
weight2 = torch.randn((hidden_neurons, vocab_size)) * 0.01
bias2 = torch.randn(vocab_size) * 0 #Zero initially

parameters = [char_embeddings, weight1, bias1, weight2, bias2]

for p in parameters:
    p.requires_grad = True #PyTorch needs gradients for loss.backward()

for itiration in range(200000):
    #Create mini batch of data from our dataset to optimize training. Less precise gradient but faster steps
    training_data = torch.randint(0, Xtr.shape[0], (32,))

    #Forward pass
    embedding = char_embeddings[Xtr[training_data]]
    embedding_cat = embedding.view(embedding.shape[0], -1)
    hidden_activations = torch.tanh(embedding_cat.view(embedding.shape[0], -1) @ weight1 + bias1) # -1 makes pytorch automatically decided the dimension
    logits = hidden_activations @ weight2 + bias2
    loss = F.cross_entropy(logits, Ytr[training_data]) #More efficient and stable than calculating loss manually with the mean of -log

    #Backward pass 
    for p in parameters: 
        p.grad = None #zero grad
    loss.backward()

    learning_rate = 0.1 if itiration < 100000 else 0.02
    #Update
    for p in parameters:
        p.data += -learning_rate * p.grad
    if itiration % 10000 == 0:
        print(f'Loss at itiration {itiration}/200000: {loss}')

#Create 20 words
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        embedding = char_embeddings[torch.tensor([context])]
        hidden_activations = torch.tanh(embedding.view(1, -1) @ weight1 + bias1)
        logits = hidden_activations @ weight2 + bias2
        probs = F.softmax(logits, dim=1)

        #Sample from dataset
        index = torch.multinomial(probs, num_samples=1).item()

        #Shift context window and track sample
        context = context[1:] + [index]
        out.append(index)

        #if we get '.' we end
        if index == 0:
            break
    print(''.join(itos[i] for i in out))

@torch.no_grad()
def split_loss(split):
    X, Y = {
        'train' : (Xtr, Ytr),
        'val' : (Xdev, Ydev),
        'test' : (Xtest, Ytest)
    }[split]
    embedding = char_embeddings[X]
    embedding_cat = embedding.view(embedding.shape[0], -1)
    hidden_activations = torch.tanh(embedding_cat @ weight1 + bias1)
    logits = hidden_activations @ weight2 + bias2
    loss = F.cross_entropy(logits, Y)
    print(f'Loss {split}: {loss}')
    
split_loss('train')
split_loss('val')