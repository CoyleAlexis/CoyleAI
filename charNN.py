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
block_size = 7
embedding_dim = 10
hidden_neurons = 200
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
Xtest, Ytext = build_dataset(words[nbWords2:]) #Other 10% of words

#Hidden layer
char_embeddings = torch.randn((vocab_size, embedding_dim))
weight1 = torch.randn((block_size * embedding_dim, hidden_neurons))
bias1 = torch.randn(hidden_neurons)

#Output layer
weight2 = torch.randn((hidden_neurons, vocab_size))
bias2 = torch.randn(vocab_size)

parameters = [char_embeddings, weight1, bias1, weight2, bias2]

for p in parameters:
    p.requires_grad = True #PyTorch needs gradients for loss.backward()

for itiration in range(200000):
    #Create mini batch of data from our dataset to optimize training. Less precise gradient but faster steps
    training_data = torch.randint(0, Xtr.shape[0], (32,))

    #Forward pass
    embedding = char_embeddings[Xtr[training_data]]
    hidden_activations = torch.tanh(embedding.flatten(start_dim=1) @ weight1 + bias1) # -1 makes pytorch automatically decided the dimension
    logits = hidden_activations @ weight2 + bias2
    loss = F.cross_entropy(logits, Ytr[training_data]) #More efficient and stable than calculating loss manually with the mean of -log

    #Backward pass 
    for p in parameters: 
        p.grad = None #zero grad
    loss.backward()

    learning_rate = 0.1 if itiration < 100000 else 0.01
    #Update
    for p in parameters:
        p.data += -learning_rate * p.grad


#General training loss
embedding = char_embeddings[Xtr] 
hidden_activations = torch.tanh(embedding.flatten(start_dim=1) @ weight1 + bias1) # -1 makes pytorch automatically decided the dimension
logits = hidden_activations @ weight2 + bias2
loss = F.cross_entropy(logits, Ytr)
print(f'General training loss: {loss.item()}')

#Dev loss
embedding = char_embeddings[Xdev] #Model trained on Xtr and Ytr so we test the loss on Xdev and Ydev, aka new data
hidden_activations = torch.tanh(embedding.flatten(start_dim=1) @ weight1 + bias1) # -1 makes pytorch automatically decided the dimension
logits = hidden_activations @ weight2 + bias2
loss = F.cross_entropy(logits, Ydev)
print(f'Dev loss: {loss.item()}')