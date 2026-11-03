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

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        #Params trained with backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        #Buffers trained with a momentum update
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        #Calculate forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) #Normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        #Update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum)  * self.running_var + self.momentum * xvar
        return self.out
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
def forward(X):
    embedding = char_embeddings[X]
    flattened_embedding = embedding.view(embedding.shape[0], -1)

    output = flattened_embedding

    for layer in layers:
        output = layer(output)

    return output
layers = [
    Linear(embedding_dim * block_size, hidden_neurons), BatchNorm1d(hidden_neurons), Tanh(),
    Linear(            hidden_neurons, hidden_neurons), BatchNorm1d(hidden_neurons), Tanh(),
    Linear(            hidden_neurons, hidden_neurons), BatchNorm1d(hidden_neurons), Tanh(),
    Linear(            hidden_neurons, hidden_neurons), BatchNorm1d(hidden_neurons), Tanh(),
    Linear(            hidden_neurons, hidden_neurons), BatchNorm1d(hidden_neurons), Tanh(),
    Linear(            hidden_neurons, vocab_size), BatchNorm1d(vocab_size)
]

with torch.no_grad():
    layers[-1].gamma *= 0.1 #If last layer is BatchNorm1d we make gamma less confident if its Linear then its the weight
    for layer in layers[:-1]:
        if isinstance(layer,Linear):
            layer.weight *= 5/3


parameters = [char_embeddings] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True #PyTorch needs gradients for loss.backward()

nb_steps = 200000
batch_size = 32
for itiration in range(nb_steps):
    #Create mini batch of data from our dataset to optimize training. Less precise gradient but faster steps
    training_data = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xbatch, Ybatch = Xtr[training_data], Ytr[training_data]

    #Forward pass
    logits = forward(Xbatch)
    loss = F.cross_entropy(logits, Ybatch)

    #Backward pass 
    for p in parameters:
        p.grad = None
    loss.backward()

    learning_rate = 0.1 if itiration < 100000 else 0.02
    #Update
    for p in parameters:
        p.data += -learning_rate * p.grad
    if itiration % 10000 == 0:
        print(f'Loss at itiration {itiration}/{nb_steps}: {loss.item():.4f}')

#Change model mode
for layer in layers:
    if isinstance(layer,BatchNorm1d):
        layer.training = False

#Create 20 words
for _ in range(100):
    out = []
    context = [0] * block_size
    while True:
        input = torch.tensor([context])

        logits = forward(input)
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

    #Eval mode
    for layer in layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False

    logits = forward(X)
    
    loss = F.cross_entropy(logits, Y)  #More efficient and stable than calculating loss manually with the mean of -log
    print(f'Loss {split}: {loss}')
    
split_loss('train')
split_loss('val')