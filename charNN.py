import torch
import torch.nn.functional as F
words = open('names.txt').read().splitlines()
N = torch.zeros((27,27), dtype=torch.int32)

#Create a lookup table for letters to int value 
chars = sorted(list(set(''.join(words)))) #Removes duplicate and makes all the words into a single array
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

for word in words:
    chars = ['.'] + list(word) + ['.']
    for char1, char2 in zip(chars, chars[1:]):
        ix1 = stoi[char1]
        ix2 = stoi [char2]
        N[ix1,ix2] += 1

itos = {i:s for s,i in stoi.items()} #reverse mapping of stoi

probMatrix = (N + 1).float()
probMatrix = probMatrix / probMatrix.sum(1, keepdim = True)
generator = torch.Generator().manual_seed(2147483647)

for i in range(10):
    out =[]
    index = 0 
    while True:
        probability = probMatrix[index]
        index = torch.multinomial(probability, num_samples=1, replacement=True, generator = generator).item()
        out.append(itos[index])
        if index == 0:
            break
    print(''.join(out))

#Calculating loss
log_likelihood = 0.0
n = 0
for word in words:
    chars = ['.'] + list(word) + ['.']
    for char1, char2 in zip(chars, chars[1:]):
        ix1 = stoi[char1]
        ix2 = stoi [char2]
        prob = probMatrix[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
negLob_likelihood = -log_likelihood
print(f'Loss: {negLob_likelihood/n}')


#Create training set of bigrams
xs, ys = [], []

for word in words:
    chars = ['.'] + list(word) + ['.']

    for char1, char2 in zip (chars, chars[1:]):
        index1 = stoi[char1]
        index2 = stoi[char2]
        xs.append(index1)
        ys.append(index2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

F.one_hot(xs,num_classes = 27)