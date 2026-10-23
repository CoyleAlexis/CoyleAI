import math
import random

class Value:
    def __init__(self,data, _children = (), _op='',):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
   
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self,other): #Allows int + Value to work
        return self + other
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += (other.data * out.grad)
            other.grad += (self.data * out.grad)
        out._backward = _backward
        return out

    def __pow__(self,other):
        assert isinstance(other, (int, float)) #Only supports int and float for now
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self,other): #Allows int * Value to work
        return self*other

    def __truediv__(self,other): #self / other
        return self * other**-1 #This way covers more cases than just doing self/other

    def __sub__(self,other):
        return self + -(other)
    
    def tanh(self):
        x = self.data
        t =math.tanh(x)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        #Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
            topo.append(v)
        build_topo(self)

        #Backprop for every node
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

#Neural net using Value class

class Neuron: 
    def __init__(self, nInput):
        self.weight = [Value(random.uniform(-1,1)) for _ in range(nInput)]
        self.bias = Value(random.uniform(-1,1))
    def __call__(self,x):
        activation = sum((wi*xi for wi , xi in zip(self.weight, x)), self.bias)
        out = activation.tanh()
        return out
    def parameters(self):
        return self.weight + [self.bias]

class Layer:
    def __init__(self, nInput, nOutput):
        self.neurons = [Neuron(nInput=nInput)for _ in range(nOutput)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        #Returns each weights and biases for every neuron of the layer in one array
        return [params for neuron in self.neurons for params in neuron.parameters()]

class MLP: #Multi-Layers
    def __init__(self, nInputs, nOutputs): 
        size = [nInputs] + nOutputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(nOutputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        #Returns all weights and biases of all layers (basically of the entire network)
        return [params for layer in self.layers for params in layer.parameters()]

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])

dataset = [[2.0, 3.0, -1.0],
           [3.0, -1.0, 0.5], 
           [0.5, 1.0, 1.0], 
           [1.0, 1.0, -1.0]]

expected = [1.0, -1.0, -1.0, 1.0]

#Training loop
for iteration in range(10):
    #Forward pass
    prediction = [n(data) for data in dataset]
    loss = sum((yOutput - groundTruth)**2 for groundTruth, yOutput in zip(expected, prediction))
    
    #Backward pass
    for param in n.parameters():
        param.grad = 0
    loss.backward()

    #Update
    for param in n.parameters():
        param.data += -0.05821 * param.grad #Without the - we will just increase the loss rather than decrease it

    #Results for every iteration
    print(f'Loss {iteration}: {loss.data}')
    print(f'Prediction {iteration}: {prediction}')
    