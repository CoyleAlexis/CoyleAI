import numpy as np 
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)

        self.dbiases = np.sum(
            dvalues,
            axis=0,
            keepdims=True
        )

        self.dinputs = np.dot(
            dvalues,
            self.weights.T
        )
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        self.inputs = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        probalities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probalities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for sample in (range(len(self.output))):
            current_output = self.output[sample]
            jacobian = np.diag(current_output) - np.outer(current_output,current_output)
            self.dinputs[sample] = np.dot(dvalues[sample], jacobian) 

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
        def forward(self,y_pred, y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

            #Scalar
            if len(y_true.shape) == 1:
                correct_confidence = y_pred_clipped[range(samples), y_true]

            #One-hot encoded
            elif len(y_true.shape) == 2:
                correct_confidence = np.sum(y_pred_clipped*y_true, axis=1)
           
            negative_log_likelihood = -np.log(correct_confidence)
            return negative_log_likelihood
        def backward(self, dvalues, y_true):
            samples = len(dvalues)
            predictions = np.clip(dvalues, 1e-7, 1-1e-7)
            if len(y_true.shape) == 1:
                one_hot = np.zeros(dvalues.shape)

                for sample in range(samples):
                    one_hot[sample, y_true[sample]] = 1
                gradient = (-one_hot/predictions)/samples

                self.dinputs = gradient
                
class Optimizer_SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_params(self,layer):
        layer.weights += \
            -self.learning_rate * layer.dweights
        layer.biases += \
            -self.learning_rate * layer.dbiases

X, y = spiral_data(100,3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_SoftMax()
loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=1)

for epoch in range (10000):
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    #Backprop
    loss_function.backward(activation2.output,y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
predictions = np.argmax(activation2.output, axis=1)

accuracy = np.mean(predictions == y)

print("Accuracy:", accuracy)