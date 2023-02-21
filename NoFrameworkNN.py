import numpy as np
#---------------------------------------------------------------------
#We will use this to aply it like a relu activation function on our NN
def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
def cross_entropy(y, y_hat):
    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return np.mean(loss)
#---------------------------------------------------------------------
class NeuralNetwork:
#---------------------------------------------------------------------
#Here we put input, hidden and output variables from the structure of the NN. 
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
#---------------------------------------------------------------------
# #We put too, the weight and the sesgos using numpy to set them random.    
        self.weights1 = np.random.normal(size=(input_size, hidden_size))
        self.sesgos1 = np.zeros(hidden_size)
        self.weights2 = np.random.normal(size=(hidden_size, output_size))
        self.sesgos2 = np.zeros(output_size)
#---------------------------------------------------------------------
#This makes the propagation forwards on the NN, making complex maths(in my opinion xd).
    def forward(self, x):
#First we multiply the inputs(x) with the wights from first layer(weights1) and then we plus the sesgos from the same layer(sesgos1).
        self.a1 = np.dot(x, self.weights1) + self.sesgos1
#Now we use relue function and set it on a1
        self.b1 = relu(self.a1)
#We repeat it with the second layer and the first layer
        self.a2 = np.dot(self.b1, self.weights2) + self.sesgos2
#Another time we use the relue function and set it on a2
        self.b2 = relu(self.a2)
#This is the output
        return self.b2
#---------------------------------------------------------------------
#This is the famous "BACKPROPAGATION" with couple complex maths too(all math is hard, OK?!)
#We calculate the error rate, multipling the difference from the wanted output and the output that we have(y_hat - y) with the relu derivation function(relu_deriv) on the output layer.
    def backward(self, x, y, y_hat, learning_rate):
        delta2 = (y_hat - y) * relu_deriv(self.a2)
        d_weights2 = np.dot(self.b1.T, delta2)
        d_sesgos2 = np.sum(delta2, axis=0)
#Here we calculate the error rate from de hidden layer by multipling the error from the output layer with the weights from the output layer and with the relu derivation function(relu_deriv) on the output layer.
        delta1 = np.dot(delta2, self.weights2.T) * relu_deriv(self.a1)
        d_weights1 = np.dot(x.T, delta1)
        d_sesgos1 = np.sum(delta1, axis=0)
#Now we change the weights and the sesgos with the new data multiplied with the learning rate
        self.weights2 -= learning_rate * d_weights2
        self.sesgos2 -= learning_rate * d_sesgos2
        self.weights1 -= learning_rate * d_weights1
        self.sesgos1 -= learning_rate * d_sesgos1
#---------------------------------------------------------------------
#We first make a normal propagation with forward function and then the backpropagation with the backward
    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            y_hat = self.forward(X)
            loss = cross_entropy(y, y_hat)
            self.backward(X, y, y_hat, learning_rate)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return np.round(self.forward(X))