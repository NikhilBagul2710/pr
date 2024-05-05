

































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#write a python program to plot a few activation functions that are being used in neural networks

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def tanh(x):
    return np.tanh(x)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
# Create x values
x = np.linspace(-10, 10, 100)    #[]

# Create plots for each activation function 
fig, axs = plt.subplots(2, 2, figsize=(8, 8)) 

axs[0, 0].plot(x, sigmoid(x))
axs[0, 0].set_title('Sigmoid') 
axs[0, 1].plot(x, relu(x))
axs[0, 1].set_title('ReLU')
axs[1, 0].plot(x, tanh(x))
axs[1, 0].set_title('Tanh')
axs[1, 1].plot(x, softmax(x)) 
axs[1, 1].set_title('Softmax')

# Add common axis labels and titles 
fig.suptitle('Common Activation Functions')

# Show the plot 
plt.show()

000000000000000000000000000000000000000000000000000000000000000000000000000

#Generate ANDNOT function using McCulloch-Pitts neural net by a python program. 

import numpy as np

# function of checking thresold value 
def linear_threshold_gate(dot, T):
    if dot >= T:
        return 1 
    else:
        return 0

# matrix of inputs 
input_table = np.array([
    [0,0], # both no
    [2,1], # one no, one yes 
    [1,0], # one yes, one no 
    [1,1] # bot yes
])
weights = np.array([1,-1])
T = 1

dot_products_sum = input_table @ weights

print(f'input table:\n{input_table}') 
print(f'dot products:\n{dot_products_sum}')

for i in range(0,4):
    activation = linear_threshold_gate(dot_products_sum[i], T) 
    print(f'Activation: {activation}')

111111111111111111111111111111111111111111111111111111111111111111111111111111111111

import numpy as np

j = int(input("Enter a Number (0-9): "))
step_function = lambda x: 1 if x >= 0 else 0

training_data = [
    {'input': [1, 1, 0, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 0, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 0, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 1, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 1, 0, 0, 1], 'label': 0},
]

weights = np.array([0, 0, 0, 0, 0, 1])

for data in training_data:
    input = np.array(data['input'])
    label = data['label']
    output = step_function(np.dot(input, weights))
    error = label - output
    weights += input * error

input = np.array([int(x) for x in list('{0:06b}'.format(j))])
output = "odd" if step_function(np.dot(input, weights)) == 0 else "even"
print(j, " is ", output)

-------------------------------------------------------------

import numpy as np

# Training data
training_inputs = np.array([
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],  # 0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 1
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],  # 2
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 3
    [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],  # 4
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],  # 5
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 1],  # 6
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 7
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 8
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1]   # 9
])

training_labels = np.array([
    [1],  # Even
    [0],  # Odd
    [0],  # Odd
    [1],  # Even
    [0],  # Odd
    [0],  # Odd
    [0],  # Odd
    [1],  # Even
    [0],  # Odd
    [0]   # Odd
])

# Perceptron Neural Network class
class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.zeros((num_inputs, 1))
        self.bias = 0

    def train(self, inputs, labels, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            for input_data, label in zip(inputs, labels):
                prediction = self.predict(input_data)
                error = label - prediction

                self.weights += learning_rate * error * input_data.reshape(-1, 1)
                self.bias += learning_rate * error

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation >= 0 else 0

# Training the perceptron
perceptron = Perceptron(num_inputs=10)
perceptron.train(training_inputs, training_labels, num_epochs=100, learning_rate=0.1)

# Testing the perceptron
test_inputs = np.array([
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],  # 0 (Even)
    [0, 0, 1, 0, 1, 0, 0, 1, 1, 1],  # 9 (Odd)
    [0, 1, 0, 0, 1, 0, 1, 1, 1, 0],  # 6 (Odd)
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 0]   # 8 (Even)
])

for input_data in test_inputs:
    prediction = perceptron.predict(input_data)
    number = ''.join(map(str, input_data.tolist()))

    if prediction == 1:
        print(f"{number} is even.")
    else:
        print(f"{number} is odd.")

2222222222222222222222222222222222222222222222222222222222222222222222222222

# Write a suitable example to demonstrate the perceptron learning law with its decision regions
# using python. Give the output in graphical form. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# load iris dataset 
iris = load_iris()

# extract sepal length and petal length features 
X = iris.data[:, [0, 2]]
y = iris.target
w = np.zeros(2) #[0.0 , 0.0]  # to give weightage
b = 0                     # to shift the decision boundary
lr = 0.1        # learning rate
epochs = 50

# setosa is class 0, versicolor is class 1 
y = np.where(y == 0, 0, 1)

# define perceptron function 
def perceptron(x, w, b):
    # calculate weighted sum of inputs 
    z = np.dot(x, w) + b
    # apply step function
    return np.where(z >= 0, 1, 0)

# train the perceptron
for epoch in range(epochs): 
    for i in range(len(X)):
        x = X[i] 
        target = y[i]
        output = perceptron(x, w, b) 
        error = target - output
        # the algorithm assigns more weight to features that contribute more to the error.
        w += lr * error * x
        # The bias affects the decision boundary's position and allows for shifting it 
        # up or down without changing the weights.
        b += lr * error

#model train 
# plot decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5 
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = perceptron(np.c_[xx.ravel(), yy.ravel()], w, b)
Z = Z.reshape(xx.shape)

#plot decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Petal length') 
plt.title('Perceptron decision regions') 
plt.show()

3333333333333333333333333333333333333333333333333333333333333333333333333333333

#Write a python program for bidirectional associative memory with two pairs of vectors

import numpy as np

# define two pairs of vectors 
x1 = np.array([1, 1, 1, -1])
y1 = np.array([1, -1])
x2 = np.array([-1, -1, 1, 1]) 
y2 = np.array([-1, 1])

# compute weight matrix W
W = np.outer(y1, x1) + np.outer(y2, x2)

# define BAM function 
def bam(x):
    y = W @ x
    return np.where(y>=0, 1, -1)

# test BAM with inputs
x_test = np.array([1, -1, -1, -1]) 
y_test = bam(x_test)

# print output 
print("Input x: ", x_test) 
print("Output y: ", y_test)

4444444444444444444444444444444444444444444444444444444444444444444444444444444

#Implement Artificial Neural Network training process in Python by using forward propagation,
#backpropagation.

# error = target - output
# delta = error * sig_derivative (output of that layer)
# w = w + transpose_input @ that_layer_delta
# b = b + sum of columns of delta of that layer
import numpy as np  
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size): 
        self.W1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) 
        self.b2 = np.zeros(output_size)
 
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x): 
        return x * (1 - x)

    def forward(self, X): 
        self.z1 = X @ self.W1 + self.b1 
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2 
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output): 
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output) 
        self.hidden_error = self.output_delta @ self.W2.T
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1) 
        self.W1 += X.T @ self.hidden_delta
        self.b1 += np.sum(self.hidden_delta, axis=0)
        self.W2 += self.a1.T @ self.output_delta
        self.b2 += np.sum(self.output_delta, axis=0) 
        
    def train(self, X, y, epochs): 
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output) 

    def predict(self, X):
        # Make predictions for a given set of inputs 
        return self.forward(X)

# Define the input and output datasets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network with 2 input neurons, 4 neurons in the hidden layer, and 1 output neuron
nn = NeuralNetwork(2, 4, 1)

# Train the neural network on the input and output datasets for 10000 epochs with a learning rate of 0.1
nn.train(X, y, epochs=10000)

# Use the trained neural network to make predictions on the same input dataset 
predictions = nn.predict(X)

# Print the predictions 
print(predictions)

------------------------------------------------------------
import numpy as np

# Step 1: Define the sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Step 2: Define the training function for the neural network
def train_neural_network(X, y, learning_rate, epochs):
    # Step 3: Initialize the weights and biases with random values
    input_neurons = X.shape[1]
    hidden_neurons = 4
    output_neurons = y.shape[1]
    
    hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
    hidden_bias = np.random.uniform(size=(1, hidden_neurons))
    output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
    output_bias = np.random.uniform(size=(1, output_neurons))
    
    # Step 4: Perform the training iterations
    for i in range(epochs):
        # Step 4.1: Forward propagation
        hidden_layer_activation = np.dot(X, hidden_weights) + hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
        predicted_output = sigmoid(output_layer_activation)

        # Step 4.2: Backward propagation
        error = y - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Step 4.3: Update the weights and biases
        output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    # Step 5: Return the predicted output
    return predicted_output

# Example usage
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the Neural Network
predicted_output = train_neural_network(X, y, learning_rate=0.1, epochs=10000)

print(predicted_output)

555555555555555555555555555555555555555555555555555555555555555555555555555555

import numpy as np

def initialize_weights(input_dim, category):
    weights = np.random.uniform(size=(input_dim,))
    weights /= np.sum(weights)
    return weights

def calculate_similarity(input_pattern, weights):
    return np.minimum(input_pattern, weights).sum()

def update_weights(input_pattern, weights, vigilance):
    while True:
        activation = calculate_similarity(input_pattern, weights)
        if activation >= vigilance:
            return weights
        else:
            weights[np.argmax(input_pattern)] += 1
            weights /= np.sum(weights)

def ART_neural_network(input_patterns, vigilance):
    num_patterns, input_dim = input_patterns.shape
    categories = []

    for pattern in input_patterns:
        matched_category = None
        for category in categories:
            if calculate_similarity(pattern, category["weights"]) >= vigilance:
                matched_category = category
                break

        if matched_category is None:
            weights = initialize_weights(input_dim, len(categories))
            matched_category = {"weights": weights, "patterns": []}
            categories.append(matched_category)

        matched_category["patterns"].append(pattern)
        matched_category["weights"] = update_weights(pattern, matched_category["weights"], vigilance)

    return categories

# Example usage
input_patterns = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]])
vigilance = 0.5

categories = ART_neural_network(input_patterns, vigilance)

# Print the learned categories
for i, category in enumerate(categories):
    print(f"Category {i+1}:")
    print("Patterns:")
    [print(pattern) for pattern in category["patterns"]]
    print("Weights:")
    print(category["weights"])
    print()

6666666666666666666666666666666666666666666666666666666666666666666666666


#Write a python program to design a Hopfield Network which stores 4 vectors

import numpy as np
# Define the 4 vectors to be stored
vectors = np.array([[1, 1, -1, -1],
                    [-1, -1, 1, 1],
                    [1, -1, 1, -1],
                    [-1, 1, -1, 1]])

# Calculate the weight matrix
weights = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        if i == j:
            weights[i, j] = 0
        else:
            weights[i, j] = np.sum(vectors[i] * vectors[j])
            
# Define the activation function (in this case, a sign function)
def activation(x):
    return np.where(x >= 0, 1, -1)

# Define the Hopfield network function
def hopfield(input_vector, weights):
    output_vector = activation(np.dot(weights, input_vector))
    return output_vector

# Test the Hopfield network with one of the stored vectors as input
input_vector = vectors[0]
output_vector = hopfield(input_vector, weights)
print("Input vector:")
print(input_vector)
print("Output vector:")
print(output_vector)

77777777777777777777777777777777777777777777777777777777777777777777777777777

#How to train a neural network with Tensorflow / Pytorch and evaluation of logistic 
#regression using tensorflow

import tensorflow as tf 
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import load_breast_cancer 

df=load_breast_cancer()
# two classifications -- malignant / benign

X_train,X_test,y_train,y_test=train_test_split(df.data,df.target,test_size=0.20,random_state=42) 

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=Sequential([
    Flatten(input_shape=(X_train.shape[1],)),
    Dense(1,activation='sigmoid')
    ]) 

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=5) 

test_loss,test_accuracy = model.evaluate(X_test,y_test) 

print("Test Loss: ",test_loss)
print("accuracy is",test_accuracy)

88888888888888888888888888888888888888888888888888888888888888888888888888888888

#MNIST Handwritten Character Detection using PyTorch, Keras and Tensorflow

import tensorflow as tf
from keras.datasets import mnist 
from keras.models import Sequential
from keras.layers import Dense, Flatten 

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

#reshaping of input values
X_train = X_train / 255.0   #(28,28)
X_test = X_test / 255.0

# Define the model architecture 
model = Sequential([
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'), 
    Dense(10, activation='softmax')
])

# Compile the model 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  #multi class classification problems
                metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64 )

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test) 
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

99999999999999999999999999999999999999999999999999999999999999999999999999999999