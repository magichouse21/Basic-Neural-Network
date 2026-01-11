import numpy as np

#from keras.datasets import mnist
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#im gonna try making it solely on my own, it's gonna go terribly

#i might as well use MNIST so i added it here in comments to save for later

def initialize_network(input_dim):  #input dimensions
    layers = {}
    L = input_dim
    np.random.seed(67)

    for k in range (0,L-1):   
        layers['W'+str(k)] = np.random.randn(L,L) * 0.1   #two layers both 3x3 matrices
        layers['B'+str(k)] = np.zeros(input_dim)

    return layers

def relu(x): #activation function #1
    return np.maximum(0,x)

def sigmoid(z):  #activation function  #2
    return 1/(1+np.exp(-z))


def forward_propogation(inputs, layers): #given the input, how will we find the next layer stuff
    prev_input = inputs  #we need this to progress through the hidden layers to foward prop until we hit the final output
    for k in range (0, len(inputs)-1):
        print(layers['W'+str(k)], prev_input)
        z = np.dot(layers['W'+str(k)], prev_input) + layers['B'+str(k)]
        prev_input = relu(z)  #final output of the next neuron and when the loop is over, we will have the output matrix

    return prev_input

def cost(output, actual):  #MSE cuz its super easy to understand
    return np.mean((output - actual)**2)

def back_prop(inputs, layers):

    return 0



net = initialize_network(4)
print(cost(forward_propogation([123, 234, 345], net), 9)) 


