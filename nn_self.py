import numpy as np

#from keras.datasets import mnist
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#im gonna try making it solely on my own, it's gonna go terribly

#i might as well use MNIST so i added it here in comments to save for later

def initialize_network(input_dim):  #input dimensions
    layers = {}
    L = len(input_dim)-1
    np.random.seed(67)

    for k in range(L):   
        layers['W'+str(k)] = np.random.randn(input_dim[k+1],input_dim[k]) * 0.1   #two layers
        layers['B'+str(k)] = np.zeros((input_dim[k],1))

    return layers

def relu(x): #activation function #1
    return np.maximum(0,x)

def sigmoid(z):  #activation function  #2
    return 1/(1+np.exp(-z))


def forward_propogation(inputs, layers): #given the input, how will we find the next layer stuff
    A = inputs #we need this to progress through the hidden layers to foward prop until we hit the final output
    for k in range(len(layers)//2):
        Z = layers[f'W{k}'], A + layers[f'B{k}']
        A = relu(Z)
  #final output of the next neuron and when the loop is over, we will have the output matrix

    return A

def cost(output, actual):  #MSE cuz its super easy to understand
    return np.mean((output - actual)**2)

def back_prop(inputs, layers):

    return 0



net = initialize_network([1,4,3,2,1])  #given an array with an int for each layer dimension like [1, 2, 3, 2, 1] where each layer is 1, 2, 3, 2, 1 nodes respec.
#print(net)
print(cost(forward_propogation([123], net), 9)) 


