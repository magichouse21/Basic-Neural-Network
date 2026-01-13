import numpy as np

def initialize(layer_dims): #variable is dimension of hidden layers
    L = len(layer_dims)
    layers = {}
    np.random.seed(67)

    for k in range (1, L):
        layers['W'+str(k)] = np.random.randn(layer_dims[k],layer_dims[k-1])
        layers['B'+str(k)] = np.zeros((layer_dims[k], 1))
    
    return layers
    
def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)   # stability
    expz = np.exp(z)
    return expz / np.sum(expz, axis=0, keepdims=True)

def forward_prop(ipt, layers):
    prev = ipt
    L = len(layers)//2
    caches = []

    for k in range (1, L+1): #except for the output which we want
        z = np.dot(layers['W'+str(k)], prev) + layers['B'+str(k)]
        linear_cache = (prev, layers['W'+str(k)], layers['B'+str(k)]) 
        prev = softmax(z)
        activation_cache = softmax(z)

        # storing the both linear and activation cache
        cache = (linear_cache, activation_cache)
        caches.append(cache)

    return prev, caches  #output after all props

#https://medium.com/@anishnama20/understanding-cost-functions-in-machine-learning-types-and-applications-cd7d8cc4b47d  helped me decide
def cost(ipt, actual): #categorical cross-entropy cuz imma try MNIST
    n = len(ipt)
    return -1/n * np.sum(np.sum(actual * np.log(ipt)))

#gotta learn backprop
def backprop():
    return 0

np.random.seed(67)
input = np.random.randn(784,1)
net = initialize([784,3,10])
forward = forward_prop(input,net)[0]
print(cost(forward_prop(input,net)[0], [0,1,0,0,0,0,0,0,0,0]))