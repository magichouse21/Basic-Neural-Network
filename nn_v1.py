import numpy as np
import matplotlib.pyplot as plt

#Much thanks to https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
#our weights and biases are going to be in a vector (vectorized form)

def init_params(layer_dims): #dimension for all layers for initial parameters
    np.random.seed(67) #random number gen but it's gonna be the same "random number" -- seed 67 cuz it funny rn --
    params = {} #dictionary 
    L = len(layer_dims)

    for l in range (1,L):
        params["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        params['B'+str(l)] = np.zeros(layer_dims[l])

    return params

def sigmoid(z):  #activation function
    A = 1/(1+np.exp(-z))
    cache = z

    return A, cache

#https://developers.google.com/machine-learning/crash-course/neural-networks/nodes-hidden-layers  ---this helps visualize it a bit---

def forward_prop(X, params):  #inputs the weights and biases in parameters, X is first input layer aka training data
    A = X
    caches = []

    for k in range (1, len(params)+1):
        A_prev = A

        Z = np.dot(params['W'+str(k)],A_prev) + params['B'+str(k)]
        linear_cache = (A_prev, params['W'+str(k)], params['B'+str(k)])
        A, activation_cache = sigmoid(Z)

        cache = (linear_cache, activation_cache)
        caches.append(cache)
    
    return A, caches

def cost_function(A, Y): #Y - values from previous layers of network
    m = Y.shape()[1]
    cost = -(1/m)*[np.dot(Y.T, np.log(A)) + np.dot((1 - Y.T), np.log(1 - A))] #Y.T is Y transposed
    #using Binary Cross-Entropy Loss / Log Loss cuz it's better than MSE ig?
    return cost  #the lower the better the model is doing

def one_layer_backward(dA, cache):
    activation_cache, linear_cache = cache
    #the cache holds the previous layer's activation, bias, and weight matrices
    Z = activation_cache    
    

