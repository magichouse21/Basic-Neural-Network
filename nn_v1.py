import numpy as np
import matplotlib.pyplot as plt

#our weights and biases are going to be in a vector (vectorized form)

def init_params(layer_dims): #dimension of layers for initial parameters
    np.random.seed(67) #random number gen but it's gonna be the same "random number" 67 cuz it funny rn
    params = {} #dictionary 
    L = len(layer_dims)

    for l in range (1,L):
        params["W"+str(l)] = np.random.randn(0,1)

    