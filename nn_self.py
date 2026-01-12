import numpy as np

def initialize_net(layer_dims): #variable is dimension of hidden layers
    L = len(layer_dims)-1
    layers = {}
    np.random.seed(67)

    for k in range (1, L):
        layers['W'+str(k)] = np.random.randn(layer_dims[k],layer_dims[k-1])
        layers['B'+str(k)] = np.zeros(layer_dims[k])
    
    return layers


net = initialize_net([1,2,3,2,1])

print(net)