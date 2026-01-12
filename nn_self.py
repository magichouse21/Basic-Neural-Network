import numpy as np

def initialize_net(layer_dims): #variable is dimension of hidden layers
    L = len(layer_dims)
    layers = {}
    np.random.seed(67)

    for k in range (1, L):
        layers['W'+str(k)] = np.random.randn(layer_dims[k],layer_dims[k-1])
        layers['B'+str(k)] = np.zeros(layer_dims[k])
    
    return layers

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward_prop(ipt, layers):
    prev = ipt
    L = len(layers)//2

    for k in range (1, L+1): #except for the output which we want
        z = np.dot(layers['W'+str(k)], prev) + layers['B'+str(k)]
        prev = sigmoid(z)

    return prev

#https://medium.com/@anishnama20/understanding-cost-functions-in-machine-learning-types-and-applications-cd7d8cc4b47d  helped me decide
def cost(ipt, actual): #categorical cross-entropy cuz imma try MNIST
    n = len(ipt)
    return -1/n * np.sum(np.sum(actual * np.log(ipt)))

    
net = initialize_net([3,5,3,2,4])
print(cost(forward_prop([123, 234, 345], net), 5)) #find sum of 3 numbers, figure out remainder with divisor 17
