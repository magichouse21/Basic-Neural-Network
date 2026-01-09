import numpy as np
#im gonna try making it solely on my own, it's gonna go terribly but we're only going to have 2 hidden layers of 3 nodes each

# should look like this: 

# input:

# 1  .02  .523    output
# 2  .12  .76     

#i think i might have it find the remainder of some added numbers where the divisor is 11
#given inputs 123 and 234, add up the numbers so 357 with divisor 11, remainder is 5

def initialize_network():
    layers = {}
    np.random.seed(67)

    for k in range (0,2):   
        layers['W'+str(k)] = np.random.randn(3,1) * 0.1
        layers['B'+str(k)] = 0

    return layers

def sigmoid(x):
    A = 1/(1+np.exp(-x))
    return A

def forward_propogation(input, layers): #given the input, how will we find the next layer stuff
    new_layer = []
    
    for k in input:
        z = layers['W'+str(k)]*input + layers['B'+str(k)]
        a = sigmoid(z)  #final output of the neuron



    return 0

def cost(output, actual):  #MSE cuz its super easy to understand
    return (output - actual)**2
