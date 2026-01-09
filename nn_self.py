import numpy as np
#im gonna try making it solely on my own, it's gonna go terribly

# should look like this: 

# input:

# 123 + 234 = 5 

#i think i might have it find the remainder of some added numbers where the divisor is 11
#given inputs 123 and 234, add up the numbers so 357 with divisor 11, remainder is 5
#gotta start with smthn linear should take almost no iterations to be able to find the answer

def initialize_network():
    layers = {}
    np.random.seed(67)

    for k in range (0,2):   
        layers['W'+str(k)] = np.random.randn(3,3) * 0.1
        layers['B'+str(k)] = np.zeros(3)

    return layers

def relu(x):
    return np.max(0,x)

def forward_propogation(input, layers): #given the input, how will we find the next layer stuff
    new_layer = []
    
    for k in input:
        z = layers['W'+str(k)]*input + layers['B'+str(k)]
        a += z
    a = relu(a)  #final output of the next neuron
    new_layer.append(a)


    return new_layer

def cost(output, actual):  #MSE cuz its super easy to understand
    return (output - actual)**2

print(initialize_network())