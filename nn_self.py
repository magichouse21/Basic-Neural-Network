import numpy as np
#im gonna try making it solely on my own, it's gonna go terribly

# should look like this: 

# input:

# 123 + 234 = 5 

#i think i might have it find the remainder of some added numbers where the divisor is 11
#input is three numbers 123, 234, 345, add up the numbers so 702 with divisor 11, remainder is 9
#gotta start with smthn linear should take almost no iterations to be able to find the answer

def initialize_network():
    layers = {}
    np.random.seed(67)

    for k in range (0,2):   
        layers['W'+str(k)] = np.random.randn(3,3) * 0.1   #two layers both 3x3 matrices
        layers['B'+str(k)] = np.zeros(3)

    return layers

def relu(x):
    return np.maximum(0,x)

def forward_propogation(input, layers): #given the input, how will we find the next layer stuff
    prev_input = input  #we need this to progress through the hidden layers to foward prop until we hit the final output
    print(len(layers))
    for k in range (0, len(input)-1):
        z = np.dot(layers['W'+str(k)], prev_input) + layers['B'+str(k)]
        a = z 
        prev_input = relu(a)  #final output of the next neuron and when the loop is over, we will have the output matrix

    return prev_input

def cost(output, actual):  #MSE cuz its super easy to understand
    return (output - actual)**2

net = initialize_network()
print(forward_propogation([123, 234, 345], net))