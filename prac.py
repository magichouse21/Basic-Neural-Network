import numpy as np

def main1():
    np.random.seed(67)
    print(np.random.randn(3,3)*.1) #basically gives us random numbers from a normal distribution where mean is 0 and variance is 1 
    #the first input is how many columns we want, the second is how many rows'


main1()