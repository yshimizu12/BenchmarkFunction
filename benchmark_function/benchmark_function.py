"""
@author: Yuki Shimizu
Benchmarck functions for optimization
"""

import math

import numpy as np


# check dimension
def check(x):
    x = np.array(x)
    if len(x.shape)!=2: raise Exception('Only 2D array is expected. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.')
    return x

# benchmark function
class BenchmarkFunction:
    """
    Parameter------
    input_array : 
        2D array of shape (n, m) is expected.
        n : number of data
        m : input size
    ---------------
    
    Functions------
    Sphere : 
        Basic function
        Search area: [-5.12,5.12]^n
        Optimal solusion: (0,...,0)
    
    Ellipsoid : 
        Weak ill-scale
        Search area: [-5.12,5.12]^n
        Optimal solusion: (0,...,0)
    
    kTablet : 
        strong ill-scale
        Search area: [-5.12,5.12]^n
        Optimal solusion: (0,...,0)

    RosenbrockStar :
        Strong parameter dependency between x1 and the others
        Search area: [-2.048,2.048]^n
        Optimal solusion: (1,...,1)

    RosenbrockChain : 
        Strong parameter dependency between neighboring parameters
        Search area: [-2.048,2.048]^n
        Optimal solusion: (1,...,1)
    
    Bohachevsky : 
        Weak multimodality
        Search area: [-5.12,5.12]^n
        Optimal solusion: (0,...,0)

    Ackley : 
        Weak multimodality
        Search area: [-32.768,32.768]^n
        Optimal solusion: (0,...,0)

    Schaffer : 
        Strong multimodality
        Search area: [-100,100]^n
        Optimal solusion: (0,...,0)

    Rastrigin : 
        Strong multimodality
        Search area: [-5.12,5.12]^n
        Optimal solusion: (1,...,1)
    ---------------
    
    search_area------
    Return the search areas (type: dict) for each function 
    ---------------

    optimal_solution------
    Return the optimal solutions (type: dict) for each function 
    dimension (int) :
        dimension of input variable
    ---------------
    """
    def __init__(self):
        pass
    
    def Sphere(self, input_array): 
        input_array = check(input_array)
        return (input_array**2).sum(axis=-1)
    
    def Ellipsoid(self, input_array): 
        input_array = check(input_array)
        dim = input_array.shape[1]
        
        coef = [1000**(i/(dim-1)) for i in range(dim)]
        coef = np.array(coef)
        
        return ((input_array*coef)**2).sum(axis=1)
    
    def kTablet(self, input_array):
        input_array = check(input_array)
        dim = input_array.shape[1]
        k = math.ceil(dim/4)
                
        return (input_array[:,:k]**2).sum(axis=1) \
                +((100*input_array[:,k:])**2).sum(axis=1)
        
    def RosenbrockStar(self, input_array):
        input_array = check(input_array)
        input_array_1st = input_array[:,:1]
        input_array_rest = input_array[:,1:]
        
        return (100*(input_array_1st-input_array_rest**2)**2 \
                +(1-input_array_rest)**2).sum(axis=1)

    def RosenbrockChain(self, input_array):
        input_array = check(input_array)
        
        return (100*(input_array[:,1:]-input_array[:,:-1]**2)**2 \
                +(1-input_array[:,:-1])**2).sum(axis=1)
        
    def Bohachevsky(self, input_array):
        input_array = check(input_array)
        
        return (input_array[:,:-1]**2 \
                +2*input_array[:,1:]**2 \
                -0.3*np.cos(3*np.pi*input_array[:,:-1]) \
                -0.4*np.cos(4*np.pi*input_array[:,1:]) \
                +0.7).sum(axis=1)
        
    def Ackley(self, input_array):
        input_array = check(input_array)
        dim = input_array.shape[1]
        
        return 20 \
                -20*np.exp(-0.2*((input_array**2).sum(axis=1)/dim)**0.5) \
                +np.e \
                -np.exp((np.cos(2*np.pi*input_array)).sum(axis=1)/dim)

    def Schaffer(self, input_array):
        input_array = check(input_array)
        
        return ((input_array[:,:-1]**2+input_array[:,1:]**2)**0.25 \
                 *(np.sin(50*(input_array[:,:-1]**2+input_array[:,1:]**2)**0.1)**2 \
                +1.0)).sum(axis=1)

    def Rastrigin(self, input_array):
        input_array = check(input_array)
        dim = input_array.shape[1]
        
        return 10*dim + \
                ((input_array-1)**2-10*np.cos(2*np.pi*(input_array-1))).sum(axis=1)

    def search_area(self):
        search_area = {
            'Sphere' : [-5.12,5.12],
            'Ellipsoid' : [-5.12,5.12],
            'kTablet' : [-5.12,5.12],
            'RosenbrockStar' : [-2.048,2.048],
            'RosenbrockChain' : [-2.048,2.048],
            'Bohachevsky' : [-5.12,5.12],
            'Ackley' : [-32.768,32.768],
            'Schaffer' : [-100,100],
            'Rastrigin' : [-5.12,5.12],
        }
        return search_area
    
    def optimal_solution(self, dimension=2):
        zeros = [0]*dimension
        ones = [1]*dimension
        
        optimal_solution = {
            'Sphere' : zeros,
            'Ellipsoid' : zeros,
            'kTablet' : zeros,
            'RosenbrockStar' : ones,
            'RosenbrockChain' : ones,
            'Bohachevsky' : zeros,
            'Ackley' : zeros,
            'Schaffer' : zeros,
            'Rastrigin' : ones,
        }
        return optimal_solution
