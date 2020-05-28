#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class TestFunctions:
    """
    Collection of test functions.
    
    ...
    
    Attributes
    ----------
    x1_min : float
             Minimum value of axis 1. 
    x1_max : float
             Maximum value of axis 1.
    x2_min : float
             Minimum value of axis 2.
    x2_max : float
             Maximum value of axis 2.
    n_samples : int
                Number of samples to evaluate
    noise_level : float
                  Specify the magnitude of noise influence.
    

    Methods
    ---------
    f1(x)
        tanh + exp + noise
    f2( )
        peak + noise (optional)
    f3(x)
        sin + linear part + noise
    f4(x1, x2)
        affensattel
    """
    def __init__(self, x1_min=0, x1_max=1, x2_min=0, x2_max=1, n_samples=1000, noise_level=0.25):
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_min = x2_min
        self.x2_max = x2_max
        self.n_samples = n_samples
        self.noise_level = noise_level
        
    def f1(self):
        x = np.linspace(self.x1_min, self.x1_max, self.n_samples)
        y = np.tanh(-(x-5)) + np.exp(-(x)**2) + np.random.randn(len(x))*self.noise_level
        return x, y
    
    def f2(self, a=1, b=1):
        x = np.linspace(self.x1_min, self.x1_max, self.n_samples)
        y = a / (1 + (b*x)**2) + np.random.randn(len(x))*self.noise_level
        return x, y
    
    def f3(self): 
        x = np.linspace(self.x1_min, self.x1_max, self.n_samples)
        y = 0.5*np.sin(x) + 3.5*np.linspace(0,1,len(x)) + np.random.randn(len(x))*self.noise_level
        return x, y
    
    def f4(self, grid=True):
        """Evaluate the 2-d function on a grid """
        x1 = np.linspace(self.x1_min, self.x1_max, self.n_samples)
        x2 = np.linspace(self.x2_min, self.x2_max, self.n_samples)
        y = x1**3 - 3*x1*x2**2 + np.random.randn(len(x1))*self.noise_level
        return x1, x2, y



