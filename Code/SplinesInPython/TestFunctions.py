#!/usr/bin/env python
# coding: utf-8

# ### Implementation of some test functions
# 
# This is the Jupyter notebook which implements the class TestFunctions with 4 different functions. Further work could be done on some plotting functionality.

# In[ ]:


# get_ipython().system('jupyter nbconvert --to script TestFunctions.ipynb')


# In[7]:


import numpy as np
import pandas as pd       

#import plotly.express as px
#t = TestFunctions()
#x, y = t.f3()
#px.scatter(x=x, y=y).show()
#t.save_data("f3_data", x=x, y=y)
    
class TestFunctions(SaveData):
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
        self.x1 = None
        self.x2 = None
        self.y = None
        self.data = None
        
    def save_data(self, fname, x=None, y=None):
        """Save data to csv."""
        df = pd.DataFrame(data={"x": x, "y": y})
        df.to_csv(fname+".csv", index=False)
        return
        
    def f1(self):
        self.x = np.random.normal(self.x1_min, self.x1_max, self.n_samples)
        self.y = np.tanh(-(self.x-5)) + np.exp(-(self.x)**2) + np.random.randn(len(self.x))*self.noise_level
        return self.x, self.y
    
    def f2(self, a=1, b=1):
        self.x = np.random.normal(self.x1_min, self.x1_max, self.n_samples)
        self.y = a / (1 + (b*x)**2) + np.random.randn(len(x))*self.noise_level
        return self.x, self.y
    
    def f3(self): 
        self.x = np.random.normal(self.x1_min, self.x1_max, self.n_samples)
        self.y = 0.5*np.sin(x) + 3.5*np.random.normal(0,1,len(x)) + np.random.randn(len(x))*self.noise_level
        return self.x, self.y
    
    def f4(self, grid=True):
        """Evaluate the 2-d function on a grid """
        self.x1 = np.random.normal(self.x1_min, self.x1_max, self.n_samples)
        self.x2 = np.random.normal(self.x2_min, self.x2_max, self.n_samples)
        if grid:
            self.x1g, self.x2g = np.meshgrid(x1, x2)
            self.y = x1g**3 - 3*x1g*x2g**2 + np.random.randn(*x1g.shape)*self.noise_level
            return self.x1g, self.x2g, self.y
        else:
            self.y = x1**3 - 3*x1*x2**2 + np.random.randn(len(x1))*self.noise_level
            return self.x1, self.x2, self.y

    

