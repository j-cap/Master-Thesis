#!/usr/bin/env python
# coding: utf-8

# **Implementation of the penalty matrices**

# In[2]:


# convert jupyter notebook to python script
#!jupyter nbconvert --to script PenaltyMatrices.ipynb


# In[1]:


import numpy as np
from scipy.sparse import diags

class PenaltyMatrix():
    """Implementation of the various penalty matrices for penalized B-Splines."""
    def __init__(self, n_param):
        self.n_param = n_param
        self.D1 = None
        self.D2 = None
        
    def D1_difference_matrix(self, n_param=0, print_shape=False):
        """Calculated the first order difference matrix.  
        
        Parameters:
        ------------
        n_param : integer  - Dimension of the difference matrix, overwrites
                             the specified dimension.
        print_shape : bool - Prints the dimension of the penalty matrix.
        
        Returns:
        ------------
        D1 : ndarray  - a matrix of size [k x k], 
                        where the last row contains only zeros.
        """
        if n_param == 0:
            k = self.n_param
        else:
            k = n_param
        assert (type(k) is int), "Type of input k must be integer!"
        d = np.array([-1*np.ones(k), np.ones(k)])
        offset=[0,1]
        D1 = diags(d,offset, dtype=np.int).toarray()
        D1[-1:] = 0.
        if print_shape:
            print("Shape of D1-Matrix: {}".format(D1.shape))
        self.D1 = D1
        return D1

    def D2_difference_matrix(self, n_param=0, print_shape=False):
        """Calculated the second order difference matrix. 

        Parameters:
        ------------
        n_param : integer  - Dimension of the difference matrix, overwrites
                             the specified dimension.
        print_shape : bool - Prints the dimension of the penalty matrix.
        
        Returns:
        ------------
        D2 : ndarray  - a matrix of size [k x k], 
                        where the last two row contains only zeros.
        """
        if n_param == 0:
            k = self.n_param
        else:
            k = n_param
        assert (type(k) is int), "Type of input k is not integer!"
        d = np.array([np.ones(k), -2*np.ones(k), np.ones(k)])
        offset=[0,1,2]
        D2 = diags(d,offset, dtype=np.int).toarray()
        D2[-2:] = 0.
        if print_shape:
            print("Shape of D2-Matrix: {}".format(D2.shape))
        self.D2 = D2
        return D2
    


# In[ ]:




