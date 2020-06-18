#!/usr/bin/env python
# coding: utf-8

# **Implementation of the 1D smooth for a BSpline basis with penalties**

# In[5]:


# convert jupyter notebook to python script
#get_ipython().system('jupyter nbconvert --to script Smooth.ipynb')


# In[1]:


import numpy as np

from ClassBSplines import BSpline
from TensorProductSplines import TensorProductSpline



class Smooths(BSpline):
    """Implementation of the 1d smooth used in Additive Models."""

    def __init__(self, x_data, n_param, penalty="smooth", y_peak_or_valley=None, lam_c=None, lam_s=None):
        print("Type x_data: ", type(x_data))
        self.x_data = x_data
        self.n_param = n_param
        self.penalty = penalty
        self.lam_constraint = lam_c
        self.lam_smooth = lam_s
        self.bspline = BSpline()
        self.b_spline_basis(x_basis=self.x_data, k=self.n_param)
        
        # sanity check
        if penalty is "peak" or penalty is "valley":
            assert (y_peak_or_valley is not None), "Include real y_data in Smooths()"
        
        if penalty is "inc":
            self.penalty_matrix = self.D1_difference_matrix()
        elif penalty is "dec":
            self.penalty_matrix = -1 * self.D1_difference_matrix() 
        elif penalty is "conv":
            self.penalty_matrix = self.D2_difference_matrix()
        elif penalty is "conc":
            self.penalty_matrix = -1 * self.D2_difference_matrix()
        elif penalty is "smooth":
            self.penalty_matrix = self.Smoothness_matrix()
        elif penalty is "peak":
            self.penalty_matrix = self.Peak_matrix(basis=self.basis, y_data=y_peak_or_valley)
        elif penalty is "valley":
            self.penalty_matrix = self.Valley_matrix(basis=self.basis, y_data=y_peak_or_valley)
        else:
            print(f"Penalty {penalty} not implemented!")
    
class TP_Smooths(TensorProductSpline):
    """Implementation of the 2d tensor product spline smooth in Additive Models."""
    
    def __init__(self, x_data=None, n_param=(1,1), penalty="smooth", lam_c=None, lam_s=None):
        self.x_data = x_data
        self.x1, self.x2 = x_data[:,0], x_data[:,1]
        self.n_param = n_param
        self.penalty = penalty
        self.lam_constraint = lam_c
        self.lam_smooth = lam_s
        self.tps = TensorProductSpline()
        self.tensor_product_spline_2d_basis(x_basis=self.x_data, k1=n_param[0], k2=n_param[1])
        
        if penalty is "inc":
            self.penalty_matrix = self.D1_difference_matrix()
        elif penalty is "dec":
            self.penalty_matrix = -1 * self.D1_difference_matrix() 
        elif penalty is "conv":
            self.penalty_matrix = self.D2_difference_matrix()
        elif penalty is "conc":
            self.penalty_matrix = -1 * self.D2_difference_matrix()
        elif penalty is "smooth":
            self.penalty_matrix = self.Smoothness_matrix()
        elif penalty is "tps":
            self.penalty_matrix = self.D2_difference_matrix(
                n_param=int(np.product(self.n_param)))
        else:
            print(f"Penalty {penalty} not implemented!")
        


# In[14]:


def test():
    import pandas as pd
    import numpy as np
    X = pd.DataFrame(data={"x1": np.logspace(0.001,0.99,1000), #, 
                           "x2": np.linspace(0,1,1000), #}) #,
                           "x3": np.linspace(-2,2,1000), })

    TP = TP_Smooths(x_data=X[["x1", "x2"]].values, n_param=(10,10), penalties="smooth")
    

