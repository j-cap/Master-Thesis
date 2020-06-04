#!/usr/bin/env python
# coding: utf-8

# **Implementation of the 1D smooth for a BSpline basis with penalties**

# In[5]:


# convert jupyter notebook to python script
#get_ipython().system('jupyter nbconvert --to script Smooth.ipynb')


# In[4]:

from ClassBSplines import BSpline

class Smooths(BSpline):
    """Implementation of the 1d smooth used in Additive Models."""

    def __init__(self, x_data, n_param, penalty="no"):
        print("Type x_data: ", type(x_data))
        self.x_data = x_data
        self.n_param = n_param
        self.penalty = penalty
        self.lam = None
        self.bspline = BSpline()
        self.b_spline_basis(x_basis=self.x_data, k=self.n_param)
        
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
            self.penalty_matrix = self.D1_difference_matrix()
        elif penalty is "no":
            self.penalty_matrix = self.Zero_matrix(n_param)
            print("No penalty used!")
        else:
            print(f"Penalty {penalty} not implemented!")
    
