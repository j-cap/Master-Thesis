#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import plotly.graph_objects as go

from numpy.linalg import lstsq
from scipy.sparse import diags, kron

from Helper import addVertLinePlotly
from PenaltyMatrices import PenaltyMatrices

class BSpline(PenaltyMatrices):
    
    def __init__(self, x=None, order="cubic"):
        self.order = order
        self.x = x
        self.basis = None
        self.knots = None
        if order is "cubic":
            self.m = 2

    def b_spline(self, k, i, m=2):
        """Compute the i-th b-spline basis function of order m at the values given in x.
        
        Parameter:
        ---------------------
        k   : array,     of knot locations
        i   : int,       index of the b-spline basis function to compute
        m   : int,       order of the spline, default is 2 (cubic)
        """
        if m==-1:
            # return 1 if x is in {k[i], k[i+1]}, otherwise 0
            return (self.x >= k[i]) & (self.x < k[i+1]).astype(int)
        else:
            #print("m = ", m, "\t i = ", i)
            z0 = (self.x - k[i]) / (k[i+m+1] - k[i])
            z1 = (k[i+m+2] - self.x) / (k[i+m+2] - k[i+1])
            return z0*self.b_spline(k, i, m-1) + z1*self.b_spline(k, i+1, m-1)
        
    def b_spline_basis(self, k=10, m=2):
        """Set up model matrix for the B-spline basis.
        One needs k + m + 1 knots for a spline basis of order m with k parameters
        
        Parameters:
        -------------
        k : integer   - number of parameters (== number of B-splines)
        m : interger  - specifies the order of the spline, m+1 = order
        
        """
        x = self.x
        n = len(x) # n = number of data
        xmin, xmax = np.min(x), np.max(x)
        xk = np.quantile(a=x, q=np.linspace(0,1,k))
        dx = xk[-1] - xk[-2]
        xk = np.insert(xk, 0, np.arange(xmin-(m+1)*dx, xmin, dx))    
        xk = np.append(xk, np.arange(xmax+dx, xmax+(m+1)*dx, dx))
        X = np.zeros(shape=(n, k))
        for i in range(k):
            X[:,i] = self.b_spline(k=xk, i=i+1, m=m)
            
        self.basis = X
        self.knots = xk
        self.n_param = int(X.shape[1])
        return 
    
#    def fit(self, y):
#        """Compute Least Squares Fit of the B-spline basis with data y.
#        fit the linear model with the basis given by the B-spline basis of k parameters using
#        linear least squares with y = X.T beta
#        Parameters:
#        ---------------
#        y     : array     - target data
#        """
#        self.y = y
#        assert (self.basis is not None), "Please instantiate the B-Spline basis"
#        print("Basis: ", self.basis.shape)
#        print("Data: ", y.shape)
#        fit = lstsq(a=self.basis, b=y, rcond=None)
#        self.lstsq_fit = fit
#        self.coef_ = fit[0]
#        return 
    
    def plot_b_spline_basis(self):
        """Plot the B-spline basis matrix and the knot loactions.
        They are indicated by a vertical line.
        """
        if self.basis is None or self.knots is None:
            #k = int(input("Please specify the number of knots k: (int)"))
            k = 10
            self.b_spline_basis(k=k, m=self.m)

        fig = go.Figure()
        for i in range(self.basis.shape[1]):
            fig.add_trace(go.Scatter(x=self.x, y=self.basis[:,i], 
                                     name=f"BSpline {i+1}", mode="lines"))
        for i in self.knots:
            addVertLinePlotly(fig, x0=i)
        fig.update_layout(title="B-Spline basis")
        fig.show()
        return
    
    def plot_fitted_b_spline(self):
        """Plot the computed LS-fit."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.x, y=self.y, name="Data", mode="markers"))
        fig.add_trace(go.Scatter(x=self.x, y=self.basis @ self.coef_, name="Fit", mode="markers+lines"))
        fig.update_layout(title="Least Squares Fit with B-Splines")
        fig.show()
        return
                


# In[6]:


# convert jupyter notebook to python script
# get_ipython().system('jupyter nbconvert --to script ClassBSplines.ipynb')


# In[ ]:




