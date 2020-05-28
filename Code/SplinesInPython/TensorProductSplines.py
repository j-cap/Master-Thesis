#!/usr/bin/env python
# coding: utf-8

# **Tensor product spline implementation**

# In[9]:


# convert jupyter notebook to python script
# get_ipython().system('jupyter nbconvert --to script TensorProductSplines.ipynb')


# In[4]:


import numpy as np
import plotly.graph_objects as go

from scipy.sparse import kron

from ClassBSplines import BSpline
from PenaltyMatrices import PenaltyMatrix


class TensorProductSpline(BSpline):
    """Implementation of the tensor product spline according to Simon Wood, 2006."""
    
    def __init__(self, x1=None, x2=None):
        """It is important that len(x1) == len(x2)."""
        self.x1 = x1
        self.x2 = x2
        self.basis = None
        
    def tensor_product_spline_2d(self, k1=5, k2=5, print_shapes=False):
        """Calculate the TPS from two 1d B-splines.
        
        Parameters:
        -------------
        k1 : integer   - Number of knots for the first B-spline.
        k2 : integer   - Number of knots for the second B-Spline.
        print_shape : bool - prints the dimensions of the basis matrices.
        
        """
        self.k1 = k1
        self.k2 = k2
        BSpline_x1 = BSpline(self.x1)
        BSpline_x2 = BSpline(self.x2)
        BSpline_x1.b_spline_basis(k=self.k1)
        BSpline_x2.b_spline_basis(k=self.k2)
        self.X1 = BSpline_x1.basis
        self.X2 = BSpline_x2.basis
        self.basis = kron(self.X1, self.X2).toarray()

        if print_shapes:
            print("Shape of the first basis: ", self.X1.shape)
            print("Shape of the second basis: ", self.X2.shape)
            print("Shape of the tensor product basis: ", self.basis.shape)
        return
        
    def plot_basis(self):
        """Plot the tensor product spline basis matrix for a 2d TPS."""
        fig = go.Figure()
        x1g, x2g = np.meshgrid(self.x1, self.x2)
        #print("x1g: ", x1g.shape)
        #print("x2g: ", x2g.shape)
        for i in range(self.basis.shape[1]):
            fig.add_trace(
                go.Surface(
                    x=x1g, y=x2g,
                    z=self.basis[:,i].reshape((self.X2.shape[0], self.X1.shape[0])),
                    name=f"TPS Basis {i+1}",
                    showscale=False
                )
            )
                
        fig.update_layout(
            scene=dict(
                xaxis_title="x1",
                yaxis_title="x2",
                zaxis_title=""
            ),
            title="Tensor product spline basis", 
        )
        fig.show()
        return

