#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!jupyter nbconvert --to script Helper.ipynb


# In[ ]:

import numpy as np
import plotly.graph_objects as go

def addVertLinePlotly(fig, x0=0, y0=0, y1=1):
    """ plots a vertical line to the given figure at position x"""
    fig.add_shape(dict(type="line", x0=x0, x1=x0, y0=y0, y1=1.2*y1, 
                       line=dict(color="LightSeaGreen", width=1)))
    return

# check if x1 is none, if x2 is ndarray, use it
def check_if_none(x1, x2, cls):
    if x1 is None:
        if type(x2) is None:
            print("Type of x: ", type(x))
            print("Include data for 'x'!")
            return    
        elif type(x2) is np.ndarray:
            print("Use 'x_basis' for the spline basis!")
            cls.x = x2
        else:
            print("Datatype for 'x' not supported!")
            return
    else:
        print("'x' from initialization is used for the spline basis!")
    return