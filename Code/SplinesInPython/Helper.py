#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!jupyter nbconvert --to script Helper.ipynb


# In[ ]:


import plotly.graph_objects as go

def addVertLinePlotly(fig, x0=0, y0=0, y1=1):
    """ plots a vertical line to the given figure at position x"""
    fig.add_shape(dict(type="line", x0=x0, x1=x0, y0=y0, y1=1.2*y1, 
                       line=dict(color="LightSeaGreen", width=1)))
    return

