#!/usr/bin/env python
# coding: utf-8

# ### File for saving usefull code snippets for
# 
# - Plotting
# - Fast calculations
# - Test

# In[ ]:


# convert jupyter notebook to python script
#get_ipython().system('jupyter nbconvert --to script Code_snippets.ipynb')


# In[8]:

# In[6]:


def find_peak_and_basis(x=None, y=None):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.signal import find_peaks
    from Smooth import Smooths
    from Helper import addVertLinePlotly

    np.random.seed(42)

    if x is None and y is None:
        x = np.linspace(-3,3,1000)
        y = np.sin(x) + 0.1*np.random.randn(len(x))
    
    peaks, properties = find_peaks(y, distance=int(len(x)))
    x_peak, y_peak = x[peaks], y[peaks]

    s = Smooths(x_data=x, n_param=20)

    s.basis[peaks]



    fig = go.Figure()
    for i in range(s.basis.shape[1]):
        fig.add_trace(go.Scatter(x=s.x, y=s.basis[:,i], 
                                 name=f"BSpline {i+1}", mode="lines"))
    for i in s.knots:
        addVertLinePlotly(fig, x0=i)

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data"))
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=x_peak, 
            y=y_peak,
            marker=dict(color="red", size=18, line=dict(color="DarkSlateGrey", width=2)),
            name="Peak Locations"
        )

    )   
    fig.show()


# In[ ]:




