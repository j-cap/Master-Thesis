#!/usr/bin/env python
# coding: utf-8

# In[10]:


# convert jupyter notebook to python script
# get_ipython().system('jupyter nbconvert --to script DiagnosticPlot.ipynb')


# In[9]:


import plotly.graph_objects as go
import numpy as np
    
class DiagnosticPlotter():
    """ DiagnosticPlotter class. """
    
    def __init__(self):
        
        pass
    
    def bar_chart_of_coefficient_dataframe(self, df):
        """Takes the dataframe Model.df_beta and plots a bar chart of the rows. """
        
        fig = go.Figure()
        x = np.arange(df.shape[1])
        
        for i in range(df.shape[0]):
            fig.add_trace(go.Bar(x=x, y=df.iloc[i], name=f"Iteration {i}"))

        fig.update_layout(title="Coefficient adaption of IRLS")
        fig.show()
        
    def bar_chart_of_coefficient_difference_dataframe(self, df):
        """Takes the dataframe Model.df_beta and plots a bar chart of the rows. """
        
        fig = go.Figure()
        x = np.arange(df.shape[1]-1)
        
        for i in range(df.shape[0]):
            fig.add_trace(go.Bar(x=x, y=np.diff(df.iloc[i]), name=f"Iteration {i}"))

        fig.update_layout(title="Difference in coefficients", )
        fig.show()
        
