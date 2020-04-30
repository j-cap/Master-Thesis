
#%%
# Univariante Splines from the scipy.interpolate toolbox
from scipy.interpolate import BSpline, UnivariateSpline
import numpy as np 
import plotly.express as px

x = np.linspace(-1,1,50)
y = np.exp(-x**2)+ 0.1*np.random.randn(50)
fig = px.scatter(x=x,y=y)
# spline with default parameters
spl = UnivariateSpline(x=x, y=y, k=3, ext=1)
xs = np.linspace(-1.2,1.2,1000)
fig.add_scatter(x=xs, y=spl(xs))
# spline with user defined smoothing
spl.set_smoothing_factor(0.4)
fig.add_scatter(x=xs, y=spl(xs), name="Smoothing Spline")

#%%