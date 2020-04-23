"""
https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html

Example for GPR from sklearn

Date: 15.04.2020
"""

#%%
import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import plotly.graph_objects as go 
import plotly.express as px 
import plotly.offline as py
import plotly.io as pio
np.random.seed(1)

def f(x):
    """ The function to predict. """
    return x * np.sin(x)
#%%
# -------------------------------------------------------------------------------
# First the noiseless case
X = np.atleast_2d([1.,3.,5.,6.,7.,8.]).T

# Observations
y = f(X).ravel()

# Mesh the input space for evaluation of the real funtion, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0,10,1000)).T

# Instantiate the GPR model
kernel = C(1.0, (1e-2, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit the data through ML estimation of the parameters
gp.fit(X,y)

# Make predictions on the meshed x-axis 
y_pred, sigma = gp.predict(x, return_std=True)
#%% Plot the figure
X = X.ravel()
x = x.ravel()
y_grid = f(x).ravel()
fig = go.Figure()
t1 = go.Scatter(x=X, y=y, name="$f(x) = x \\sin(x)$", mode="markers", marker=dict(size=10, color="red"))
t2 = go.Scatter(x=x, y=y_grid, name="f(x)-grid", mode="lines", line=dict(color="firebrick", width=2, dash="dot"))
t3 = go.Scatter(x=x, y=y_pred, name="prediction", mode="lines", line=dict(color="blue", width=2, dash="dash"))
t4 = go.Scatter(x=x, y=y_pred-sigma, name=None, line=dict(color="blue", width=1))
t5 = go.Scatter(x=x, y=y_pred+sigma, name="CI", line=dict(color="blue", width=1), fill="tonexty")
data = [t1, t2, t3, t4, t5]
layout = go.Layout(
    title="GPR for one dim - noiseless ",
    xaxis_title="x",
    yaxis_title="y=f(x)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='sklearn_example', include_mathjax='cdn')
# pio.write_image(fig, 'pic_23.png', width = 1280, height = 1024)
# %% ----------------------------------------------------------------------------------
# noisy example
X = np.linspace(0.1,9.9,20)
X = np.atleast_2d(X).T 

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 0.1 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Instantiate the GPR model
gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2, n_restarts_optimizer=10)

# Fit to data using ML method
gp.fit(X,y)

# Make predictions on the meshed x-axis
x = np.atleast_2d(np.linspace(0,10,1000)).T
y_pred, sigma = gp.predict(x, return_std=True)

#%% Plot the figure
X = X.ravel()
x = x.ravel()
y_grid = f(x).ravel()
fig = go.Figure()
t1 = go.Scatter(x=X, y=y, name="$f(x) = x \\sin(x)$", mode="markers", marker=dict(size=10, color="red"))
t2 = go.Scatter(x=x, y=y_grid, name="f(x)-grid", mode="lines", line=dict(color="firebrick", width=2, dash="dot"))
t3 = go.Scatter(x=x, y=y_pred, name="prediction", mode="lines", line=dict(color="blue", width=2, dash="dash"))
t4 = go.Scatter(x=x, y=y_pred-sigma, name=None, line=dict(color="blue", width=1))
t5 = go.Scatter(x=x, y=y_pred+sigma, name="CI", line=dict(color="blue", width=1), fill="tonexty")
data = [t1, t2, t3, t4, t5]
layout = go.Layout(
    title="Data with noise",
    xaxis_title="x",
    yaxis_title="y=f(x)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='sklearn_example', include_mathjax='cdn')

# %%
