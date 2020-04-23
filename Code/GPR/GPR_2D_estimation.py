
"""

First tries of some toolboxes and algorithms
    - sklearn Gaussian Process Regression

Author: jweber
Date: 15.03.2020
"""

#%%
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern as M, ConstantKernel as C
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import plotly.io as pio
np.random.seed(1)
import time
# from tg_bot import send_TG_msg

pio.renderers.default = "browser"

#%% # load data
df = pd.read_csv('../../Data/data_2D-gaus-lin.csv', dtype=np.float64)
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[-2:]], df[df.columns[1]], test_size=0.33, random_state=42
)


# scatter raw data
fig = px.scatter_3d(df,x="x0", y="x1", z="Exp_1")
# update the markers
fig.update_traces(marker=dict(size=2,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()
#%% Instantiate the GPR model

kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
t1 = time.time()
y_pred = gp.predict(X=X_train.values)
t2 = time.time()
print(f"GPR training took {t2 - t1} seconds!")
# Make predictions on the meshed x-axis 
# y_pred, sigma = gp.predict(x, return_std=True)
#%% create grid for plotting
r = 100
xg = np.linspace(X_train.min()[0], X_train.max()[0], r)
yg = np.linspace(X_train.min()[1], X_train.max()[1], r)
xgrid, ygrid = np.meshgrid(xg, yg)
grid = np.vstack((xgrid.flatten(), ygrid.flatten())).T
z = gp.predict(X=grid)
#%% Plot the figure
fig = px.scatter_3d(x=X_test["x0"], y=X_test["x1"], z=y_test)
fig.add_trace(go.Scatter3d(x=X_test["x0"], y=X_test["x1"], z=gp.predict(X=X_test.values), name="Test predictions"))
#fig.add_trace(go.Surface(z=z.reshape((r,r))))
fig.show()
#%%

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


#%%
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as M, RBF as R

X = np.array(
    [[1.,2], [3.,4], [5.,1], [6.,5], [4, 7], [9,8.], [1.,2], [3.,4.], [5.,1],
     [6.,5],[4, 7.], [9,8.], [1.,2], [3.,4], [5.,1], [6.,5], [4, 7.], [9,8.]])

y=[0.84147098,  0.42336002, -4.79462137, -1.67649299,  4.59890619,  7.91486597, 0.84147098,  0.42336002, -4.79462137,
  -1.67649299,  4.59890619,  7.91486597, 0.84147098,  0.42336002, -4.79462137, -1.67649299,  4.59890619,  7.91486597]

kernel = R(X[0]) * M(X[1])
gp = GaussianProcessRegressor(kernel=kernel)

ft = gp.fit(X, y)

#%%
xg = np.linspace(X.min(), X.max(), 10)
xgrid, ygrid = np.meshgrid(xg, xg)
grid = np.vstack((xgrid.flatten(), ygrid.flatten())).T
z = gp.predict(X=grid)


#%%
fig = px.scatter_3d(x=X[:,0], y=X[:,1], z=y)
fig.add_trace(go.Surface(z=z.reshape((10,10))))
fig.show()
#%%