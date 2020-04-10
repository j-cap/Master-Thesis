
"""

First tries of some toolboxes and algorithms

Author: jweber
Date: 09.03.2020
"""

#%% 
# import libraries
import numpy as np 
import pandas as pd 
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from pygam import LinearGAM, s, te, PoissonGAM, f, GAM
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

# load data
df = pd.read_csv('../../Data/data_1D-tan-lin-cos2.csv', dtype=np.float64)
X, y = df["t"].values, df["Exp_1"].values
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# scatter raw data
fig = px.scatter(df,x="t", y="Exp_1", trendline="ols")
# update the markers
fig.update_traces(marker=dict(size=2,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()

#%%
# pyGAM
# train
gam = LinearGAM(s(0, constraints="monotonic_inc"), n_splines=25).gridsearch(X_train.reshape((-1,1)), y_train.reshape((-1,1)))
# predict
XX = gam.generate_X_grid(term=0, n=500)
y = gam.predict(XX)
y_pred = gam.predict(X_test)
y_CI = gam.prediction_intervals(XX, width=.95)
#%%
# plot prediction and confindence intervals
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=XX.reshape((-1,)), y=y, name="Prediction", line=dict(color="firebrick", width=1))
    )
fig.add_trace(
    go.Scatter(x=XX.reshape((-1,)), y=y_CI[:,0], name="95% Confidence", line=dict(
        color="green", width=1, dash="dash"))
    )
fig.add_trace(
    go.Scatter(x=XX.reshape((-1,)), y=y_CI[:,1], name="95% Confidence", line=dict(color="green", width=1, dash="dash"))
    )
fig.add_trace(
    go.Scatter(
        x=X_train, y=y_train, name="Data", mode="markers", marker_size=4, marker_symbol="square-open",
        marker_line_color="midnightblue", marker_color="lightskyblue", marker_line_width=2) 
    )
fig.add_trace(
    go.Scatter(
        x=X_test, y=y_pred, name="Test Data Prediction", mode="markers", marker_size=6, 
        marker_symbol="circle-open", marker_line_color="greenyellow", marker_color="black",
        marker_line_width=2 )
    )
fig.add_trace(
    go.Scatter(
        x=X_test, y=y_test,name="Test Data", mode="markers", marker_size=6, 
        marker_symbol="circle-open", marker_line_color="greenyellow", marker_color="red",
        marker_line_width=2 )
    )

fig.update_layout(
    title="1-dim pyGAM Fit",
    xaxis_title="x",
    yaxis_title="f(x)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig.write_html("pyGAM-1D-fit.html")
fig.show()

