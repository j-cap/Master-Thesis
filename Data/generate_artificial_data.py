"""

Generate artificial data with specified monotonicity and min/max values.


Author: jweber
Data: 30.03.2020

"""

#%%
import numpy as np 
import matplotlib.pyplot as plt  
import plotly.express as px 
import plotly.graph_objects as go
import pandas as pd 
from scipy.stats import multivariate_normal

def main():
    """
    generates 3 different datasets, two 1D and one 2D dataset with noise
    """
    print("Generate 3 data series. ")
    print("Two are 1D - one is 2D.")
    minV = -10
    maxV = 10
    nPoints = 1000
    t = np.linspace(minV,maxV,nPoints)
    t_2D = np.linspace(minV/10, maxV/10, int(nPoints/10))
    mesh_t = np.meshgrid(t_2D, t_2D)

    print("generate first: 1D, f(x) = tanh(x) + white noise !")
    data = generate_data_1D(func=tanh_plus_noise, x=t, noise_level=0.05, plot=1, fname="data_1D-tanh")
    print("generate second: 1D, f(x) = tanh(x) + cos(x)^2 + x + white noise !")
    data_2 = generate_data_1D(func=non_linear_1D, x=t, noise_level=0.05, plot=1, fname="data_1D-tan-lin-cos2")
    print("generate third: 2D - f(x0,x1) = gaussian(x0,x1) + gaussian(x0,x1) + x0 + x0*x1 + white noise !")
    data_3 = generate_data_2D(      
        func=non_linear_2D, 
        x=np.c_[mesh_t[0].flatten(), mesh_t[1].flatten()],
        noise_level=0.01, 
        plot=1,
        nr_data_series=5,
        fname="data_2D-gaus-lin")
    print("Finished!")


def gauss(x, mean, sigma):
    return np.exp(-0.5*(x - mean)**2 / sigma**2)

def linear(x, p0, p1):
    y = p0 + p1 * x 
    return y

def quadratic(x, p0, p1, p2):
    y = linear(x, p0, p1) + p2*x**2
    return y

def non_linear_1D(seed=42, noise_level=0.1):
    def nl(x):
        np.random.seed(seed)
        y_lin = linear(x, 0, 0.2)# , 0.2)
        t = np.tanh(x)
        left_idx = np.argwhere(y_lin > t)
        right_idx = np.argwhere(y_lin < t)
        y_lin[left_idx] = t[left_idx]
        y = y_lin + t + + 0.2*np.cos(x)**2 + np.random.normal(scale=noise_level, size=np.size(x))
        return y + np.abs(np.min(y))
    return nl

def tanh_plus_noise(seed=42, noise_level=0.1):
    def nl(x):
        np.random.seed(seed)
        return np.tanh(x) + np.random.normal(scale=noise_level, size=np.size(x))
    return nl

def non_linear_2D(seed=42, noise_level=0.1):
    def nl(x):
        np.random.seed(seed)
        rv_1 = multivariate_normal(mean=[0, 0], cov=[[0.1,0.1], [0.001, 1.8]])
        rv_2 = multivariate_normal(mean=[0.4, 0.2], cov=[[0.1,0.01], [0.001, 1]])
        #c = np.c_[x,y]
        return rv_1.pdf(x) + rv_2.pdf(x) + 0.2*x[:,0] + 0.1*x[:,0]*x[:,1] + np.random.normal(scale=noise_level, size=np.shape(x)[0])
    return nl

def generate_data_1D(func, x, noise_level=0.1, fname=0, plot=0):
    """ generate data according to the function func and save it under fname """
    y = {"Exp_"+str(i): func(seed=np.int(i), noise_level=noise_level)(x=x) for i in np.arange(1,10)}
    y["t"] = x
    d = pd.DataFrame(data=y, columns=y.keys())
    if fname:
        d.to_pickle(path=fname+".pkl")
        d.to_csv(path_or_buf=fname+".csv", index=False)
    if plot:
        fig = go.Figure()
        for colName in d.columns[:-1]:
            fig.add_trace(go.Scatter(x=d["t"], y=d[colName], mode="markers", name=str(colName)))
        fig.show()     
    return d

def generate_data_2D(func, x, noise_level=0.1, fname=0, plot=0, nr_data_series=5):
    """ generate data according to the function func and save it under fname """
    y = {"Exp_"+str(i): func(seed=np.int(i), noise_level=noise_level)(x=x) for i in np.arange(1,int(nr_data_series+1))}
    y["x0"] = x[:,0]
    y["x1"] = x[:,1]
    d = pd.DataFrame(data=y, columns=y.keys())
    if fname:
        d.to_pickle(path=fname+".pkl")
        d.to_csv(path_or_buf=fname+".csv", index=False)
    if plot:
        fig = go.Figure()
        for colName in d.columns[:-2]:
            fig.add_trace(go.Scatter3d(x=d["x0"], y=d["x1"], z=d[colName], mode="markers", name=str(colName)))
        fig.show()     
    return d

if __name__=="__main__":
    main()

