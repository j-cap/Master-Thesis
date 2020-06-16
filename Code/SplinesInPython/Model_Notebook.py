#!/usr/bin/env python
# coding: utf-8

# In[16]:


# convert jupyter notebook to python script
get_ipython().system('jupyter nbconvert --to script Model_Notebook.ipynb')


# In[1]:


import plotly.express as px
import plotly.graph_objects as go
import numpy as np
np.random.seed(42)
import pandas as pd
from numpy.linalg import lstsq
from scipy.linalg import block_diag
from sklearn.metrics import mean_squared_error


from Smooth import Smooths as s
from Smooth import TP_Smooths as tps
from TensorProductSplines import TensorProductSpline as t
from ClassBSplines import BSpline as b
from PenaltyMatrices import PenaltyMatrix
from DiagnosticPlot import DiagnosticPlotter

class Model(DiagnosticPlotter):
    
    possible_penalties = { "smooth": PenaltyMatrix().D2_difference_matrix, 
                           "inc": PenaltyMatrix().D1_difference_matrix,
                           "dec": PenaltyMatrix().D1_difference_matrix,
                           "conc": PenaltyMatrix().D2_difference_matrix, 
                           "conv": PenaltyMatrix().D2_difference_matrix,
                           "peak": None }
    
    def __init__(self, descr, n_param=20):
        """
        descr : tuple - ever entry describens one part of 
                        the model, e.g.
                        descr =( ("s(0)"", "smooth", 10),
                                 ("s(1)"", "inc", 10), 
                                (t(0,1), 5) 
                               )
                        with the scheme: (type of smooth, number of knots)
        
        !!! currently only smooths with BSpline basis !!!
        
        TODO:
            [ ] incorporate tensor product splines
        """
        self.description_str = descr
        self.description_dict = { t: (p, n) for t, p, n  in self.description_str}
        self.smooths = None
        self.coef_ = None
        
    def create_basis(self, X):
        """Create the unpenalized BSpline basis for the data X.
        
        Parameters:
        ------------
        X : np.ndarray - data
        n_param : int  - number of parameters for the spline basis 
        
        TODO:
            [x] include TPS
        
        """
        assert (len(self.description_str) == X.shape[1]),"Nr of smooths must match Nr of predictors!"
       
        # smooths without tensor product splines
        #self.smooths = [ 
        #    s(x_data=X[:, int(k[2])-1], 
        #      n_param=int(v[1]), 
        #      penalty=v[0]) for k, v in self.description_dict.items()
        #]    
        
        # smooths with tensor product splines
        self.smooths = list()
        for k,v in self.description_dict.items():
            if k[0] is "s":
                self.smooths.append(s(x_data=X[:,int(k[2])-1], n_param=int(v[1]), penalty=v[0]))
            elif k[0] is "t":
                self.smooths.append(tps(x_data=X[:, [int(k[2])-1, int(k[4])-1]], n_param=list(v[1]), penalty=v[0]))    
        
        self.basis = np.concatenate([smooth.basis for smooth in self.smooths], axis=1) 
        
        return 
    
    def create_penalty_block_matrix(self, beta_test=None):
        """Create the penalty block matrix specified in self.description_str.
        
        Looks like: ------------
                    |p1 0  0  0|  
                    |0 p2  0  0|
                    |0  0 p3  0|
                    |0  0  0 p4|
                    ------------
        where p_i is a a matrix according to the specified penalty.

        Parameters:
        ---------------
        X : np.ndarray  - data
        
        TODO:
            [x]  include the weights !!! 
            [ ]  include TPS penalty
        
        """
        assert (self.smooths is not None), "Run Model.create_basis() first!"
        
        if beta_test is None:
            beta_test = np.zeros(self.basis.shape[1])
        
        idx = 0      
        penalty_matrix_list = []
        
        for smooth in self.smooths:
            
            n = smooth.basis.shape[1]
            b = beta_test[idx:idx+n]
            
            D = smooth.penalty_matrix
            V = check_constraint(beta=b, constraint=smooth.penalty)

            penalty_matrix_list.append(D.T @ V @ D )
            idx += n
            
        #self.penalty_matrix_list_and_weight = np.concatenate(penalty_matrix_list, axis=1)
        self.penalty_matrix_list = penalty_matrix_list
        self.penalty_block_matrix = block_diag(*penalty_matrix_list)

        return
       
    def calc_y_pred_and_mse(self, y):
        """Calculates y_pred and prints the MSE.
        
        Parameters:
        --------------
        y : array    - target values for training data.
        """
        assert (self.coef_ is not None), "Model is untrained, run Model.fit(X, y) first!"
        y_fit = self.basis @ self.coef_
        mse = mean_squared_error(y, y_fit)
        print(f"Mean squared error on data for unconstrained LS fit: {np.round(mse, 4)}")
        return y_fit, mse
    
    
    def fit(self, X, y, lam_c=1, plot_=True, max_iter=5):
        """Lstsq fit using Smooths.
        
        Parameters:
        -------------
        X : pd.DataFrame or np.ndarray
        y : pd.DataFrame or np.array
        plot_ : boolean
        
        TODO:
            [x] check constraint violation in the iterative fit
            [ ] incorporate TPS in the iterative fit
        """
        
        
        # create the basis for the initial fit without penalties
        self.create_basis(X)    

        fitting = lstsq(a=self.basis, b=y, rcond=None)
        beta_0 = fitting[0].ravel()
        self.coef_ = beta_0
        self.calc_y_pred_and_mse(y=y)
        
        # check constraint violation
        v_old = check_constraint_full_model(self)
        
        # create dataframe to save the beta values 
        colN = [ f"b_{i}" for i in range(len(beta_0))]        
        df_beta = pd.DataFrame(columns=colN)
        d = dict(zip(colN, beta_0))
        df_beta = df_beta.append(pd.Series(d), ignore_index=True)
        
        beta = np.copy(beta_0)
        for i in range(max_iter):
            print("Create basis with penalty and weight")
            self.create_penalty_block_matrix(beta_test=beta)
            
            print("Least squares fit iteration ", i+1)
            B = self.basis
            D_c = self.penalty_block_matrix
        
            BB = B.T @ B
            DVD = lam_c * D_c.T @ D_c
            By = B.T @ y
            
            beta_new = (np.linalg.pinv(BB + DVD) @ By).ravel()

            self.calc_y_pred_and_mse(y=y)
            
            # create dict
            d = dict(zip(colN, beta_new))
            df_beta = df_beta.append(pd.Series(d), ignore_index=True)
            
            
            # check constraint violation
            v_new = check_constraint_full_model(self)
            #px.imshow(np.diag(v_new)).show()
            
            delta_v = np.sum(v_new - v_old)
            if delta_v == 0:
                print("Iteration converged!")
                break
            else:
                v_old = v_new                
                beta = beta_new
                print("\n Violated constraints: ", np.sum(v_new))
            
        self.df_beta = df_beta
        self.coef_ = self.df_beta.iloc[-1].values
        
        y_fit = self.basis @ self.coef_
    
        self.mse = mean_squared_error(y, y_fit)
        print(f"Mean squared error on the data: {np.round(self.mse, 4)}")
        
        if plot_:
            dim = X.shape[1]
            if dim == 1:
                fig = self.plot_xy(x=X[:,0], y=y.ravel(), name="Data")
                fig.add_trace(go.Scatter(x=X[:,0], y=y_fit, name="Fit", mode="markers"))
            elif dim == 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=y.ravel(), name="Data", mode="markers"))
                fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=y_fit, name="Fit", mode="markers"))

                
            fig.update_traces(
                marker=dict(
                    size=12, 
                    line=dict(width=2, color='DarkSlateGrey')),
                selector=dict(mode='markers'))
            fig.show()
            
        return 
    
    # not trusted
    def predict(self, X, y=None, plot_=False):
        """Prediction of the trained model on the data in X."""
        if self.coef_ is None:
            print("Model untrained!")
            return
        
        self.create_basis(X, penalty=None)
        # prediction_basis, pad_y = self.multiple_smooths(X.values)
        #if pad_y is False:
        #    pass
        #else:
        #    prediction_basis = prediction_basis[:-len(pad_y)]
        
        print("Shape prediction basis: ", self.basis.shape)
        print("Shape coef_: ", self.coef_.shape)
        pred = self.basis @ self.coef_
        if plot_:
            fig = self.plot_xy(x=X[:,0], y=pred, name="Prediction")
            if type(y) is not None:
                print("shape y: ", y.shape)
                fig.add_trace(go.Scatter(x=X[:,0], y=y.reshape((-1,)), name="Data", mode="markers"))
            fig.show()
        return pred
    
    def plot_xy(self, x, y, title="Titel", name="Data", xlabel="xlabel", ylabel="ylabel"):
        """Basic plotting function."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="markers"))
        fig.update_layout(title=title)
        fig.update_xaxes(title=xlabel)
        fig.update_yaxes(title=ylabel)
        return fig
    
    def plot_basis(self, matrix):
        """Plot the matrix."""
        fig = go.Figure(go.Image(z=matrix)).show()
        return
                        
    
def check_constraint(beta, constraint, print_idx=False):
    """Checks if beta fits the constraint."""
    V = np.zeros((len(beta), len(beta)))
    b_diff = np.diff(beta)
    b_diff_diff = np.diff(b_diff)
    if constraint is "inc":
        v = [0 if i > 0 else 1 for i in b_diff] 
    elif constraint is "dec":
        v = [0 if i < 0 else 1 for i in b_diff] 
    elif constraint is "conv":
        v = [0 if i > 0 else 1 for i in b_diff_diff]
    elif constraint is "conc":
        v = [0 if i < 0 else 1 for i in b_diff_diff]
    elif constraint is "no":
        v = np.zeros(len(beta))
    elif constraint is "smooth":
        v = np.ones(len(b_diff_diff))
    else:
        print(f"Constraint [{constraint}] not implemented!")
        return
    
    V = np.diag(v)
    if print_idx:
        print("Constraint violated at the following indices: ")
        print([idx for idx, n in enumerate(v) if n == 1])
    return V
    

def check_constraint_full_model(model):
    """Checks if the coefficients in the model violate the given constraints.
    
    Parameters:
    -------------
    model : class Model() 
    
    Returns:
    -------------
    v : list   - list of boolean wheter the constraint is violated. 
    """

    v = []
    n_coef_list = np.array([smooth.n_param for smooth in model.smooths])
    n_coef_cumsum = np.append(0, np.cumsum(n_coef_list))
    
    for i, smooth in enumerate(model.smooths):
        beta = model.coef_[n_coef_cumsum[i]:n_coef_cumsum[i+1]]
        penalty = smooth.penalty
        V = check_constraint(beta, constraint=penalty)
        v += list(np.diag(V))
    
    return np.array(v, dtype=np.int)    
    
def bar_chart_of_coefficient_difference_dataframe(df):
    """Takes the dataframe Model.df_beta and plots a bar chart of the rows. """

    fig = go.Figure()
    x = np.arange(df.shape[1]-1)
    xx = df.columns[1:]
    
    for i in range(df.shape[0]):
        fig.add_trace(go.Bar(x=xx, y=np.diff(df.iloc[i]), name=f"Iteration {i}"))
        
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=20
    )
    fig.update_layout(title="Difference in coefficients", )
    fig.show()        
        
def line_chart_of_coefficient_dataframe(df):
    """Takes the dataframe Model.df_beta and plots a line chart of the rows. """

    fig = go.Figure()
    x = np.arange(df.shape[1])

    for i in range(df.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=df.iloc[i], name=f"Iteration {i}",
                                mode="lines"))

    fig.update_layout(title="Coefficients at different iterations",)
    fig.show()


# In[ ]:




