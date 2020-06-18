#!/usr/bin/env python
# coding: utf-8

# In[19]:


# convert jupyter notebook to python script
#get_ipython().system('jupyter nbconvert --to script Model_Notebook.ipynb')


# In[14]:


import plotly.express as px
import plotly.graph_objects as go
import numpy as np
np.random.seed(42)
import pandas as pd
from numpy.linalg import lstsq
from scipy.linalg import block_diag
from scipy.signal import find_peaks
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
                        descr =( ("s(1)", "smooth", 10, 1),
                                 ("s(2)", "inc", 10, 1), 
                                 ("t(1,2)", "tps", [5,5], 1) 
                               )
                        with the scheme: (type of smooth, number of knots)
               
        TODO:
            [x] incorporate tensor product splines
        """
        self.description_str = descr
        self.description_dict = {
            t: {"constraint": p, "n_param": n, 
                "lambda" : {"smoothness": l[0], "constraint": l[1]}
               } 
            for t, p, n, l  in self.description_str}
        self.smooths = None
        self.coef_ = None
        self.y = None
        self.X = None
        
    def create_basis(self, X, y=None):
        """Create the unpenalized BSpline basis for the data X.
        
        Parameters:
        ------------
        X : np.ndarray - data
        n_param : int  - number of parameters for the spline basis 
        y : np.ndarray or None  - For peak/valley penalty. 
                                  Catches assertion if None and peak or valley penalty. 
        TODO:
            [x] include TPS
        
        """
        
        self.smooths = list()
        self.y = y.ravel()
        
        for k,v in self.description_dict.items():
            if k[0] is "s":
                self.smooths.append(
                    s(
                        x_data=X[:,int(k[2])-1], 
                        n_param=v["n_param"], 
                        penalty=v["constraint"], 
                        y_peak_or_valley=y.ravel(),
                        lam_s=v["lambda"]["smoothness"],
                        lam_c=v["lambda"]["constraint"]
                    )
                )
            elif k[0] is "t":
                self.smooths.append(
                    tps(
                        x_data=X[:, [int(k[2])-1, int(k[4])-1]], 
                        n_param=list(v["n_param"]), 
                        penalty=v["constraint"],
                        lam_s=v["lambda"]["smoothness"],
                        lam_c=v["lambda"]["constraint"]
                    )
                )    
        
        self.basis = np.concatenate([smooth.basis for smooth in self.smooths], axis=1) 
               
        self.smoothness_penalty_list = [np.sqrt(s.lam_smooth) * s.Smoothness_matrix() for s in self.smooths]
        self.smoothness_penalty_matrix = block_diag(*self.smoothness_penalty_list)

        n_coef_list = [0] + [np.product(smooth.n_param) for smooth in self.smooths]
        n_coef_cumsum = np.cumsum(n_coef_list)
        self.coef_list = n_coef_cumsum
        
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
        beta_test  : array  - Test beta for sanity checks.
        lam_c      : array  - Array of lambdas for the different constraints. 
        
        TODO:
            [x]  include the weights !!! 
            [x]  include TPS smoothnes penalty
            [ ]  include TPS shape penalty
        
        """
        assert (self.smooths is not None), "Run Model.create_basis() first!"
        assert (self.y is not None), "Run Model.fit(X,y) first!"
        
        if beta_test is None:
            beta_test = np.zeros(self.basis.shape[1])
        
        idx = 0      
        penalty_matrix_list = []
        
        for smooth in self.smooths:
            
            n = smooth.basis.shape[1]
            b = beta_test[idx:idx+n]
            
            D = smooth.penalty_matrix
            V = check_constraint(beta=b, constraint=smooth.penalty, y=self.y, basis=smooth.basis)
            
            penalty_matrix_list.append(smooth.lam_constraint * D.T @ V @ D )
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
        
        self.X, self.y = X, y.ravel()
        # create the basis for the initial fit without penalties
        self.create_basis(X=X, y=y)    

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
            D_s = self.smoothness_penalty_matrix
        
            BB = B.T @ B
            By = B.T @ y

            # user defined constraint
            DVD = D_c.T             
            # smoothing constraint
            DD = D_s.T @ D_s
            
            beta_new = (np.linalg.pinv(BB + DD + DVD) @ By).ravel()

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
                    size=8, 
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
                        
    
def check_constraint(beta, constraint, print_idx=False, y=None, basis=None):
    """Checks if beta fits the constraint.
    
    Parameters:
    ---------------
    beta  : array     - Array of coefficients to be tested against the constraint.
    constraint : str  - Name of the constraint.
    print_idx : bool  - .
    y  : array        - Array of data for constraint "peak" and "valley".
    basis : ndarray   - Matrix of the basis for constraint "peak" and "valley".
    
    Returns:
    ---------------
    V : ndarray       - Matrix with 1 where the constraint is violated, 0 else.
    
    """
    V = np.zeros((len(beta), len(beta)))
    b_diff = np.diff(beta)
    b_diff_diff = np.diff(b_diff)
    if constraint is "inc":
        v = [0 if i > 0 else 1 for i in b_diff] + [0]
    elif constraint is "dec":
        v = [0 if i < 0 else 1 for i in b_diff] + [0]
    elif constraint is "conv":
        v = [0 if i > 0 else 1 for i in b_diff_diff] + [0,0]
    elif constraint is "conc":
        v = [0 if i < 0 else 1 for i in b_diff_diff] + [0,0]
    elif constraint is "no":
        v = list(np.zeros(len(beta), dtype=np.int))
    elif constraint is "smooth":
        v = list(np.ones(len(b_diff_diff), dtype=np.int)) + [0,0]
    elif constraint is "tps":
        v = list(np.ones(len(beta), dtype=np.int))
    elif constraint is "peak":
        assert (y is not None), "Include y in check_constraints for penalty=[peak]"
        assert (basis is not None), "Include basis in check_constraints for penalty=[peak]"

        peak, properties = find_peaks(x=y, distance=int(len(y)))
        border = np.argwhere(basis[peak,:] > 0)
        left_border_spline_idx = int(border[0][1])
        right_border_spline_idx = int(border[-1][1])
        v_inc = [0 if i > 0 else 1 for i in b_diff[:left_border_spline_idx]]
        v_dec = [0 if i < 0 else 1 for i in b_diff[right_border_spline_idx:]]
        v_plateau = np.zeros(right_border_spline_idx - left_border_spline_idx + 1)
        v = np.concatenate([v_inc, v_plateau, v_dec]) 
        
    elif constraint is "valley":
        assert (y is not None), "Include y in check_constraints for penalty=[valley]"
        assert (basis is not None), "Include basis in check_constraints for penalty=[peak]"

        peak, properties = find_peaks(x= -1*y, distance=int(len(y)))
        border = np.argwhere(basis[peak,:] > 0)
        left_border_spline_idx = int(border[0][1])
        right_border_spline_idx = int(border[-1][1])
        v_dec = [0 if i < 0 else 1 for i in b_diff[:left_border_spline_idx:]]
        v_inc = [0 if i > 0 else 1 for i in b_diff[right_border_spline_idx:]]
        v_plateau = np.zeros(right_border_spline_idx - left_border_spline_idx + 1)
        v = np.concatenate([v_dec, v_plateau, v_inc])
    
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

    assert (model.coef_ is not None), "Please run Model.fit(X, y) first!"
    v = []

    for i, smooth in enumerate(model.smooths):
        beta = model.coef_[model.coef_list[i]:model.coef_list[i+1]]
        penalty = smooth.penalty
        V = check_constraint(beta, constraint=penalty, y=model.y, basis=model.basis)
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




