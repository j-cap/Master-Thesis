"""

Cubic Splines according to the basis from the Books of Wahba (1990) and Gu (2002)
from the Book "GAM - An Introduction with R" 2006 by Simon Wood (Chapter 3)

For this basis: 
    b1(x) = 1, 
    b2(x) = x and 
    b_{i+2} = R(x, x*_i) for i = 1 . . . q − 2 where
        R(x, z) = [(z − 1/2)**2 − 1/12][(x − 1/2)2 − 1/12]/4 -
                  [(|x − z| − 1/2)**4 − 1/2 (|x − z| − 1/2)**2 + 7/240]/24. 


Date: 27.04.2020
"""
#%%
import numpy as np
from numpy.linalg import lstsq
import plotly.express as px
import scipy.linalg as slinalg

def cubicSplineFromWhaba(x, z):
    """ compute R(x,z) for a cubic spline on [0,1] according to Wahba [1990]"""
    R = ((z - 1/2)**2 - 1/12)*((x - 1/2)**2 - 1/12)/4 - \
        ((np.abs(x - z) - 1/2)**4 - 1/2*(np.abs(x - z) - 1/2)**2 + 7/240) /24
    return R

def modelMatrix(x, xk):
    """ set up model matrix for cubic penalized regression spline """
    q, n = len(xk) + 2, len(x) # q = number of parameters, n = number of data
    X = np.ones(shape=(n, q))
    X[:,1] = x
    for i in range(0,n):
        for j in range(2, q):
            X[i, j] = cubicSplineFromWhaba(x[i], xk[j-2])
    return X

def penaltyMatrix(xk, returnSqrt=False):
    """ 
    compute the penalty matrix for penalized B-Splines according to Wood (p.126) 
    return the square root of the matrix if returnSqrt is True 
    """
    q = len(xk) + 2 # number of parameters
    S = np.zeros(shape=(q,q))
    for i in range(0, q-2):
        for j in range(0, q-2):
            S[i+2, j+2] = cubicSplineFromWhaba(xk[i], xk[j])
    return S if returnSqrt else slinalg.sqrtm(S)

def modelMatrixPenalized(y, x, xk, lam=0):
    """
    Fit the penalized B-Spline with knots at xk to the data x with lam as penalty weight
    """
    q = len(xk) + 2 # number of parameters
    n = len(x) # number of data
    # create augmented model matrix
    Xa = np.concatenate((modelMatrix(x,xk),np.sqrt(lam)*penaltyMatrix(xk, returnSqrt=True)))
    y = np.concatenate((y, np.zeros((q,))))
    return y, Xa

#%% get data
Size = np.array([1.42,1.58,1.78,1.99,1.99,1.99,2.13,2.13,2.13,
2.32,2.32,2.32,2.32,2.32,2.43,2.43,2.78,2.98,2.98])
Wear = np.array([4.0,4.2,2.5,2.6,2.8,2.4,3.2,2.4,2.6,4.8,2.9,
3.8,3.0,2.7,3.1,3.3,3.0,2.8,1.7])

# standardize data
x = Size - np.min(Size)
x = x/np.max(x)

#%% compute the model matrix for regression splines
xk = np.arange(0.2,1,0.2)
X = modelMatrix(x, xk)
# compute least squares fit for regression splines
regSplines_LS = lstsq(a=X, b=Wear)
# %% compute the penalty matrix S for penalized B-Splines (p. 126)
xk_p = np.arange(1,8) / 8
lam = [10, 0.01, 0.00000001]
mod = list()
[mod.append(modelMatrixPenalized(Wear, x, xk_p, lam=l)) for l in lam]
# Wear_aug, Xa = modelMatrixPenalized(Wear, x, xk_p, lam=0.1)
pSplines_LS_fit = list()
[pSplines_LS_fit.append(lstsq(a=m[1], b=m[0])) for m in mod]
# pSplines_LS = lstsq(a=Xa, b=Wear_aug)

#%% plot the prediction
xp = np.linspace(-0.1,1.1,100)
Xp = modelMatrix(xp, xk)
Xp_aug = modelMatrix(xp, xk_p)

#%%
fig = px.scatter(x=x, y=Wear)
fig.update_layout(
    xaxis_title="Scaled motor size",
    yaxis_title="Wear index")
fig.add_scatter(x=xp, y=np.dot(Xp, regSplines_LS[0]), mode="lines", name="BSpline fit")
#fig.add_scatter(x=xp, y=np.dot(Xp_aug, pSplines_LS[0]), mode="lines", name="PSpline fit, lam={}".format(0.1))
for i,fit in enumerate(pSplines_LS_fit):
    fig.add_scatter(x=xp, y=np.dot(Xp_aug,fit[0]), mode="lines", name="PSpline fit, lam={}".format(lam[i]))

fig.show()
# %%
# %%
