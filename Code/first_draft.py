
"""

First tries of some toolboxes and algorithms

Author: jweber
Date: 09.03.2020
"""

#%% 
# Imports, data loading and train-test split
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
#%matplotlib notebook

data = pd.read_csv('../Data/Table_alpha_Data.txt', header=0, dtype=np.float64)
X, y = data.values[:,:2], data.values[:,2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

#%%
# plotting
# plotting
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,1][::10], X[:,0][::10], y[::10], c=y[::10], cmap='Spectral')
plt.show()
# pyGAM
from pygam import LinearGAM, s, te, PoissonGAM, f, GAM

gam = GAM(
    s(0, constraints="monotonic_inc", n_splines=15) + 
    s(1) + #, constraints="concave", n_splines=100) +
    te(1,0)
)
gam.fit(X_train,y_train)

titles = ['QDot[l/min*m]', 'TemperaturStart']
fig, axs = plt.subplots(1,len(titles))

# plot partial dependences
for i, ax in enumerate(axs):
    print("i = ", i)
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r')
    ax.set_title(titles[i])
plt.show()
#%%
# plot meshs
x1 = np.linspace(data.min()[0], data.max()[0], 30)
x2 = np.linspace(data.min()[1], data.max()[1], 30)

XX, YY = np.meshgrid(x1, x2)
Z = gam.predict(np.array([XX.flatten(), YY.flatten()]).T).reshape(30,30)

fig = plt.figure(figsize=(12,8))
ax = plt.axes(projection="3d")
ax.plot_wireframe(XX, YY, Z, color='green', label="Prediction")
ax.scatter(X[:,0], X[:,1], y, c='r', label="Data")
ax.set_xlabel('QDot')
ax.set_ylabel('Temp')
ax.set_zlabel('HTC')
ax.legend()
plt.show()
gam.summary()

# %%
