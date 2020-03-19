"""

First tries of some toolboxes and algorithms

- TensorFlow Lattice

Author: jweber
Date: 18.03.2020
"""

#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import pandas as pd 
import tensorflow as tf
import tensorflow_lattice as tfl
import itertools
import logging
import matplotlib
import sys
from tensorflow import feature_column as fc
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
from data_preprocessing import load_normalize_data
logging.disable(sys.maxsize)

# for GraphViz
import os 
os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\Graphviz2.38\\bin"
# from Code import data_preprocessing 

LEARNING_RATE = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 500
PREFITTING_NUM_EPOCHS = 10

data, descriptor = load_normalize_data(normalize=True)
data = data.rename(columns={
    "QDot[l/min*m]":"QDot",
    "TemperaturStart":"Temp"
    }
)
data_train, data_test = train_test_split(
    data, test_size=0.3, random_state=42, shuffle=True
)
# data_train, data_val = train_test_split(
#     data_train, test_size=0.2, random_state=42, shuffle=True
# )

#%%
# plotting 
fig = plt.figure(figsize=(11,9))
ax = plt.axes(projection='3d')
nr = 2

# ax.scatter3D(data["Temp"], data["QDot"], data["HTC"], c=data["HTC"], facecolors=None) #, cmap='Spectral')
ax.scatter3D(data_test["Temp"], data_test["QDot"], data_test["HTC"], c='r', marker='d', label="Test")
ax.scatter3D(data_train["Temp"], data_train["QDot"], data_train["HTC"], c='k', marker='x', label="Train")

ax.set_xlabel("Temp")
ax.set_ylabel("QDot")
ax.set_zlabel("HTC")
plt.legend()
plt.show()

data_train.head()

#%%
# Feature Columns: As for any other TF estimator, data needs to be passed to the estimator, 
#      which is typically via an input_fn and parsed using FeatureColumns.

feature_columns = [
    fc.numeric_column("Temp"),
    fc.numeric_column("QDot")
]

# creating input_fn: As for any other estimator, you can use an input_fn to feed data to the 
#     model for training and evaluation. TFL estimators automatically calculate quantiles of
#     the features and use them as input keypoints for the PWL calibration layer. To do so, 
#     they require passing a feature_analysis_input_fn, which is similar to the training 
#     input_fn but with a single epoch or a subsample of the data.
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=data_train[data_train.columns[:2]],
    y=data_train["HTC"],
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    num_threads=1
)

# feature_analysis_input_fn is used to collect statistics about the input.
feature_analysis_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=data_train[data_train.columns[:2]],
    y=data_train["HTC"],
    shuffle=False,
    batch_size=BATCH_SIZE,
    # we only need one pass over the data
    num_epochs=1,
    num_threads=1
)

test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=data_test[data_test.columns[:2]],
    y=data_test["HTC"],
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_epochs=1,
    num_threads=1
)

# serving_input_fn is used to create saved models
serving_input_fn = (
    tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec=fc.make_parse_example_spec(feature_columns)
    )
)

# Feature calibration and per-feature configurations are set using 
#   tfl.configs.FeatureConfig. Feature configurations include mono-
#   tonicity constraints, per-feature regularization 
#   (see tfl.configs.RegularizerConfig), and lattice sizes for lattice models.

# Feature configs are used to specify how each feature is calibrated and used
feature_configs = [
    tfl.configs.FeatureConfig(
        name="Temp",
        lattice_size=2,
        # By default, input keypoints of pwl are quantiles of the features
        pwl_calibration_num_keypoints=20,
        # monotonicity="increasing",
        # pwl_calibration_clip_max=1,
        pwl_calibration_convexity="concave",
        # Per feature regularization
        regularizer_configs=[
            tfl.configs.RegularizerConfig(name="calib_wrinkle", l2=0.1),
        ],
    ),
    tfl.configs.FeatureConfig(
        name="QDot",
        pwl_calibration_num_keypoints=20,
        monotonicity="increasing",
    ),
]

#%%
# Training Calibrated Lattice Model
model_config = tfl.configs.CalibratedLatticeConfig(
    feature_configs=feature_configs,
    regularizer_configs=[
        # Torsion regularizer applied to the lattice to make it more linear.
        # tfl.configs.RegularizerConfig(name='torsion', l2=1e-4),
        # Globally defined calibration regularizer is applied to all features.
        tfl.configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
    ]
)

# A CannedClassifier is constructed from the givel model config
estimator = tfl.estimators.CannedRegressor(
    feature_columns=feature_columns,
    model_config=model_config,
    feature_analysis_input_fn=feature_analysis_input_fn,
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    config=tf.estimator.RunConfig(tf_random_seed=42)
)
estimator.train(input_fn=train_input_fn)
results = estimator.evaluate(input_fn=test_input_fn)
print('Prediction mean: {}'.format(results['prediction/mean']))
saved_model_path = estimator.export_saved_model(estimator.model_dir,
                                                serving_input_fn)
model_graph = tfl.estimators.get_model_graph(saved_model_path)
tfl.visualization.draw_model_graph(model_graph, calibrator_dpi=100)


# %%

# visualize prediction test set
def get_predictions(estimator, input_fn):
    pred_dicts = list(estimator.predict(input_fn))
    preds = pd.Series([pred['predictions'][0] for pred in pred_dicts])
    return preds

pred = get_predictions(estimator=estimator, input_fn=test_input_fn)

fig = plt.figure(figsize=(11,9))
print(data_test["HTC"].values.shape)
print(pred.shape)

plt.plot(data_test["HTC"].values, pred, marker='x', linestyle='')
plt.xlabel("True data")
plt.ylabel("Prediction")        
plt.xlim((0, data_test["HTC"].max()))
plt.ylim((0, data_test["HTC"].max()))
plt.show()
#%%
#---------------------------------------------------------------------------------------------------------
# %%
# plot grid of prediction vs real data

nr_points = 100
grid = np.linspace(0,1,nr_points)
X,Y = np.meshgrid(grid, grid)

grid_data = pd.DataFrame(data={
    'qDotGrid':X.flatten(), 
    'tempGrid':Y.flatten(), 
    "zGrid": np.zeros(nr_points*nr_points)
    })

#---------------------------------------------------------------------------------------------------------
# def input_fn(features, batch_size=BATCH_SIZE):
#     """An input function for prediction."""
#     # Convert the inputs to a Dataset without labels.
#     return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

# Z = estimator.predict(input_fn=lambda : input_fn(grid_data))
# prediction = np.zeros(nr_points*nr_points)
# for i,p in enumerate(Z):
#     print(f"i: {i} -- p: {p}")
#     prediction[i] = p["predictions"]

#%%
#---------------------------------------------------------------------------------------------------------
grid_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=grid_data[grid_data.columns[:2]],
    y=grid_data["zGrid"],
    shuffle=False,
    num_epochs=1,
    num_threads=1
)
Z = get_predictions(estimator=estimator, input_fn=grid_input_fn)
#%%
#---------------------------------------------------------------------------------------------------------
from mpl_toolkits import mplot3d
fig = plt.figure(figsize=(11,9))
ax = plt.axes(projection='3d')
ax.plot_surface(Y, X, Z, cmap='viridis', edgecolor='none')
ax.scatter(data_test["Temp"], data_test["QDot"], data_test["HTC"], c='k', marker='x', label="Predictions on test data")
plt.legend()
ax.set_title('Surface plot')
plt.show()


# %%
