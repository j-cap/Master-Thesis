
"""

First tries of some toolboxes and algorithms
    - TensorFlow Lattice

One Dim was tried, there is some confusig error


Author: jweber
Date: 10.03.2020
"""

#%% Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import logging
import numpy as np
import pandas as pd
import sys
import tensorflow_lattice as tfl
from tensorflow import feature_column as fc
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

logging.disable(sys.maxsize)
pio.renderers.default = "browser"

#%% load data
df = pd.read_csv('../../Data/data_1D-tan-lin-cos2.csv', dtype=np.float64)
X, y = df["t"].values, df["Exp_1"].values
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
df.head()
#%% first plotting
# scatter raw data
fig = px.scatter(df,x="t", y="Exp_1", trendline="lowess")
# update the markers
fig.update_traces(marker=dict(size=2,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()

# %% set Constants
LEARNING_RATE = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 500

#%% get train and test dataframes
X_train_pd = pd.DataFrame(X_train, columns=["x"])
X_test_pd = pd.DataFrame(X_test, columns=["x"])
y_train_pd = pd.DataFrame(y_train, columns=["y(x)"])
y_test_pd = pd.DataFrame(y_test, columns=["y(x)"])


#%% set Feature colums:
feature_columns = [fc.numeric_column("x")]

#%% Creating input_fns
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_train_pd,
    y=y_train_pd,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    num_threads=1
)
# feature_analysis_input_fn is used to collect statistics about the input.
feature_analysis_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_train_pd,
    y=y_train_pd,
    shuffle=False,
    batch_size=BATCH_SIZE,
    # we only need one pass over the data
    num_epochs=1,
    num_threads=1
)

test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_test_pd,
    y=y_test_pd,
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
#%% set Feature Configs
# Feature calibration and per-feature configurations are set using 
#   tfl.configs.FeatureConfig. Feature configurations include mono-
#   tonicity constraints, per-feature regularization 
#   (see tfl.configs.RegularizerConfig), and lattice sizes for lattice models.

# Feature configs are used to specify how each feature is calibrated and used
feature_configs = [
    tfl.configs.FeatureConfig(
        name="x",
        lattice_size=10,
        # By default, input keypoints of pwl are quantiles of the features
        pwl_calibration_num_keypoints=10,
        monotonicity="increasing",
        # pwl_calibration_clip_max=1,
        # pwl_calibration_convexity="concave",
        # Per feature regularization
        regularizer_configs=[
            tfl.configs.RegularizerConfig(name="calib_wrinkle", l2=0.1),
        ],
    ),
    # tfl.configs.FeatureConfig(
    #     name="QDot",
    #     pwl_calibration_num_keypoints=100,
    #     monotonicity="increasing",
    #     lattice_size=100,
    # ),
]
#%% set model config
model_config = tfl.configs.CalibratedLinearConfig(
    feature_configs=feature_configs,
    use_bias=True,
    output_calibration=True,
    regularizer_configs=[
        # Regularizer for the output calibrator.
        tfl.configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
    ])

#%% set estimator
estimator = tfl.estimators.CannedRegressor(
    feature_columns=feature_columns,
    model_config=model_config,
    feature_analysis_input_fn=feature_analysis_input_fn,
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    config=tf.estimator.RunConfig(tf_random_seed=42)
)   
# Training Calibrated Lattice Model
#model_config = tfl.configs.CalibratedLinearConfig(
#    feature_configs=feature_configs,
    #num_lattices=1,
    #lattice_rank=2,
    #regularizer_configs=[
        # Torsion regularizer applied to the lattice to make it more linear.
        # tfl.configs.RegularizerConfig(name='torsion', l2=1e-4),
        # Globally defined calibration regularizer is applied to all features.
    #    tfl.configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
    #]
#)


#%% Training Calibrated Linear Model
estimator.train(input_fn=train_input_fn)
results = estimator.evaluate(input_fn=test_input_fn)
print('Prediction mean: {}'.format(results['prediction/mean']))

#%%
saved_model_path = estimator.export_saved_model(estimator.model_dir,
                                                serving_input_fn)
model_graph = tfl.estimators.get_model_graph(saved_model_path)
tfl.visualization.draw_model_graph(model_graph, calibrator_dpi=100)


