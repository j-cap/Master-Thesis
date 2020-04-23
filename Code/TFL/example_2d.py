
"""

First tries of some toolboxes and algorithms
    - TensorFlow Lattice

-> 2D example works with:
    - CalibratedLinearConfig: not shape constraint, monotonic, shape constraint
    - CalibratedLatticeConfig: 
Author: jweber
Date: 10.03.2020
"""

#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import time
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
from sklearn.metrics import mean_squared_error as mse
from tg_bot import send_TG_msg
logging.disable(sys.maxsize)
pio.renderers.default = "browser"

#%% load data
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
#%%
def surf_plot_prep():
    # only for surface plotting
    x0 = np.linspace(X_test["x0"].min(), X_test["x0"].max(), 100)
    x1 = np.linspace(X_test["x1"].min(), X_test["x1"].max(), 100)

    mx0, mx1 = np.meshgrid(x0, x1)
    df_surface_plot = pd.DataFrame(data={"x0":mx0.flatten(), "x1":mx1.flatten(), "z":np.zeros(len(mx0.flatten()))})

    input_fn_surface_plot = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=df_surface_plot[df_surface_plot.columns[:2]],
        y=df_surface_plot["z"],
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_epochs=1,
        num_threads=1
    )
    return input_fn_surface_plot, x0, x1
# %%
LEARNING_RATE = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 500

#%%
# Feature colums:
feature_columns = [
    fc.numeric_column("x0"),
    fc.numeric_column("x1")
]

#%%
# Creating input_fns
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    num_threads=1
)
# feature_analysis_input_fn is used to collect statistics about the input.
feature_analysis_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    shuffle=False,
    batch_size=BATCH_SIZE,
    # we only need one pass over the data
    num_epochs=1,
    num_threads=1
)

test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_test,
    y=y_test,
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
#%%
# Feature calibration and per-feature configurations are set using 
#   tfl.configs.FeatureConfig. Feature configurations include mono-
#   tonicity constraints, per-feature regularization 
#   (see tfl.configs.RegularizerConfig), and lattice sizes for lattice models.

# Feature configs are used to specify how each feature is calibrated and used
feature_configs = [
    tfl.configs.FeatureConfig(
        name="x0",
        lattice_size=100,
        # By default, input keypoints of pwl are quantiles of the features
        pwl_calibration_num_keypoints=100,
        unimodality="Peak",
        # pwl_calibration_clip_max=1,
        # pwl_calibration_convexity="concave",
        # Per feature regularization
        regularizer_configs=[
            tfl.configs.RegularizerConfig(name="calib_wrinkle", l2=0.01),
        ],
    ),
    tfl.configs.FeatureConfig(
        name="x1",
        lattice_size=50,
        monotonicity="increasing",
        pwl_calibration_num_keypoints=50,
        regularizer_configs=[
            tfl.configs.RegularizerConfig(name="calib_wrinkle", l2=0.01),
        ],
    )
    # tfl.configs.FeatureConfig(
    #     name="QDot",
    #     pwl_calibration_num_keypoints=100,
    #     monotonicity="increasing",
    #     lattice_size=100,
    # ),
]
#%%
# Training Calibrated Linear Model
# model_config = tfl.configs.CalibratedLinearConfig(
#     feature_configs=feature_configs,
#     use_bias=True,
#     output_calibration=True,
#     regularizer_configs=[
#         # Regularizer for the output calibrator.
#         tfl.configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
#    ])
#%% Training Calibrated Lattice Model
model_config = tfl.configs.CalibratedLatticeConfig(
    feature_configs=feature_configs,
    # use_bias=True,
    output_calibration_num_keypoints=25,
    regularizer_configs=[
        # Regularizer for the output calibrator.
        tfl.configs.RegularizerConfig(name='output_calib_hessian', l2=0.1),
    ])

#%% Initialize Calibrated Lattice Model
estimator = tfl.estimators.CannedRegressor(
    feature_columns=feature_columns,
    model_config=model_config,
    feature_analysis_input_fn=feature_analysis_input_fn,
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    config=tf.estimator.RunConfig(tf_random_seed=42)
)   
# 

#%% Train the model 
t = time.time()
estimator.train(input_fn=train_input_fn)
print(f"Training took: {time.time()-t} seconds!")
send_TG_msg("Finished Training")
results = estimator.evaluate(input_fn=test_input_fn)
#%%
saved_model_path = estimator.export_saved_model(estimator.model_dir,
                                                serving_input_fn)
model_graph = tfl.estimators.get_model_graph(saved_model_path)
tfl.visualization.draw_model_graph(model_graph, calibrator_dpi=100)

#%% Calc statistics and predictions for test set
pred = list(estimator.predict(input_fn=test_input_fn))
preds = [p["predictions"][0] for p in pred]
preds = pd.Series(data=preds,name="predictions")

mse_ = mse(y_true=y_test, y_pred=preds)
print("MSE: ", mse_)
rel_mse = mse_ / np.abs(y_test.min())
print("Rel. MSE: ", rel_mse)
#%% Surface plot calcualtions
input_fn_surface_plot, x0, x1 = surf_plot_prep()
pred_surface = list(estimator.predict(input_fn=input_fn_surface_plot))
preds_surface = np.array([p["predictions"][0] for p in pred_surface])
preds_surface = preds_surface.reshape((len(x0), len(x0)))

#%% Plot true vs prediction
# fig = px.scatter(x=y_test, y=preds)
# fig.update_layout(title="True Value vs. Prediction",
#                   yaxis_zeroline=False, xaxis_zeroline=False)

# fig.show()

#%% plot test prediction as surface over true data
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=X_test["x0"], y=X_test["x1"], z=y_test, mode='markers',name="True"))
# fig.add_trace(go.Scatter3d(x=X_test["x0"], y=X_test["x1"], z=preds, mode='markers', name="Prediction"))
fig.add_trace(go.Surface(z=preds_surface, x=x0, y=x1, name="Predictions"))
#(data=[go.Surface(z=preds.values)])
fig.show()



# %%

# %%
