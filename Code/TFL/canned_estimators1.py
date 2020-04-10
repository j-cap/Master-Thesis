"""

From: https://www.tensorflow.org/lattice/tutorials/canned_estimators 
"""

#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import logging
import numpy as np
import pandas as pd
import sys
import tensorflow_lattice as tfl
from tensorflow import feature_column as fc
logging.disable(sys.maxsize)
#%%

csv_file = tf.keras.utils.get_file(
    'heart.csv', 'http://storage.googleapis.com/applied-dl/heart.csv')
df = pd.read_csv(csv_file)
target = df.pop('target')
train_size = int(len(df) * 0.8)
train_x = df[:train_size]
train_y = target[:train_size]
test_x = df[train_size:]
test_y = target[train_size:]
df.head()


# %%
LEARNING_RATE = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 500
PREFITTING_NUM_EPOCHS = 10
#%%
# Feature columns.
# - age
# - sex
# - cp        chest pain type (4 values)
# - trestbps  resting blood pressure
# - chol      serum cholestoral in mg/dl
# - fbs       fasting blood sugar > 120 mg/dl
# - restecg   resting electrocardiographic results (values 0,1,2)
# - thalach   maximum heart rate achieved
# - exang     exercise induced angina
# - oldpeak   ST depression induced by exercise relative to rest
# - slope     the slope of the peak exercise ST segment
# - ca        number of major vessels (0-3) colored by flourosopy
# - thal      3 = normal; 6 = fixed defect; 7 = reversable defect
feature_columns = [
    fc.numeric_column('age', default_value=-1),
    fc.categorical_column_with_vocabulary_list('sex', [0, 1]),
    fc.numeric_column('cp'),
    fc.numeric_column('trestbps', default_value=-1),
    fc.numeric_column('chol'),
    fc.categorical_column_with_vocabulary_list('fbs', [0, 1]),
    fc.categorical_column_with_vocabulary_list('restecg', [0, 1, 2]),
    fc.numeric_column('thalach'),
    fc.categorical_column_with_vocabulary_list('exang', [0, 1]),
    fc.numeric_column('oldpeak'),
    fc.categorical_column_with_vocabulary_list('slope', [0, 1, 2]),
    fc.numeric_column('ca'),
    fc.categorical_column_with_vocabulary_list(
        'thal', ['normal', 'fixed', 'reversible']),
]
#%%
# Creating input_fn
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=train_x,
    y=train_y,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    num_threads=1)

# feature_analysis_input_fn is used to collect statistics about the input.
feature_analysis_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=train_x,
    y=train_y,
    shuffle=False,
    batch_size=BATCH_SIZE,
    # Note that we only need one pass over the data.
    num_epochs=1,
    num_threads=1)

test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=test_x,
    y=test_y,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_epochs=1,
    num_threads=1)

# Serving input fn is used to create saved models.
serving_input_fn = (
    tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec=fc.make_parse_example_spec(feature_columns)))

# %%
# Feature configs are used to specify how each feature is calibrated and used.
feature_configs = [
    tfl.configs.FeatureConfig(
        name='age',
        lattice_size=3,
        # By default, input keypoints of pwl are quantiles of the feature.
        pwl_calibration_num_keypoints=5,
        monotonicity='increasing',
        pwl_calibration_clip_max=100,
        # Per feature regularization.
        regularizer_configs=[
            tfl.configs.RegularizerConfig(name='calib_wrinkle', l2=0.1),
        ],
    ),
    tfl.configs.FeatureConfig(
        name='cp',
        pwl_calibration_num_keypoints=4,
        # Keypoints can be uniformly spaced.
        pwl_calibration_input_keypoints='uniform',
        monotonicity='increasing',
    ),
    tfl.configs.FeatureConfig(
        name='chol',
        # Explicit input keypoint initialization.
        pwl_calibration_input_keypoints=[126.0, 210.0, 247.0, 286.0, 564.0],
        monotonicity='increasing',
        # Calibration can be forced to span the full output range by clamping.
        pwl_calibration_clamp_min=True,
        pwl_calibration_clamp_max=True,
        # Per feature regularization.
        regularizer_configs=[
            tfl.configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
        ],
    ),
    tfl.configs.FeatureConfig(
        name='fbs',
        # Partial monotonicity: output(0) <= output(1)
        monotonicity=[(0, 1)],
    ),
    tfl.configs.FeatureConfig(
        name='trestbps',
        pwl_calibration_num_keypoints=5,
        monotonicity='decreasing',
    ),
    tfl.configs.FeatureConfig(
        name='thalach',
        pwl_calibration_num_keypoints=5,
        monotonicity='decreasing',
    ),
    tfl.configs.FeatureConfig(
        name='restecg',
        # Partial monotonicity: output(0) <= output(1), output(0) <= output(2)
        monotonicity=[(0, 1), (0, 2)],
    ),
    tfl.configs.FeatureConfig(
        name='exang',
        # Partial monotonicity: output(0) <= output(1)
        monotonicity=[(0, 1)],
    ),
    tfl.configs.FeatureConfig(
        name='oldpeak',
        pwl_calibration_num_keypoints=5,
        monotonicity='increasing',
    ),
    tfl.configs.FeatureConfig(
        name='slope',
        # Partial monotonicity: output(0) <= output(1), output(1) <= output(2)
        monotonicity=[(0, 1), (1, 2)],
    ),
    tfl.configs.FeatureConfig(
        name='ca',
        pwl_calibration_num_keypoints=4,
        monotonicity='increasing',
    ),
    tfl.configs.FeatureConfig(
        name='thal',
        # Partial monotonicity:
        # output(normal) <= output(fixed)
        # output(normal) <= output(reversible)        
        monotonicity=[('normal', 'fixed'), ('normal', 'reversible')],
    ),
]
#%%
# Model config defines the model structure for the estimator.
model_config = tfl.configs.CalibratedLinearConfig(
    feature_configs=feature_configs,
    use_bias=True,
    output_calibration=True,
    regularizer_configs=[
        # Regularizer for the output calibrator.
        tfl.configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
    ])
# A CannedClassifier is constructed from the given model config.
estimator = tfl.estimators.CannedClassifier(
    feature_columns=feature_columns[:5],
    model_config=model_config,
    feature_analysis_input_fn=feature_analysis_input_fn,
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    config=tf.estimator.RunConfig(tf_random_seed=42))
estimator.train(input_fn=train_input_fn)
results = estimator.evaluate(input_fn=test_input_fn)
print('Calibrated linear test AUC: {}'.format(results['auc']))

#%%
# estimator.predict() returns a generator !!!
pred = list(estimator.predict(input_fn=test_input_fn))
preds = [int(p["class_ids"]) for p in pred]
preds[0]

missclass = target - np.array(preds) 
#%%
saved_model_path = estimator.export_saved_model(estimator.model_dir,
                                                serving_input_fn)
model_graph = tfl.estimators.get_model_graph(saved_model_path)
tfl.visualization.draw_model_graph(model_graph)


#%%