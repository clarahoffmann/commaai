######### Train Precise Learner #######
# Author: Clara Hoffmann
# Last changed: 12.01.2021
# With this code you can train the precise learner
# using a tfestimator that trains on the training shards
# created previously.
# The models are saved as checkpoint files in the model dir.
# Stopping is done by monitoring the validation loss
# by loading the checkpoint files into tensorboard.
# Stop the model when the validation loss shows no
# further improvement

import keras
import glob 
import os
import tensorflow as tf
import random
from utils import imgs_input_fn2, imgs_input_fn_val, rmse, build_model, _parse_function_train, _parse_function_val

#define paths 
# directory where checkpoints should be saved
model_dir = '../../../data/models/copula_cil/'
# directory where shards were saved
shard_path = '../../../data/commaai/training_files/unfiltered_smaller_images_y/'
val_path = '../../../data/commaai/test_files/'

c = 300000
EPOCHS = 50
STEPS = c*EPOCHS

# read in shards from shard directory
shard_files = glob.glob(os.path.join(shard_path, "*.tfrecords")) 
print('training with ' + str(len(shard_files)) + ' shards.')

val_files = ['../../../data/commaai/test_files/val.tfrecords']

# shuffle shards
random.shuffle(shard_files)

# build keras model
keras_model = build_model()

# specify optimizer
adam_optimizer = keras.optimizers.Adam(learning_rate=0.00025)

# compile model
keras_model.compile(
    loss = 'mse',
    optimizer = adam_optimizer,
    metrics=[rmse, 'mse', 'mae'])

# transform keras model to tf estimator
keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model, model_dir = model_dir,
      config = tf.estimator.RunConfig(
          log_step_count_steps=10, 
          save_summary_steps = 10,
          save_checkpoints_steps = 1000,
          keep_checkpoint_max=2000))


# generators for training and validation data
cust_train_input_fn = lambda: imgs_input_fn2(shard_files)
cust_val_input_fn = lambda: imgs_input_fn_val(val_files)

# specify training and evaluation spec
train_spec = tf.estimator.TrainSpec(input_fn = cust_train_input_fn,
                                    max_steps = STEPS)
eval_spec = tf.estimator.EvalSpec(input_fn = cust_val_input_fn, 
                                  steps = 500) 
# train and evaluate model
tf.estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)