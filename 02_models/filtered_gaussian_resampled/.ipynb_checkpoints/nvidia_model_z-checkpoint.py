# import packages
import keras
import numpy as np
import glob 
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
#import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
import random
from keras import backend
from utils import imgs_input_fn, imgs_input_fn_val, rmse, build_model, _parse_function_train, _parse_function_val

#define paths 
model_dir = '../../../data/models/20201027_filtered_gaussian_resampled/'
shard_path = '../../../data/commaai/training_files_filtered/tfrecords/'
val_path = '../../../data/commaai/test_files/val_file_filtered/'

# read in shards from shard directory
shard_files = glob.glob(os.path.join(shard_path, "*.tfrecords")) 
print('training with ' + str(len(shard_files)) + ' shards.')

# shuffle shards for better results
random.shuffle(shard_files)

# read in val file(s)
val_files = ['../../../data/commaai/test_files/val_file_filtered/val.tfrecords']

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
          save_checkpoints_steps = 2000,
          keep_checkpoint_max = 2000))

# generators for training and validation data
cust_train_input_fn = lambda: imgs_input_fn(shard_files)
cust_val_input_fn = lambda: imgs_input_fn_val(val_files)

c = 55000
EPOCHS = 60
STEPS = c*EPOCHS

# specify training and evaluation spec
train_spec = tf.estimator.TrainSpec(input_fn = cust_train_input_fn, max_steps = STEPS)
eval_spec = tf.estimator.EvalSpec(input_fn = cust_val_input_fn, 
                                  steps = 100) 

# train and evaluate model
tf.estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)