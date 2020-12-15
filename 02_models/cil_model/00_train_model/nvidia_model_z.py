import keras
import numpy as np
import glob 
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random
import cv2
from keras import backend
from scipy.stats import norm

#val_path = ['../../../../data/tfrecords/val/20200925/val_5000.tfrecords'
model_dir = '../../../data/models/20201021_unrestr_gaussian_resampled/'
shard_path = '../../../data/commaai/training_files/unrestricted_gauss_dens_resampled/'
val_path = '../../../data/commaai/test_files/'

c = 300000
EPOCHS = 50
STEPS = c*EPOCHS

shard_files = glob.glob(os.path.join(shard_path, "*.tfrecords")) 
print('training with ' + str(len(shard_files)) + ' shards.')
random.shuffle(shard_files)
#shard_files = ['../data/commaai/train_files_from_bag_test/0_new2.tfrecords']

#val_files = glob.glob(os.path.join(val_path, "*.tfrecords")) 
val_files = ['../../../data/commaai/test_files/_test.tfrecords']

random.shuffle(shard_files)

def _parse_function_train(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        'label': tf.io.FixedLenFeature([], tf.float32),
                        'rows': tf.io.FixedLenFeature([], tf.int64),
                        'cols': tf.io.FixedLenFeature([], tf.int64),
                        'depth': tf.io.FixedLenFeature([], tf.int64),
                        'tr_label': tf.io.FixedLenFeature([], tf.float32)
                       }

    # Load one example
    parsed_example = tf.io.parse_single_example(proto, keys_to_features)

    # fourth channel does not contain anything
    image_shape = tf.stack([874, 1164, 4])
    image_raw = parsed_example['image']

    label = tf.cast(parsed_example['label'], tf.float32)
    tr_label = tf.cast(parsed_example['tr_label'], tf.float32)
    image = tf.io.decode_raw(image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)

    image = tf.reshape(image, image_shape)
    image = tf.image.resize(image, [218, 291])[76:142, 45:245,0:3]
    image = image/255 

    return {'image':image}, tr_label
    
# is repeat count epochs ? i think yes
# source: https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
def imgs_input_fn(filenames, perform_shuffle = True, repeat_count = EPOCHS, batch_size = 32): 
    
    # reads in single training example and returns it in a format that the estimator can
    # use

    dataset = tf.data.TFRecordDataset(filenames = filenames)
    dataset = dataset.map(_parse_function_train)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    
    return batch_features, batch_labels

def _parse_function_val(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        'label': tf.io.FixedLenFeature([], tf.float32),
                        'rows': tf.io.FixedLenFeature([], tf.int64),
                        'cols': tf.io.FixedLenFeature([], tf.int64),
                        'depth': tf.io.FixedLenFeature([], tf.int64),
                        'tr_label': tf.io.FixedLenFeature([], tf.float32)
                       }

    # Load one example
    parsed_example = tf.io.parse_single_example(proto, keys_to_features)

    image_shape = tf.stack([874, 1164, 3])
    image_raw = parsed_example['image']

    label = tf.cast(parsed_example['label'], tf.float32)
    tr_label = tf.cast(parsed_example['tr_label'], tf.float32)
    image = tf.io.decode_raw(image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)

    image = tf.reshape(image, image_shape)
    image = tf.image.resize(image, [218, 291])[76:142, 45:245,0:3]
    image = image/255 

    return {'image':image}, tr_label
    
def imgs_input_fn_val(filenames, perform_shuffle = False, repeat_count = 1, batch_size = 100): 
    
    # reads in single training example and returns it in a format that the estimator can
    # use

    dataset = tf.data.TFRecordDataset(filenames = filenames)
    dataset = dataset.map(_parse_function_val)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    
    return batch_features, batch_labels
# batch norm should be between conv layers and dropout only between fully connected layers
# maybe try out a cyclical learning rate? https://www.machinecurve.com/index.php/2020/02/25/training-your-neural-network-with-cyclical-learning-rates/
# put dropout layer behind first dense layer
# subtract image mean to standardize images
# this allows more parameter sharing
# maybe less learning rate decay?
# use mae instead of mse since it correlates more with closed-loop performance
# cite: https://arxiv.org/pdf/2003.06404.pdf
# or thresholded relative error?
Input = tf.keras.layers.Input(shape=(66, 200, 3,), name='image')
x = Conv2D(24, kernel_size=(5, 5), activation='relu', strides=(2, 2))(Input)
x = BatchNormalization()(x)
x = Conv2D(36, kernel_size=(5, 5), activation='relu', strides=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(48, kernel_size=(5, 5), activation='relu', strides=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
#x = BatchNormalization()(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1164)(x)
x = Dropout(0.5)(x)
x = Dense(100)(x)
x = Dropout(0.5)(x)
x = Dense(50)(x) 
x = Dropout(0.2)(x)
x = Dense(10)(x)
Output = Dense(1, name = 'output_layer')(x)

keras_model = tf.keras.models.Model(
      inputs = [Input], outputs = [Output])

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

adam_optimizer = keras.optimizers.Adam(learning_rate=0.00025)
# old
#keras.optimizers.Adadelta(learning_rate = 0.001, rho = 0.95, epsilon = 1e-07, name = 'Adadelta')
keras_model.compile(
    loss = 'mse',
    optimizer = adam_optimizer,
    metrics=[rmse, 'mse', 'mae'])


keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model, model_dir = model_dir,
      config = tf.estimator.RunConfig(
          log_step_count_steps=10, 
          save_summary_steps = 10,
          save_checkpoints_steps = 1000,
          keep_checkpoint_max=2000))

cust_train_input_fn = lambda: imgs_input_fn(shard_files)
cust_val_input_fn = lambda: imgs_input_fn_val(val_files)

train_spec = tf.estimator.TrainSpec(input_fn = cust_train_input_fn, max_steps = STEPS)#, hooks = [early_stopping]
eval_spec = tf.estimator.EvalSpec(input_fn = cust_val_input_fn, 
                                  steps = 500) # , exporters = exporter for exporter, receiver must also be defined
# f.e. like here https://stackoverflow.com/questions/48269372/tensorflow-serving-input-receiver-fn-with-arguments

tf.estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)