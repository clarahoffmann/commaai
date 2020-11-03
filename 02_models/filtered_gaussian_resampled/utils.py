import keras
from keras import backend
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
import numpy as np
import pandas as pd

# choose number of epochs
c = 55000
EPOCHS = 60
STEPS = c*EPOCHS


def _parse_function_train(proto):
    '''
    Reads in single training example and returns it in a format that the estimator can
    use

    Input: - single training examples written in tfrecords format

    Output: - image: tensor image (=covariates)
            - tr_label: associated z value

    '''
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
    image_shape = tf.stack([66, 200, 4])
    image_raw = parsed_example['image']

    label = tf.cast(parsed_example['label'], tf.float32)
    tr_label = tf.cast(parsed_example['tr_label'], tf.float32)
    image = tf.io.decode_raw(image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)

    image = tf.reshape(image, image_shape)[:,:,0:3]/255 

    return {'image':image}, tr_label
    
def imgs_input_fn(filenames, perform_shuffle = True, repeat_count = EPOCHS, batch_size = 32): 
    
    '''
    Function to continuously generate fresh training batches and provide them to
    a tf estimator
    
    Inputs:
           - filenames: list of paths to training shards
           - perform_shuffle: bool, whether training examples within a batch should
                         be shuffled
           - repeat_count: how many times to repeat the dataset (= number of epochs)
           - batch_size: batch size for the data passed to the tf estimator
    
    Outputs:
           - batch_features: tensor batch of features
           - batch_labels: tensor batch of associated labels 
     
    For more information, see f.e. the blog post https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/
    '''
    
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

    image_shape = tf.stack([66, 200, 3])
    image_raw = parsed_example['image']

    label = tf.cast(parsed_example['label'], tf.float32)
    tr_label = tf.cast(parsed_example['tr_label'], tf.float32)
    image = tf.io.decode_raw(image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)

    image = tf.reshape(image, image_shape)/255 

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


def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def build_model():
    # use mae instead of mse since it correlates more with closed-loop performance
# cite: https://arxiv.org/pdf/2003.06404.pdf
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
    
    return(keras_model)

def find_closest_element(y: float, arr: np.ndarray):
    index = np.searchsorted(arr,y)
    if (index >= 1) & (index < arr.shape[0]):
        res = [arr[index - 1], arr[index]]
    elif (index < arr.shape[0]):
        return np.array(index)
    else:
        return np.array(index - 1)

    if res[0] == res[1]:
        return np.array(index - 1)
    else:
        diff_pre = np.abs(y-res[0])
        diff_aft = np.abs(y-res[1])
        if diff_pre == diff_aft:
            return np.array(index - 1)
        else:
            return index - 1 if diff_pre < diff_aft else index
        
def Fy(y, density):
    integral = density.loc[find_closest_element(y, density['axes']),'cdf']
    return(integral)  