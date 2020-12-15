import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization

def imgs_input_fn(filepath , perform_shuffle=False, repeat_count=1, batch_size=256): 
    
    # reads in single training example and returns it in a format that the estimator can
    # use
    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                            "label": tf.io.FixedLenFeature([], tf.float32),
                            'rows': tf.io.FixedLenFeature([], tf.int64),
                            'cols': tf.io.FixedLenFeature([], tf.int64),
                            'depth': tf.io.FixedLenFeature([], tf.int64)}

        # Load one example
        parsed_example = tf.io.parse_single_example(proto, keys_to_features)

        image_shape = image_shape = tf.stack([640 , 360, 3])
        image_raw = parsed_example['image']

        label = tf.cast(parsed_example['label'], tf.float32)
        image = tf.io.decode_raw(image_raw, tf.uint8)
        image = tf.cast(image, tf.float32)

        image = tf.reshape(image, image_shape)

        return {'image':image},label
    
    dataset = tf.data.TFRecordDataset(filenames = filepath)
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    
    return batch_features, batch_labels

# build model

def build_model(model_dir, checkpoint_path):
    
    # construct keras model
    Input = tf.keras.layers.Input(shape=(640, 360, 3,), name='image')
    x = Conv2D(24, kernel_size=(5, 5), activation='swish', strides=(2, 2))(Input)
    x = BatchNormalization()(x)
    x = Conv2D(36, kernel_size=(5, 5), activation='swish', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(36, kernel_size=(3, 3), activation='swish', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, kernel_size=(3, 3), activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='swish')(x)
    x = Dense(128)(x)
    x = Dropout(0.8)(x)
    x = Dense(96)(x)
    x = Dropout(0.8)(x)
    x = Dense(64)(x)
    x = Dropout(0.8)(x)
    x = Dense(32)(x)
    x = Dropout(0.8)(x)
    x = Dense(16)(x)

    keras_model = tf.keras.models.Model(
          inputs = [Input], outputs = [x])

    # update model with weights
    keras_model.load_weights(tf.train.latest_checkpoint(checkpoint_path))

    keras_model.compile(
        loss = 'mae',
        optimizer = keras.optimizers.Adadelta(learning_rate = 0.0005, rho = 0.95, epsilon = 1e-07, name = 'Adadelta'),
        metrics=['mse', 'mae'])

    B_zeta_estimator = tf.keras.estimator.model_to_estimator(
          keras_model = keras_model, model_dir = model_dir)
    
    return(B_zeta_estimator)