import tensorflow as tf
import numpy as np
#from utils import build_model
import pandas as pd
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import keras
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras.regularizers import l2

def build_model():
    '''
    Build PilotNet keras model
    
    Input:
        None
    
    Output:
        - keras_model: keras model with PilotNet architecture
    '''
    
    Lambda = 10**(-6)
    # build model
    Input = tf.keras.layers.Input(shape=(66, 200, 3,), name='image')
    x = Conv2D(24, kernel_size=(5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(Input)
    x = Dropout(.05)(x, training = True)
    #x = BatchNormalization()(x)
    x = Conv2D(36, kernel_size=(5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(x)
    x = Dropout(.05)(x, training = True)
    #x = BatchNormalization()(x)
    x = Conv2D(48, kernel_size=(5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(x)
    x = Dropout(.05)(x, training = True)
    #x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(x)
    #x = BatchNormalization()(x)
    x = Dropout(.05)(x, training = True)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(x)
    x = Flatten()(x, training = True)
    x = Dropout(0.05)(x, training = True)
    x = Dense(1164, kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda), activation='relu')(x)
    x = Dropout(0.05)(x, training = True)
    x = Dense(100, kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda), activation='relu')(x)
    x = Dropout(0.05)(x, training = True)
    x = Dense(50, kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda), activation='relu')(x) 
    x = Dropout(0.05)(x, training = True)
    x = Dense(10, activation='relu')(x)
    Output = Dense(1, name = 'output_layer')(x)

    # compile
    keras_model = tf.keras.models.Model(
          inputs = [Input], outputs = [Output])
    
    return(keras_model)

checkpoint_path = '../../../data/models/mc_dropout_cpl/export/'
model = build_model()
model.load_weights(tf.train.latest_checkpoint(checkpoint_path))

path_all_imgs = '../../../data/commaai/training_files_filtered/indices/val_indices.csv'
all_img_df = pd.read_csv(path_all_imgs)
img_path_base = '../../../data/commaai/train_bags_2/'
density_path = '../../../data/commaai/density/gaussian_density_filtered.csv'
density = pd.read_csv(density_path)

mc_samples = 1000
preds = []
for i in tqdm(range(int(all_img_df.shape[0]/2), all_img_df.shape[0])):
    img = (imageio.imread(str(img_path_base + all_img_df.loc[i,'path']))[:,:,0:3]/255).reshape(1,66,200,3)
    x_pred = np.repeat(img, mc_samples, axis = 0)
    pred = model.predict(x_pred.reshape(-1,66,200,3))
    preds.append(pred)
preds = np.array(preds)

np.save('../../../data/commaai/predictions/mc_preds_1.npy', preds)
