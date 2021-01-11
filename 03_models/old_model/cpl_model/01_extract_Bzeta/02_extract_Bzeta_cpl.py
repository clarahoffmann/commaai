# load packages
import os
import glob
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import backend
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm
import imageio
import cv2
import multiprocessing
from utils import find_closest_element, Fy

# path of model checkpoints
checkpoint_path = '../../../data/models/20201027_filtered_gaussian_resampled/export/'
shard_path = '../../../data/commaai/training_files_filtered/tfrecords/'
shard_files = glob.glob(os.path.join(shard_path, "*.tfrecords")) 
extracted_coefficients_directory_beta = '../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/beta/'
extracted_coefficients_directory_Bzeta = '../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/Bzeta/'

# define model and load weights from training
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

# load weights
keras_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)) # tf.train.latest_checkpoint(checkpoint_path)

# get coefficients of last layer
i = 0
for layer in keras_model.layers: 
    i += 1
    if i == 20:
        beta = layer.get_weights()
        print(layer.get_config()) #, layer.get_weights()
        
# last element of beta is bias, exclude this 
# since copulas are location free -> beta_0 = 0
beta_coeff = beta[0]
# save
np.savetxt(str(extracted_coefficients_directory_beta +"beta.csv"), beta_coeff, delimiter=",")

# keras model for basis functions B_zeta
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

B_zeta_model = tf.keras.models.Model(
      inputs = [Input], outputs = [x])

# load weights from training
B_zeta_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)) # tf.train.latest_checkpoint(checkpoint_path)

# all training images and paths
path_all_imgs = '../../../data/commaai/training_files_filtered/indices/train_indices.csv'
all_img_df = pd.read_csv(path_all_imgs)
img_path_base = '../../../data/commaai/train_bags_2/'
density_path = '../../../data/commaai/density/gaussian_density_filtered.csv'
density = pd.read_csv(density_path)

labels = []
tr_labels = []
B_zetas = []
for i in tqdm(range(0,all_img_df.shape[0])): 
    img = imageio.imread(str(img_path_base + all_img_df.loc[i,'path']))
    img = cv2.resize(img, dsize = (291,218), interpolation = cv2.INTER_LINEAR)[76:142, 45:245,0:3].reshape(1,66,200,3)/255
    B_zeta = B_zeta_model.predict(img)
    label = all_img_df.loc[i,'angle']
    tr_label = norm.ppf(Fy(label, density))
    labels.append(label)
    tr_labels.append(tr_label)          
    B_zetas.append(B_zeta)
    
labels = np.array(labels)
B_zetas = np.array(B_zetas)
tr_labels = np.array(tr_labels)

np.save(str(extracted_coefficients_directory_Bzeta + 'labels.csv'), labels)
np.save(str(extracted_coefficients_directory_Bzeta + 'B_zeta.csv'), B_zetas)
np.save(str(extracted_coefficients_directory_Bzeta + 'tr_labels.csv'), tr_labels)