######### Obtain Bzeta for Precise Learner #######
# Author: Clara Hoffmann
# Last changed: 12.01.2021

# load packages
import os
import glob
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tqdm import tqdm
from scipy.stats import norm
import imageio
import cv2
from utils import find_closest_element, Fy, build_model_bzeta, build_model

# path of model checkpoints
checkpoint_path = '../../data/models/copula_cpl/export/'
shard_path = '../../data/commaai/training_files_filtered/tfrecords/'
shard_files = glob.glob(os.path.join(shard_path, "*.tfrecords")) 
extracted_coefficients_directory_beta = '../../data/commaai/extracted_coefficients/copula_cpl/beta/'
extracted_coefficients_directory_Bzeta = '../../data/commaai/extracted_coefficients/copula_cpl/Bzeta/'

####### 1. Build keras model and load weights #######

# define model and load weights from training
keras_model = build_model()

# load weights
keras_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)) 

# get coefficients of last layer
i = 0
for layer in keras_model.layers: 
    i += 1
    if i == 20:
        beta = layer.get_weights()
        print(layer.get_config()) 
        
# last element of beta is bias, exclude this 
# since copulas are location free -> beta_0 = 0
beta_coeff = beta[0]
# save
np.savetxt(str(extracted_coefficients_directory_beta +"beta.csv"), beta_coeff, delimiter=",")

####### 2. Extract Basis Functions Bzeta #######

# keras model for basis functions B_zeta
B_zeta_model = build_model_bzeta()

# load weights from training
B_zeta_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)) 

# all training images and paths
# switch train_indices to val indices to get validation Bzetas
path_all_imgs = '../../data/commaai/training_files_filtered/indices/val_indices.csv'
all_img_df = pd.read_csv(path_all_imgs)
img_path_base = '../../data/commaai/train_bags_2/'
density_path = '../../data/commaai/density/gaussian_density_filtered.csv'
density = pd.read_csv(density_path)

# extract Bzetas by reading in images and predicting
labels = []
tr_labels = []
B_zetas = []
for i in tqdm(range(0,all_img_df.shape[0])): 
    img = imageio.imread(str(img_path_base + all_img_df.loc[i,'path']))[:,:,0:3]/255
    B_zeta = B_zeta_model.predict(img.reshape(1,66,200,3))
    label = all_img_df.loc[i,'angle']
    tr_label = norm.ppf(Fy(label, density))
    labels.append(label)
    tr_labels.append(tr_label)          
    B_zetas.append(B_zeta)

# rearrange to arrays
labels = np.array(labels)
B_zetas = np.array(B_zetas)
tr_labels = np.array(tr_labels)

# save
np.save(str(extracted_coefficients_directory_Bzeta + 'labels_val.npy'), labels)
np.save(str(extracted_coefficients_directory_Bzeta + 'B_zeta_val.npy'), B_zetas)
np.save(str(extracted_coefficients_directory_Bzeta + 'tr_labels_val.npy'), tr_labels)