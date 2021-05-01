import keras
import glob 
import os
import tensorflow as tf
import random
from utils import imgs_input_fn, imgs_input_fn_val, rmse, build_model
import mdn
import pandas as pd
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import imageio
import numpy as np
from mdn import sample_from_output
import matplotlib.pyplot as plt
from utils import build_model, find_closest_element, Fy
from scipy.stats import norm

#define paths
# directory where checkpoints should be saved
model_dir = '../../../../data/models/mdn_cil/export/'
checkpoint_path = '../../../../data/models/mdn_cil/export/'

# define model and load weights from training
keras_model = build_model()

# load weights
keras_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)) 

# read in val data
path_all_imgs = '../../../../commaai_code/02_write_shards_optional/cil_shards/df_paths.csv'
all_img_df = pd.read_csv(path_all_imgs)
all_img_df = all_img_df[np.abs(all_img_df['true_y']) < 40].reset_index()
img_path_base = '../../../data/commaai/train_bags_2/'
true_y = all_img_df['true_y']

density_path = '../../../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)

no_samp = 1000

# preds Bzetas by reading in images and predicting
preds = []
samples = []
labels  = []

#labels = list(np.load('../../../../data/commaai/predictions/mdn/cil/labels.npy'))
#preds = list(np.load('../../../../data/commaai/predictions/mdn/cil/preds.npy'))
#samples = list(np.load('../../../../data/commaai/predictions/mdn/cil/samples.npy'))
preds = []  
    
for i in tqdm(range(len(preds),all_img_df.shape[0])): 
    img = imageio.imread(str(img_path_base + all_img_df.loc[i,'path']))[:,:,0:3]/255
    pred = keras_model.predict(img.reshape(1,66,200,3))   
    #y_samples = np.array([np.apply_along_axis(sample_from_output, 1, pred, 1, 50, temp=1.0) for i in range(0,no_samp)])
    #samples.append(y_samples)
    #label = all_img_df.loc[i,'true_y']
    #labels.append(label) 
    preds.append(pred)
    if i % 1000:
        #np.save('../../../../data/commaai/predictions/mdn/cil/labels.npy', np.array(labels))
        np.save('../../../../data/commaai/predictions/mdn/cil/preds.npy', np.array(preds))
        #np.save('../../../../data/commaai/predictions/mdn/cil/samples.npy', np.array(samples))
    
#np.save('../../../../data/commaai/predictions/mdn/cil/labels.npy', np.array(labels))
np.save('../../../../data/commaai/predictions/mdn/cil/preds.npy', np.array(preds))
#np.save('../../../../data/commaai/predictions/mdn/cil/samples.npy', np.array(samples))

#samples = np.array(samples)
#pred_mdn = np.mean(samples.reshape(-1, 1000), axis = 1)
#np.save('../../../../data/commaai/predictions/mdn/cpl/mdn_preds.npy', np.array(pred_mdn))
