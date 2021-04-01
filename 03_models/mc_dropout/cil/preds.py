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
import statsmodels.api as sm
from utils import build_model

checkpoint_path = '../../../../data/models/mc_dropout_cil/export/'

model = build_model()
model.load_weights(tf.train.latest_checkpoint(checkpoint_path))

# all training images and paths
path_all_imgs = '../../../../commaai_code/02_write_shards_optional/cil_shards/df_paths.csv'
all_img_df = pd.read_csv(path_all_imgs)
img_path_base = '../../../../data/commaai/test_files/val_files_unfiltered/'
density_path= '../../../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)
all_img_df = all_img_df[np.abs(all_img_df['true_y']) < 40].reset_index()

# extract Bzetas by reading in images and predicting
labels = []
preds = []
for i in tqdm(range(0,all_img_df.shape[0])): 
    img = imageio.imread(str(img_path_base + all_img_df.loc[i,'path']))[:,:,0:3]/255
    pred = model.predict(img.reshape(1,66,200,3))
    label = all_img_df.loc[i,'true_y']
    labels.append(label)       
    preds.append(pred)

# rearrange to arrays
labels = np.array(labels)
preds = np.array(preds)

np.save('preds_cil.npy', preds)
np.save('labels_cil.npy', labels)