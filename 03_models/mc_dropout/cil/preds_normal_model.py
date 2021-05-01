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
from utils import build_model, find_closest_element, Fy
from scipy.stats import norm

checkpoint_path = '../../../../data/models/mc_dropout_cil/export/'

model = build_model()
model.load_weights(tf.train.latest_checkpoint(checkpoint_path))

# all training images and paths
path_all_imgs = '../../../../commaai_code/02_write_shards_optional/cil_shards/df_paths.csv'
all_img_df = pd.read_csv(path_all_imgs)
all_img_df = all_img_df[np.abs(all_img_df['true_y']) < 40].reset_index()

# extract Bzetas by reading in images and predicting
labels = []
preds = []
for i in tqdm(range(0,all_img_df.shape[0])): 
    img = imageio.imread(str(all_img_df.loc[i,'path']))[:,:,0:3]/255
    pred = model.predict(img.reshape(1,66,200,3))
    label = all_img_df.loc[i,'true_y']
    labels.append(label)       
    preds.append(pred)

# rearrange to arrays
labels = np.array(labels)
preds = np.array(preds)
np.save('../../../../data/commaai/predictions/y_preds_cil_no_mc.npy', preds)
np.save('../../../../data/commaai/predictions/y_labels_cil_no_mc.npy', labels)