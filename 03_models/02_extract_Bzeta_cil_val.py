######### Obtain Bzeta for Imprecise Learner #######
# Author: Clara Hoffmann
# Last changed: 12.01.2021


# load packages
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import imageio
from utils import find_closest_element, Fy, build_model

# path of model checkpoints
checkpoint_path = '../../../../data/models/20201021_unrestr_gaussian_resampled/'
shard_path = '../../../../data/commaai/training_files/unrestricted_gauss_dens_resampled'
extracted_coefficients_directory_beta = '../../../../data/commaai/extracted_coefficients/20201021_unrestr_gaussian_resampled/beta/'
extracted_coefficients_directory_Bzeta = '../../../../data/commaai/extracted_coefficients/20201021_unrestr_gaussian_resampled/Bzeta/'

# build model and # load weights
B_zeta_model = build_model_bzeta()
B_zeta_model.load_weights(tf.train.latest_checkpoint(checkpoint_path))

# all training images and paths
path_all_imgs = '../../../../commaai_code/01_sort_into_bags/02_b_cil_shards/val_shards/df_paths.csv'
all_img_df = pd.read_csv(path_all_imgs)
img_path_base = '../../../../data/commaai/test_files/val_files_unfiltered/'
density_path= '../../../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)
all_img_df = all_img_df[np.abs(all_img_df['true_y']) < 40].reset_index()

labels = []
tr_labels = []
B_zetas = []
y_preds = []
for i in tqdm(range(0,all_img_df.shape[0])): 
    img = imageio.imread(str(img_path_base + all_img_df.loc[i,'path']))/255
    B_zeta = B_zeta_model.predict(img.reshape(1, 66, 200, 3))
    label = all_img_df.loc[i,'true_y']
    tr_label = norm.ppf(Fy(label, density))
    labels.append(label)
    tr_labels.append(tr_label)          
    B_zetas.append(B_zeta)
    
labels = np.array(labels)
B_zetas = np.array(B_zetas).reshape(95838, 10)
tr_labels = np.array(tr_labels)

#np.save(str(extracted_coefficients_directory_Bzeta + 'labels_val.npy'), labels)
#np.save(str(extracted_coefficients_directory_Bzeta + 'B_zeta_val.npy'), B_zetas)
#np.save(str(extracted_coefficients_directory_Bzeta + 'tr_labels_val.npy'), tr_labels)