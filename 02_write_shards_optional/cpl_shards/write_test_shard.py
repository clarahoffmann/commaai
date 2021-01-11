# Write test shard for cpl
import pandas as pd
import matplotlib.pyplot as plt
from helpers import wrap_int64, wrap_bytes, wrap_float, Fy
from tqdm import tqdm
from scipy.stats import norm
import numpy as np
import tensorflow as tf
import imageio
import cv2
import random
from helpers import *

print('reading files')
# dir where test shard goes
destination = '../../../data/commaai/test_files/val_file_filtered/'
# bags with training images
bag_path = '../../../data/commaai/train_bags_2/'

# read in validation indices
df = pd.read_csv('../../../data/commaai/training_files_filtered/indices/val_indices.csv')
df = df.loc[df['use'] == 1, :]

# read in density
density_path = '../../../data/commaai/density/gaussian_density_filtered.csv'
density = pd.read_csv(density_path)

# transform angles y to z = \Phi^(-1)(F(y))
print('transforming to z')
tqdm.pandas()
df['tr_angle'] = df['angle'].progress_apply(lambda x: norm.ppf(Fy(x, density)))

print('start writing shards')

out_path_shard = str(destination + 'val.tfrecords')

# write to tfrecords file
with tf.io.TFRecordWriter(out_path_shard) as writer:
    
    # loop over angles and their paths
    for i in tqdm(range(0, df['angle'].shape[0])):
        
        # load image file for angle
        current_file = df.loc[i, 'path']
        current_file_ext = str( bag_path + current_file)
        img = imageio.imread(current_file_ext)
        # resize image and crop
        img = cv2.resize(img, dsize = (291,218), interpolation = cv2.INTER_LINEAR)[76:142, 45:245,0:3]
        
        # get y and z
        label = float(df.loc[i, 'angle'])
        tr_label = float(df.loc[i, 'tr_angle'])
        
        img_bytes = img.tostring()
        rows = img.shape[0]
        cols = img.shape[1]
        depth = img.shape[2]
        
        # save image and label in dictionary
        data = \
                {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_float(label),
                    'rows': wrap_int64(rows),
                    'cols': wrap_int64(cols),
                    'depth': wrap_int64(depth),
                    'tr_label': wrap_float(tr_label)

                }
        feature = tf.train.Features(feature=data)
        
        # wrap as tf example
        example = tf.train.Example(features=feature)

        serialized = example.SerializeToString()
        
        # write to tfrecords file
        writer.write(serialized)

        # save angles in additional file
        with open(str(destination + 'angle.csv'), 'a') as csvfile:
            csvfile.write("%s\n" % label)
        with open(str(destination + 'angle_transformed.csv'), 'a') as csvfile:
            csvfile.write("%s\n" % tr_label)