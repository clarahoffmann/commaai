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

print('reading files')
destination = '../../../data/commaai/training_files/filtered_gauss_dens_resampled_new_vers_buckets/'
bag_path = '../../../data/commaai/train_bags_2/'

df = pd.read_csv('../../../data/commaai/training_files_filtered/indices/train_indices.csv')
df = df.loc[df['use'] == 1, :]

density_path = '../../../data/commaai/density/gaussian_density_filtered.csv'
density = pd.read_csv(density_path)

print('transforming to z')
tqdm.pandas()
df['tr_angle'] = df['angle'].progress_apply(lambda x: norm.ppf(Fy(x, density)))

df0a = df.loc[abs(df['angle']) < 1].reset_index()
df0b = df.loc[((abs(df['angle']) < 2) & (abs(df['angle']) >= 1)) ,:].reset_index()
df0c = df.loc[(abs(df['angle']) >= 2) & (abs(df['angle']) < 5)  ].reset_index()
df1 = df.loc[(abs(df['angle']) <= 10) & (abs(df['angle']) > 5)].reset_index()
df2 = df.loc[(abs(df['angle']) <= 20) & (abs(df['angle']) > 10)].reset_index()
df3 = df.loc[(abs(df['angle']) <= 30) & (abs(df['angle']) > 20)].reset_index()
df4 = df.loc[(abs(df['angle']) <= 40) & (abs(df['angle']) > 30)].reset_index()
df5 = df.loc[(abs(df['angle']) <= 50) & (abs(df['angle']) > 40)].reset_index()

max_bins = [8000, 10000, 15000, 12000, 1000, 300, 80, 20]
counts = np.zeros(8)

print('start writing shards')

counts_minus_1 = np.zeros(8)
n = 0
for m in tqdm(range(0,int(12000/12))):
    
    out_path_shard = str(destination + str(m) + '.tfrecords')
    
    with tf.io.TFRecordWriter(out_path_shard) as writer:

        # write 300 images to each shard
        for j in range(0, 20):

             for i in range(0,8):

                if i == 0:
                    df = df0a
                elif i == 1:
                    df = df0b
                elif i == 2:
                    df = df0c
                elif i == 3:
                    df = df1
                elif i == 4:
                    df = df2
                elif i == 5:
                    df = df3
                elif i == 6:
                    df = df4
                elif i == 7:
                    df = df5

                # draw new sample if bin is not full yet
                if counts[i] <= max_bins[i]:

                    # load random example from bag i
                    current_file = random.choice(df['path'])
                    current_file_ext = str(bag_path + current_file)
                    
                    try: 
                        img = imageio.imread(current_file_ext)
                        # resize image and crop
                        img = cv2.resize(img, dsize = (291,218), interpolation = cv2.INTER_LINEAR)[76:142, 45:245,:]
                        row = df.loc[df['path'] == current_file, ['angle', 'tr_angle']]
                        label = float(row['angle'])
                        tr_label =  float(row['tr_angle'])
                        # write to shard
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

                        # Wrap again as a TensorFlow Example.
                        example = tf.train.Example(features=feature)

                        serialized = example.SerializeToString()


                        with open(str(destination + 'angle.csv'), 'a') as csvfile:
                            csvfile.write("%s\n" % label)
                        with open(str(destination + 'angle_transformed.csv'), 'a') as csvfile:
                            csvfile.write("%s\n" % tr_label)

                        # Write the serialized data to the TFRecords file.
                        writer.write(serialized)

                        counts[i] += 1

                        # mirror 50 percent of the images
                        if random.random() > 0.5: 

                            img = np.fliplr(img)
                            label = - label
                            tr_label = norm.ppf(Fy(label, density))

                            # write to shard
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

                            # Wrap again as a TensorFlow Example.
                            example = tf.train.Example(features=feature)

                            serialized = example.SerializeToString()


                            with open(str(destination + 'angle.csv'), 'a') as csvfile:
                                csvfile.write("%s\n" % label)

                            with open(str(destination + 'angle_transformed.csv'), 'a') as csvfile:
                                csvfile.write("%s\n" % label)

                            # Write the serialized data to the TFRecords file.
                            writer.write(serialized)

                    except:
                        print('couldnt open file')