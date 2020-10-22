import glob
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
import imageio
import cv2
from tqdm import tqdm

destination = '../../data/commaai/trash/'
bag_path = '../../data/commaai/train_bags_2/'
density_path = '../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)

max_bins = [10000, 15000, 35000, 40000, 10000, 2000, 500, 500]
counts = np.zeros(8)

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def find_closest_element(y: float, arr: np.ndarray):
    index = np.searchsorted(arr,y)
    if (index >= 1) & (index < arr.shape[0]):
        res = [arr[index - 1], arr[index]]
    elif (index < arr.shape[0]):
        return np.array(index)
    else:
        return np.array(index - 1)

    if res[0] == res[1]:
        return np.array(index - 1)
    else:
        diff_pre = np.abs(y-res[0])
        diff_aft = np.abs(y-res[1])
        if diff_pre == diff_aft:
            return np.array(index - 1) 
        else:
            return index - 1 if diff_pre < diff_aft else index
def Fy(y, density, density_type = 'fast_kde' ):
    integral = density.loc[find_closest_element(y, density['axes']),'cdf']
    return(integral)  

path0 = '../../data/commaai/train_bags_2/0/angles_filename.csv'
path1 = '../../data/commaai/train_bags_2/1/angles_filename.csv'
path2 = '../../data/commaai/train_bags_2/2/angles_filename.csv'
path3 = '../../data/commaai/train_bags_2/3/angles_filename.csv'
path4 = '../../data/commaai/train_bags_2/4/angles_filename.csv'
path5 = '../../data/commaai/train_bags_2/5/angles_filename.csv'
path6 = '../../data/commaai/train_bags_2/6/angles_filename.csv'

df0 = pd.read_csv(path0, header = None)
df0.columns = ['filename', 'angle', 'tr_angle']
tqdm.pandas()
df0['tr_angle'] = df0['angle'].progress_apply(lambda x: norm.ppf(Fy(x, density)))

df0a = df0.loc[abs(df0['angle']) < 1,:]
df0a = df0a.reset_index()
df0b = df0.loc[((abs(df0['angle']) < 2) & (abs(df0['angle']) >= 1)) ,:]
df0b = df0b.reset_index()
df0c = df0.loc[abs(df0['angle']) >= 2,:]
df0c = df0c.reset_index()

print(str(df0a.shape),str(df0b.shape), str(df0c.shape) )
df1 = pd.read_csv(path1, header = None)
df1.columns = ['filename', 'angle', 'tr_angle']
df1['tr_angle'] = df1['angle'].progress_apply(lambda x: norm.ppf(Fy(x, density)))

df2 = pd.read_csv(path2, header = None)
df2.columns = ['filename', 'angle', 'tr_angle']
df2['tr_angle'] = df2['angle'].progress_apply(lambda x: norm.ppf(Fy(x, density)))

df3 = pd.read_csv(path3, header = None)
df3.columns = ['filename', 'angle', 'tr_angle']
df3['tr_angle'] = df3['angle'].progress_apply(lambda x: norm.ppf(Fy(x, density)))

df4 = pd.read_csv(path4, header = None)
df4.columns = ['filename', 'angle', 'tr_angle']
df4['tr_angle'] = df4['angle'].progress_apply(lambda x: norm.ppf(Fy(x, density)))

df5 = pd.read_csv(path5, header = None)
df5.columns = ['filename', 'angle', 'tr_angle']
df5['tr_angle'] = df5['angle'].progress_apply(lambda x: norm.ppf(Fy(x, density)))

# new size for resizing
new_height = 291
new_width = 218

counts_minus_1 = np.zeros(8)
n = 0
for m in range(0,int(40000/12)):

    out_path_shard = str(destination + str(m) + '.tfrecords')

    print(out_path_shard)

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
                    current_file = random.choice(df['filename'])
                    current_file_ext = str('../../data/' + current_file)
                    print(current_file, i)
                    try: 
                        #x = np.load(current_file, allow_pickle=True)
                        #img = x[0]
                        #label = x[1]
                        #tr_label = x[2]

                        img = imageio.imread(current_file_ext)
                        # resize image and crop
                        #img = cv2.resize(img, dsize = (new_height,new_width), interpolation = cv2.INTER_LINEAR)[76:142, 45:245,:]
                        row = df.loc[df['filename'] == current_file, ['angle', 'tr_angle']]
                        label = float(row['angle'])
                        tr_label =  float(row['tr_angle'])
                        # write to shard
                        print(img.shape)
                        img_bytes = img.tostring()
                        print(len(img_bytes))
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


                        with open(str(destination + 'yaws.csv'), 'a') as csvfile:
                            csvfile.write("%s\n" % label)
                        with open(str(destination + 'yaws_transformed.csv'), 'a') as csvfile:
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


                            with open(str(destination + 'yaws.csv'), 'a') as csvfile:
                                csvfile.write("%s\n" % label)

                            # Write the serialized data to the TFRecords file.
                            writer.write(serialized)

                    except:
                        print('couldnt open file')
