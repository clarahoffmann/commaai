import glob
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
import imageio

destination = '../../data/commaai/train_files_from_bag_test/'
bag_path = '../../data/commaai/train_bags/'
density_path = '../../data/commaai/density/densityfastkde_density.csv'
density = pd.read_csv(density_path)

all_files = [glob.glob(os.path.join(str(bag_path + str(i) + '/'), "*.npy")) for i in range(1,7)]

max_bins = [10000, 10000, 2500, 1000, 1000, 1000]
counts = np.zeros(6)

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
            return np.array(index - 1), 
        else:
            return index - 1 if diff_pre < diff_aft else index
def Fy(y, density, density_type = 'fast_kde' ):
    integral = density.loc[find_closest_element(y, density['axes']),'cdf']
    return(integral)  

path0 = '../../data/commaai/train_bags/0/angles_filename.csv'
path1 = '../../data/commaai/train_bags/1/angles_filename.csv'
path2 = '../../data/commaai/train_bags/2/angles_filename.csv'
path3 = '../../data/commaai/train_bags/3/angles_filename.csv'
path4 = '../../data/commaai/train_bags/4/angles_filename.csv'
path5 = '../../data/commaai/train_bags/5/angles_filename.csv'
path6 = '../../data/commaai/train_bags/6/angles_filename.csv'
df0 = pd.read_csv(path0, header = None)
df0.columns = ['filename', 'angle', 'tr_angle']
df1 = pd.read_csv(path1, header = None)
df1.columns = ['filename', 'angle', 'tr_angle']
df2 = pd.read_csv(path2, header = None)
df2.columns = ['filename', 'angle', 'tr_angle']
df3 = pd.read_csv(path3, header = None)
df3.columns = ['filename', 'angle', 'tr_angle']
df4 = pd.read_csv(path4, header = None)
df4.columns = ['filename', 'angle', 'tr_angle']
df5 = pd.read_csv(path5, header = None)
df5.columns = ['filename', 'angle', 'tr_angle']

df_list_expect_zero = df1.append(df2).append(df3).append(df4).append(df5)

counts_minus_1 = np.zeros(6)
n = 0
for m in range(0,int(10000/12)):

    out_path_shard = str(destination + str(m) + '_new2.tfrecords')

    print(out_path_shard)

    with tf.io.TFRecordWriter(out_path_shard) as writer:
        
        # write 120 images to each shard
        for j in range(0, 12):
            
             for i in range(0,len(all_files)):
                
                if i == 0:
                    df = df0
                else:
                    df = df_list_expect_zero

                if all_files[i]: 

                    # draw new sample if bin is not full yet
                    if counts[i] <= max_bins[i]:

                        # load random example from bag i
                        current_file = str(random.choice(all_files[i]))
                        print(current_file, len(all_files[i]))
                        try: 
                            #x = np.load(current_file, allow_pickle=True)
                            #img = x[0]
                            #label = x[1]
                            #tr_label = x[2]
                            
                            img = imageio.imread(current_file)
                            label, tr_angle = df.loc[df['filename'] == current_file, ['angle', 'tr_angle']]
                            
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

                                counts[i] += 1
                        except:
                            print('couldnt open file')
