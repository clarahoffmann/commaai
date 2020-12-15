import glob
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm

destination = '../../data/commaai/train_files_from_bag/'
bag_path = '../../data/commaai/train_bags_2/'
density_path = '../../data/commaai/density/fastkde_density.csv'
density = pd.read_csv(density_path)

all_files = [glob.glob(os.path.join(str(bag_path + str(i) + '/'), "*.npy")) for i in range(1,21)]

max_bins = [30000, 30000, 10000, 2000, 500, 500]
counts = np.zeros(10)

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
    
counts_minus_1 = np.zeros(10)
counts = np.zeros(10)
n = 0
    
    #start new shard when more than 256 images have been written
    if (np.sum(counts - counts_minus_1) >= 256):
        n += 1
    counts_minus_1 = counts.copy()
                
    out_path_shard = str(destination + str(n) + '_new2.tfrecords')

    print(out_path_shard)
            # start tfrecords writer

    with tf.io.TFRecordWriter(out_path_shard) as writer:

         for i in range(0,len(all_files)):

            if all_files[i]: 

                # draw new sample if bin is not full yet
                if counts[i] <= max_bins[i]:

                    # load random example from bag i
                    current_file = str(random.choice(all_files[i]))
                    print(current_file, len(all_files[i]))
                    try: 
                        x = np.load(current_file, allow_pickle=True)
                        img = x[0]
                        label = x[1]
                        #tr_label = x[2]
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
