import imageio
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import json
import pandas as pd
import numpy as np
import os
import math
import csv  
import tensorflow as tf
import sys
from scipy.integrate import simps
from scipy.stats import norm
import random
from tqdm import tqdm

def find_closest_element(y: float, arr: np.ndarray):
    '''
    Find index of element closest to y from array arr
    
    Input:
        - y: value to which we want to find the closest value
        - arr: array in which we want to find the closest value
    
    Output:
        - index: index where element is closest to y in arr
    '''
        
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
        
def Fy(y, density):
    '''
    Distribution function at y of numerical cdf
    Input:
        - y: scalar, value at which cdf should be evaluated
        - density: dataframe with columns: axes, cdf
    
    Output: 
        - integral: cdf at y
    '''
    integral = density.loc[find_closest_element(y, density['axes']),'cdf']
    return(integral)   
            
def read_vid_angles(vid_path, value_path, t_path, density):
    '''
    Read frames from comma.ai 2k19 video files and
    merge with associated angles
    
    Input: 
        - vid_path: path of video file
        - value_path: associated angle file
        - t_path: device boot time file
        - density: density dataframe with pdf and cdf
    
    Output:
        - rez_frames: resized and downsampled frames
                      on 5 frames per second bases
        - target_angles: associated steering angles y
        - trans_label: associated transformed steering angles z
    '''
    
    # read video
    vid = imageio.get_reader(vid_path, 'ffmpeg')
    frames = np.array([im for im in vid.iter_data()], dtype=np.uint8)
    vid.close()
    
    # read steering angles
    angle = np.load(value_path)
    
    # read device boot time
    t = np.load(t_path)
    
    # dataframe of angles and timestamps
    angles = pd.DataFrame({'t' : t, 'angle': angle})
    
    # get timestamps of frames
    timestamps_frames = np.zeros(frames.shape[0])
    start_stamp = t[0] 
    timestamps_frames[0] = start_stamp
    for i in range(1, len(timestamps_frames)):
        timestamps_frames[i] = timestamps_frames[i - 1] + 0.05
    
    # get angles per frame
    target_angles = [angles.loc[find_closest_element(timestamps_frames[i], np.array(angles['t'])),'angle'] for i in range(0, len(timestamps_frames))]
    
    
    rez_frames = []
    for frame in frames:
        rez_frame = cv2.resize(frame, dsize = (291,218), interpolation = cv2.INTER_LINEAR)[76:142, 45:245,0:3]
        rez_frames.append(rez_frame)
    
    rez_frames = np.array(rez_frames)
    target_angles = target_angles[::5].copy()
    trans_label = [norm.ppf(Fy(target_angles[i], density)) for i in range(0, len(target_angles))]
    
    return(rez_frames[::5], target_angles, trans_label)

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def convert_val(train_filenames, filepath,  out_path_base, density_path):
    '''
    Convert validation examples to tfrecords file
    
    Input:
        - train_filenames: names of all video files for training
        - filepath: path of video files
        - out_path_base: folder where tfrecords should go
        - density_path: density df from training examples
    
    Output: 
        - created tfrecords file in out_path_base
    
    '''   
    density = pd.read_csv(density_path)
    
    # get full paths for yaws
    train_vid_files = [str(filepath + train_filenames[i] + '.hevc') for i in range(len(train_filenames))]
    train_yaw_files = [str(filepath + train_filenames[i] + '.value') for i in range(len(train_filenames))]
    train_time_files = [str(filepath + train_filenames[i] + '.t') for i in range(len(train_filenames))]
    
    out_path_shard = str(out_path_base + 'val.tfrecords')
    with tf.io.TFRecordWriter(out_path_shard) as writer:

        # iterate over each file in file chunks
        for j in tqdm(range(0, len(train_yaw_files))):

            # get single yaw and video file path
            yaw_file = train_yaw_files[j]
            vid_file = train_vid_files[j]
            time_file = train_time_files[j]

            if os.path.basename(yaw_file)[:-5] != os.path.basename(vid_file)[:-4]:
                print('WARNING! filenames do not match')
            
            exception = 0
            
            try: 
                images, yaw, tr_label = read_vid_angles(vid_file, yaw_file, time_file, density)
            except:
                exception = 1
                print('error reading file')
            
            if exception == 0:
                # write to tfrecords
                #for i, (img, label, tr_l) in enumerate(zip(images, yaw, tr_label)):
                for i in range(0, len(yaw)):
                    img = images[i]
                    label = yaw[i]
                    tr_l = tr_label[i]

                    if abs(label) <= 60:
                        # save image to string so we convert it to bytes
                        #print(img.shape)
                        img_bytes = img.tostring()
                        #print(len(img_bytes))
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
                                    'tr_label': wrap_float(tr_l)

                                }
                        feature = tf.train.Features(feature=data)

                        # Wrap again as a TensorFlow Example.
                        example = tf.train.Example(features=feature)

                        serialized = example.SerializeToString()


                        with open(str(out_path_base + 'yaws.csv'), 'a') as csvfile:
                            csvfile.write("%s\n" % label)
                       
                        with open(str(out_path_base + 'yaws_transformed.csv'), 'a') as csvfile:
                            csvfile.write("%s\n" % tr_label)

                        # Write the serialized data to the TFRecords file.
                        writer.write(serialized)
                        
    print('finished writing all files to ' + str(out_path_base))