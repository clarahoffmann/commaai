######### Videos to Frames + Sorting into Bags Based on Angles #######
# Author: Clara Hoffmann
# Last changed: 12.01.2021
# This code serves to convert the videos from the comma.ai 2k19 data set to
# frames, which are then used to train the end-to-end learners.
# Since mostly straight trajectories are features, the training data has to be
# pruned (limit the number of examples with straight trajectories) and
# oversampled (increase the number of examples with curves).
# To make pruning and oversampling easy, the training images are sorted into
# different folders based on their steering angle. For creating the training set
# we can now just simply sample a prespecified number of examples from each folder.

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import norm
import imageio
import pandas as pd
import png
import csv
from tqdm import tqdm
from helpers import find_closest_element, read_vid_angles

# path of video files
vid_train_path = r'../../data/commaai/destination/'
filepath = r'../../data/commaai/destination/'

# how many videos to use for training/validation
train_set_size = 1500
test_set_size = 500

# read names of all video files
all_vid_files = glob.glob(os.path.join(vid_train_path, "*.hevc")) 

# if you want to have a new train/validation split, uncomment
# this code and comment the first line after
#test_files_vid = all_vid_files[-test_set_size:]
#train_files_vid = all_vid_files[:train_set_size]
# get filenames without ending
#train_filenames = [os.path.basename(train_files_vid[i])[:-5] for i in range(0, len(train_files_vid))]
#test_filenames = [os.path.basename(test_files_vid[i])[:-5] for i in range(0, len(test_files_vid))]

# have chosen this before, so just use the ones specified in this file
train_filenames = np.load('train_files_run.npy', allow_pickle = True)

train_vid_files = [str(filepath + train_filenames[i] + '.hevc') for i in range(len(train_filenames))]
train_yaw_files = [str(filepath + train_filenames[i] + '.value') for i in range(len(train_filenames))]
train_time_files = [str(filepath + train_filenames[i] + '.t') for i in range(len(train_filenames))]

for j in tqdm(range(917, 1001)): 
    
    # get single video file
    video_file = train_vid_files[j]
    angle_file = train_yaw_files[j]
    time_file = train_time_files[j]
    
    # get frames and associated angles from file
    images, yaw, trans_label = read_vid_angles(video_file, angle_file, time_file)
    
    # sort each image into folder based on steering angle
    for i in range(0, images.shape[0]):
        img = images[i,:,:,:]
        label = yaw[i]
        tr_label = trans_label[i]
        
        if abs(label) <= 5 :
            path = '../../data/commaai/train_bags_small/0/'
            filename = str(path + str(i) + '_' + str(j) + 'run1.png')
            with open(str(path + 'angles_filename.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename, label, tr_label])
            plt.imsave(filename, img)
        
        if abs(label) <= 10 and abs(label) > 5:
            path = '../../data/commaai/train_bags_small/1/'
            filename = str(path + str(i) + '_' + str(j) + 'run1.png')
            with open(str(path + 'angles_filename.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename, label, tr_label])
            plt.imsave(filename, img)
        
        elif abs(label) <= 20 and abs(label) > 10:
            path = '../../data/commaai/train_bags_small/2/'
            filename = str(path + str(i) + '_' + str(j) + 'run1.png')
            with open(str(path + 'angles_filename.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename, label, tr_label])
            plt.imsave(filename, img)
        
        elif abs(label) <= 30 and abs(label) > 20:
            path = '../../data/commaai/train_bags_small/3/'
            filename = str(path + str(i) + '_' + str(j) + 'run1.png')
            with open(str(path + 'angles_filename.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename, label, tr_label])
            plt.imsave(filename, img)
        
        elif abs(label) <= 40 and abs(label) > 30:
            path = '../../data/commaai/train_bags_small/4/'
            filename = str(path + str(i) + '_' + str(j) + 'run1.png')
            with open(str(path + 'angles_filename.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename, label, tr_label])
            plt.imsave(filename, img)
        
        elif abs(label) <= 50 and abs(label) > 40:
            path = '../../data/commaai/train_bags_small/5/'
            filename = str(path + str(i) + '_' + str(j) + 'run1.png')
            with open(str(path + 'angles_filename.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename, label, tr_label])
            plt.imsave(filename, img)
        
        elif abs(label) <= 60 and abs(label) > 50:
            path = '../../data/commaai/train_bags_small/6/'
            filename = str(path + str(i) + '_' + str(j) + 'run1.png')
            with open(str(path + 'angles_filename.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename, label, tr_label])
            plt.imsave(filename, img)
