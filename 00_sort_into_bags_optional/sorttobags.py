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
from helpers import find_closest_element, read_vid_angles


vid_train_path = r'../../data/commaai/destination/'
filepath = r'../../data/commaai/destination/'

train_set_size = 1500
test_set_size = 500

# read names of all video files
all_vid_files = glob.glob(os.path.join(vid_train_path, "*.hevc")) 

# choose files for train and test set
test_files_vid = all_vid_files[-test_set_size:]
train_files_vid = all_vid_files[:train_set_size]

# get filenames without ending
train_filenames = [os.path.basename(train_files_vid[i])[:-5] for i in range(0, len(train_files_vid))]
test_filenames = [os.path.basename(test_files_vid[i])[:-5] for i in range(0, len(test_files_vid))]

train_filenames = np.load('train_files_run2.npy', allow_pickle = True)

train_vid_files = [str(filepath + train_filenames[i] + '.hevc') for i in range(len(train_filenames))]
train_yaw_files = [str(filepath + train_filenames[i] + '.value') for i in range(len(train_filenames))]
train_time_files = [str(filepath + train_filenames[i] + '.t') for i in range(len(train_filenames))]

#np.save('train_files_run2.npy', np.array(train_filenames))
#np.save('test_files_run2.npy', np.array(test_filenames))


for j in range(340, 1500): #len(train_vid_files)
    # get single file
    video_file = train_vid_files[j]
    angle_file = train_yaw_files[j]
    time_file = train_time_files[j]
    
    images, yaw, trans_label = read_vid_angles(video_file, angle_file, time_file)
    
    for i in range(0, images.shape[0]):
        img = images[i,:,:,:]
        label = yaw[i]
        tr_label = trans_label[i]
        
        print(str(i), str(j))
        if abs(label) <= 5 :
            path = '../../data/commaai/train_bags_small/0/'
            filename = str(path + str(i) + '_' + str(j) + 'run1.png')
            #np.save(filename, np.array([img, label, tr_label]))
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
