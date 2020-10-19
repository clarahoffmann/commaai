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
#train_filenames = [os.path.basename(train_files_vid[i])[:-5] for i in range(0, len(train_files_vid))]
#test_filenames = [os.path.basename(test_files_vid[i])[:-5] for i in range(0, len(test_files_vid))]

test_filenames = np.load('train_files_run2.npy', allow_pickle = True)

test_vid_files = [str(filepath + test_filenames[i] + '.hevc') for i in range(len(test_filenames))]
test_yaw_files = [str(filepath + test_filenames[i] + '.value') for i in range(len(test_filenames))]
test_time_files = [str(filepath + test_filenames[i] + '.t') for i in range(len(test_filenames))]

density_path = '../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)
