import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import glob
import os
import math
from PIL import Image
import tensorflow as tf
import sys
from scipy.integrate import simps
from helpers import *
import random

# Specify paths where data lies
vid_train_path = r'../data/commaai/destination/'
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

# set path to where train files should be exported
out_path_base_train = '../data/commaai/train/trainshard'
density_path = '../data/commaai/density/densityfastkde_density.csv'

# convert train files
convert(train_filenames, vid_train_path, density_path,  out_path_base_train, verbose = False, sampling = 'downsample')