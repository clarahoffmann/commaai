######### Write Validation Shards for Imprecise Learner #######
# Author: Clara Hoffmann
# Last changed: 12.01.2021

# load packages
import glob
import os
import numpy as np
from scipy.stats import norm
import imageio
import pandas as pd
import png
import csv
from helpers import *

# get destination path of test files
# and path of video files
out_path_base = '../../data/commaai/test_files/'
filepath = r'../../data/commaai/destination/'

# get names of test video files
test_filenames = np.load('test_files_run2.npy', allow_pickle = True)

# read in density
density_path = '../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)

# start writing validation shards
print('start writing tfrecords')
convert_val(test_filenames, filepath,  out_path_base, density_path)


