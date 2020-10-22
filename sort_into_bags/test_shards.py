import glob
import os
import numpy as np
import matplotlib.pyplot as plt
#import cv2
from scipy.stats import norm
import imageio
import pandas as pd
import png
import csv
from helpers import *

out_path_base = '../../data/commaai/test_files/'
filepath = r'../../data/commaai/destination/'

test_filenames = np.load('test_files_run2.npy', allow_pickle = True)

density_path = '../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)

print('start writing tfrecords')
convert_val(test_filenames, filepath,  out_path_base, density_path)


