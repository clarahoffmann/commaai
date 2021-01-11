import numpy as np
import cv2
import imageio
import pandas as pd
from tqdm import tqdm
import scipy.misc
from helpers import find_closest_element, Fy
from scipy.stats import norm

test_filenames = np.load('test_files_run2.npy')
filepath = r'../../../../data/commaai/destination/'
density_path = '../../../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)
checkpoint_path = '../../../../data/models/20201021_unrestr_gaussian_resampled/'
extracted_coefficients_directory_Bzeta = '../../../../data/commaai/extracted_coefficients/20201021_unrestr_gaussian_resampled/Bzeta/'
destination = '../../../../data/commaai/test_files/val_files_unfiltered/'

train_vid_files = [str(filepath + test_filenames[i] + '.hevc') for i in range(len(test_filenames))]
train_yaw_files = [str(filepath + test_filenames[i] + '.value') for i in range(len(test_filenames))]
train_time_files = [str(filepath + test_filenames[i] + '.t') for i in range(len(test_filenames))]

def read_vid_angles(vid_path, value_path, t_path, density):
    print('hello')
    # read video
    vid = imageio.get_reader(vid_path,  'ffmpeg')
    frames = np.array([im for im in vid.iter_data()], dtype=np.uint8)
    vid.close()
    
    print('hello')
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
    for i in tqdm(range(1, len(timestamps_frames))):
        timestamps_frames[i] = timestamps_frames[i - 1] + 0.05
    
    # get angles per frame
    target_angles = [angles.loc[find_closest_element(timestamps_frames[i], np.array(angles['t'])),'angle'] for i in range(0, len(timestamps_frames))]
    
    # downsamples images
    #new_height = 291
    #new_width = 218
    frames = frames[::5,:,:,:]
    rez_frames = []
    for i in tqdm(range(0, frames.shape[0])):
        frames_i = cv2.resize(frames[i,:,:,:], dsize = (291,218), interpolation = cv2.INTER_LINEAR)[76:142, 45:245,0:3]/255
        rez_frames.append(frames_i)
    frames_i = np.array(frames_i)
    #return every 5th frame
    target_angles = target_angles[::5].copy()
    trans_label = [norm.ppf(Fy(target_angles[i], density)) for i in range(0, len(target_angles))]
    
    return(rez_frames, target_angles, trans_label)

df = pd.DataFrame({'path': [], 'true_y': [], 'true_z': []})
for j in tqdm(range(0, len(train_vid_files))):

    # get single yaw and video file path
    yaw_file = train_yaw_files[j]
    vid_file = train_vid_files[j]
    time_file = train_time_files[j]
    images, yaw, trans_label = read_vid_angles(vid_file, yaw_file, time_file, density)
    
    i = 0
    for img in images:
        filename = str(destination + 'image' + str(i) + '_' + str(j) + '.png')
        imageio.imwrite(filename, img)
        df = df.append({'path': filename, 'true_y': yaw[i], 'true_z': trans_label[i]}, ignore_index=True)
        i +=  1
    df.to_csv('df_paths.csv')

df.to_csv('df_paths.csv')