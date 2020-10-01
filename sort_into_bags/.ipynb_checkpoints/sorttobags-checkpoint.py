import glob
import os
import numpy as np
import cv2
from scipy.stats import norm
import imageio
import pandas as pd


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

train_vid_files = [str(filepath + train_filenames[i] + '.hevc') for i in range(len(train_filenames))]
train_yaw_files = [str(filepath + train_filenames[i] + '.value') for i in range(len(train_filenames))]
train_time_files = [str(filepath + train_filenames[i] + '.t') for i in range(len(train_filenames))]

density_path = '../../data/commaai/density/densityfastkde_density.csv'
density = pd.read_csv(density_path)

def read_vid_angles(vid_path, value_path, t_path, density):
    
    # read video
    vid = imageio.get_reader(vid_path,  'ffmpeg')
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
    
    # downsamples images
    new_height = 174
    new_width = 131
    rez_frames = []
    for i in range(0, frames.shape[0]):
        frames_i = cv2.resize(frames[i, :, :, :], dsize = (new_height,new_width), interpolation = cv2.INTER_LINEAR)
        rez_frames.append(frames_i)
    frames_i = np.array(frames_i)
    # return every 5th frame
    print(len(rez_frames))
    trans_label = [norm.ppf(Fy(target_angles[i], density)) for i in range(0, len(target_angles))]
    
    return(rez_frames[::5], target_angles[::5], trans_label)

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
        
for j in range(0, 500): #len(train_vid_files)
    # get single file
    video_file = train_vid_files[j]
    angle_file = train_yaw_files[j]
    time_file = train_time_files[j]
    
    images, yaw, trans_label = read_vid_angles(video_file, angle_file, time_file, density)
    
    for i, (img, label, tr_label) in enumerate(zip(images, yaw, trans_label)):
        
        if abs(label) <= 10:
            path = '../../data/commaai/train_bags/1/'
            filename = str(path + str(i) + '_' + str(j) + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
        
        elif abs(label) <= 20 and abs(label) > 10:
            path = '../../data/commaai/train_bags/2/'
            filename = str(path + str(i) + '_' + str(j) + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
        
        elif abs(label) <= 30 and abs(label) > 20:
            path = '../../data/commaai/train_bags/3/'
            filename = str(path + str(i) + '_' + str(j) + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
        
        elif abs(label) <= 40 and abs(label) > 30:
            path = '../../data/commaai/train_bags/4/'
            filename = str(path + str(i) + '_' + str(j) + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
        
        elif abs(label) <= 50 and abs(label) > 40:
            path = '../../data/commaai/train_bags/5/'
            filename = str(path + str(i) + '_' + str(j) + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
        
        elif abs(label) <= 60 and abs(label) > 50:
            path = '../../data/commaai/train_bags/6/'
            filename = str(path + str(i) + '_' + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
        
        elif abs(label) <= 80 and abs(label) > 60:
            path = '../../data/commaai/train_bags/7/'
            filename = str(path + str(i) + '_' + str(j) + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
        
        elif abs(label) <= 120 and abs(label) > 80:
            path = '../../data/commaai/train_bags/8/'
            filename = str(path + str(i) + '_' + str(j) + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
        
        elif abs(label) <= 150 and abs(label) > 120:
            path = '../../data/commaai/train_bags/9/'
            filename = str(path + str(i) + '_' + str(j) + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
        
        elif abs(label) <= 180 and abs(label) > 150:
            path = '../../data/commaai/train_bags/10/'
            filename = str(path + str(i) + '_' + str(j) + 'run1')
            np.save(filename, np.array([img, label, tr_label]))
