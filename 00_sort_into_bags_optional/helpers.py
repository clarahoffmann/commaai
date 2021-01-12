import numpy as np
import pandas as pd
import cv2
import imageio

def read_vid_angles(vid_path, value_path, t_path):
    
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
    
    rez_frames = []
    for frame in frames:
        rez_frame = cv2.resize(frame, dsize = (291,218), interpolation = cv2.INTER_LINEAR)[76:142, 45:245,0:3]
        rez_frames.append(rez_frame)
    
    rez_frames = np.array(rez_frames)
    # get angles per frame
    target_angles = [angles.loc[find_closest_element(timestamps_frames[i], np.array(angles['t'])),'angle'] for i in range(0, len(timestamps_frames))]
    
    print(len(rez_frames))
    trans_label = [ 0  for i in range(0, len(target_angles))] 
    
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
        