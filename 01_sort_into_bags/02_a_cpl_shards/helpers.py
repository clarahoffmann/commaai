import numpy as np
import pandas as pd
import tensorflow as tf

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def Fy(y, density):
    
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
                return np.array(index - 1) 
            else:
                return index - 1 if diff_pre < diff_aft else index
        
    integral = density.loc[find_closest_element(y, density['axes']),'cdf']
    return(integral)  