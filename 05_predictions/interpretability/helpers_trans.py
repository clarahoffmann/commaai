from scipy.integrate import simps
import tensorflow as tf
import numpy as np
import math
from scipy.stats import norm
import scipy.stats
import pandas as pd

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
def Fy(y, density):
    integral = density.loc[find_closest_element(y, density['axes']),'cdf']
    return(integral)  


def imgs_input_fn(filepath, perform_shuffle=False, repeat_count=1, batch_size=32): 
    
    # reads in single training example and returns it in a format that the estimator can
    # use
    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                            "label": tf.io.FixedLenFeature([], tf.float32),
                            'rows': tf.io.FixedLenFeature([], tf.int64),
                            'cols': tf.io.FixedLenFeature([], tf.int64),
                            'depth': tf.io.FixedLenFeature([], tf.int64)}

        # Load one example
        parsed_example = tf.io.parse_single_example(proto, keys_to_features)

        image_shape = image_shape = tf.stack([640 , 360, 3])
        image_raw = parsed_example['image']

        label = tf.cast(parsed_example['label'], tf.float32)
        image = tf.io.decode_raw(image_raw, tf.uint8)
        image = tf.cast(image, tf.float32)

        image = tf.reshape(image, image_shape)

        return {'image':image},label
    
    dataset = tf.data.TFRecordDataset(filenames=filepath)
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    
    return batch_features, batch_labels

def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-((x_m.T).dot(np.linalg.inv(covariance)).dot(x_m)) / 2))

def find_closest_element(y: float, arr: np.ndarray):
    index = np.searchsorted(arr,y)
    if index >= 1:
        res = [arr[index - 1], arr[index]]
    else:
        return np.array(index)

    if res[0] == res[1]:
        return np.array(index - 1)
    else:
        diff_pre = np.abs(y-res[0])
        diff_aft = np.abs(y-res[1])
        if diff_pre == diff_aft:
            return np.array(index - 1, index), 
        else:
            return index - 1 if diff_pre < diff_aft else index
        
        
def predict_density(x, grid, density_y, density_pdf, beta_t, Lambda_t):
    
    psi_x0 = x
    # compute p_Y(y_0) for each y in grid
    p_y_y0 = [density_pdf[find_closest_element(y_i,density_y)] for y_i in grid]
    
    f_eta_x0 = psi_x0.dot(beta_t)
    s_0_hat = math.sqrt(1 + psi_x0.dot(Lambda_t).dot(psi_x0))
    
    part_0 = s_0_hat*f_eta_x0
    part_1 = np.array([norm.ppf(Fy(y_i, density)) for y_i in grid])
    # here occur some issues with values of -inf
    # fix invalid values later ...
    #part_1[np.isinf(part_1)] = 0.01
    
    # compute the cdf of new ys
    phi_1_z = np.array([scipy.stats.norm(0, 1).pdf(y_i) for y_i in part_1])
    
    term_1 = scipy.stats.norm(0, 1).pdf((part_1- part_0) / s_0_hat)

    p_y_single_obs_whole_dens = (p_y_y0/phi_1_z)*(1/s_0_hat)*term_1
    return(p_y_single_obs_whole_dens)

def bring_df_to_correct_format(result, grid):
    
    result_t = zip(result)
    df = pd.DataFrame(result_t)
    df=df.T
    l=[df[x].apply(pd.Series).stack() for x in df.columns]

    s=pd.concat(l,1).reset_index(level=1,drop=True)
    s.columns=df.columns
    s = s.set_index(grid[2:])
    
    return s

def find_closest_element(y: float, arr: np.ndarray):
    index = np.searchsorted(arr,y)
    if index >= 1 and index < len(arr):
        res = [arr[index - 1], arr[index]]
    elif index == len(arr) :
        return np.array(index - 1)
    else:
        return index

    if res[0] == res[1]:
        return np.array(index - 1)
    else:
        diff_pre = np.abs(y-res[0])
        diff_aft = np.abs(y-res[1])
        if diff_pre == diff_aft:
            return np.array(index - 1)
        else:
            return index - 1 if diff_pre < diff_aft else index
        
def density_at_true_value(density, true_y, true_z, B_zeta, tau_sq, beta_t):
    
    axes = np.array(density['axes'])
    p_y_y0 = [density.loc[find_closest_element(y_i,density['axes'])]['pdf']  for y_i in true_y]
    
    f_eta_x0 = B_zeta.dot(beta_t)
    s_0_hat = np.sqrt(1 + tau_sq*np.array([B_zeta[j,:].T.dot(B_zeta[j,:]) for j in range(0,n)]))
    phi_1_z = np.array([scipy.stats.norm(0, 1).pdf(z_i) for z_i in true_z])
    term_1 = (true_z - s_0_hat*f_eta_x0)/(s_0_hat)
    term_2 = np.array([scipy.stats.norm(0, 1).pdf(z_i) for z_i in term_1])
    p_y0_x0 = (p_y_y0/phi_1_z)*(1/s_0_hat)*term_2
    
    return p_y0_x0