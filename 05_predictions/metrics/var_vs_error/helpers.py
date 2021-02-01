import numpy as np
from scipy.stats import norm
from scipy.integrate import simps
from tqdm import tqdm

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

def error_vs_variance(result, grid):
    mse_list = []
    for i in grid:
        index = np.array(result['variance prediction'] < i)
        if np.sum(index) > 0:
            mse = np.mean(np.abs(true_y[index] -  np.array(result['mean prediction'])[index])**2)
        else:
            mse = float('NaN')
        mse_list.append(mse)
    return(np.array(mse_list))
