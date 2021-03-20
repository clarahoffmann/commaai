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

def confidence_intervals(alphas, cdf_is, true_y, grid):
    all_conf_int = []
    for alpha2 in tqdm(alphas):
        confidence_intervals = []
        alpha = 1- alpha2
        i = 0
        for cdf in cdf_is:
            lb = max(grid[cdf <= alpha/2])
            try:
                ub = min(grid[cdf >= 1-alpha/2])
            except: 
                print('error at index:' + str(i))
                ub = max(grid)
            i += 1
            confidence_intervals.append([lb, ub])
        all_conf_int.append(confidence_intervals)
    
    # prediction interval width
    conf_widths = []
    for level in range(0, np.append(np.linspace(0.05, 0.95, 10), float(0.99)).shape[0]):  
        conf_width = np.array([all_conf_int[level][i][1] - all_conf_int[level][i][0] for i in range(0, len(cdf_is))])
        conf_widths.append(conf_width)
    
    coverage_rate = []
    # prediction interval coverage rate
    for i in range(0, np.append(np.linspace(0.05, 0.95, 10), float(0.99)).shape[0]):
        confidence_intervals = all_conf_int[i]
        in_interval = []
        # loop over single PI 
        for i in range(0, len(true_y)):
            conf_int = confidence_intervals[i]
            if conf_int[0] <= true_y[i] <= conf_int[1]:
                in_interval.append(1)
            else:
                in_interval.append(0)
        mean_int = np.mean(in_interval)
        coverage_rate.append(mean_int)
    
    return({'prediction_intervals': all_conf_int,
            'pred_int_width': conf_widths, 
            'coverage_rate': coverage_rate})