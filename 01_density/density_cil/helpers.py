import numpy as np

def find_closest_element(y: float, arr: np.ndarray):
    '''
    Find index of element closest to y from array arr
    
    Input:
        - y: value to which we want to find the closest value
        - arr: array in which we want to find the closest value
    
    Output:
        - index: index where element is closest to y in arr
    '''
    
    # get index
    index = np.searchsorted(arr,y)
    
    # handle on case-by-case basis
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
    '''
    Distribution function at y of numerical cdf
    Input:
        - y: scalar, value at which cdf should be evaluated
        - density: dataframe with columns: axes, cdf
    
    Output: 
        - integral: cdf at y
    '''
    integral = density.loc[find_closest_element(y, density['axes']),'cdf']
    return(integral)  