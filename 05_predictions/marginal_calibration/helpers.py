import numpy as np
from scipy.stats import norm
from scipy.integrate import simps
from tqdm import tqdm
from scipy import integrate

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

def predict_single_density(x, grid, p_y_y0, part_1, phi_1_z, beta, tau_sq, Lambda, method):
    
    psi_x0 = x

    f_eta_x0 = psi_x0.dot(beta)

    if method == 'va_ridge' or method == 'hmc_ridge':
        s_0_hat = (1 + tau_sq*psi_x0.dot(psi_x0))**(-0.5)

    elif method == 'va_horseshoe' or method == 'hmc_horseshoe':
        s_0_hat = (1 + (psi_x0*Lambda**2).dot(psi_x0))**(-0.5)

    part_0 = s_0_hat*f_eta_x0

    # compute the cdf of new ys
    term_1 = norm(0, 1).pdf((part_1- part_0) / s_0_hat)
    p_y_single_obs_whole_dens = (p_y_y0/phi_1_z)*(1/s_0_hat)*term_1

    return(p_y_single_obs_whole_dens)

def confidence_interval(densities, pred_y, density, confidence_level, grid):

    integrals = []

    for i in range(0, len(densities) - 1):
        integral = simps([max(0,densities[i]),max(0,densities[i + 1])], [grid[i], grid[i+1]])
        integrals.append(integral)
    int_sum = 0
    i = 1
    start_index = np.searchsorted(density['axes'], pred_y)

    int_sum = integrals[start_index]
    while int_sum < confidence_level:
        int_sum += integrals[start_index - i] + integrals[start_index + i]
        i += 1
    return([density.loc[start_index - i,'axes'], density.loc[start_index + i,'axes']])

def compute_coverage(confidence_intervals):
    coverage = []
    for i in range(0, len(confidence_intervals)):
        int_i = confidence_intervals[i]
        if  (true_y[i] >= int_i[0])  & (true_y[i] <= int_i[1]):
            covered = 1
        else:
            covered = 0
        coverage.append(covered)   
    return(np.mean(coverage))

def get_densities(B_zeta, grid, p_y_y0, part_1, phi_1_z, beta, tau_sq, p, method, Lambda):
    densities_va = []
    for i in tqdm(range(0, B_zeta.shape[0])):
        dens = predict_single_density(B_zeta[i].reshape(p,), grid, p_y_y0, part_1, phi_1_z, beta, tau_sq, Lambda, method)
        densities_va.append(dens)
    return(densities_va)

def get_cdf(true_y, grid, densities_va):
    
    y_i_index = [find_closest_element(y, grid) for y in true_y]
    
    def f_integral(i, density, grid):
        return(integrate.trapz(density[i:(i+2)], grid[i:(i+2)]))

    Fy_list = []
    j = 0
    for dens in tqdm(densities_va):
        Fy = []
        i = 0
        Fy = 0
        while i < y_i_index[j]:
            Fy_i = f_integral(i, dens, grid)
            Fy += Fy_i
            i += 1
        Fy_list.append(Fy)
        j += 1
    return(np.array(Fy_list))