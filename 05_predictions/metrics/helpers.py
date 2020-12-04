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

def predict_single_density(x, grid, p_y_y0, part_1, phi_1_z, beta, tau_sq, Lambda, method):
    
    psi_x0 = x
    if method == 'dnn_c':
        f_eta_x0 = psi_x0.dot(beta[0:p]) + beta[p]
    else:
        f_eta_x0 = psi_x0.dot(beta)

    if method == 'va_ridge':
        s_0_hat = (1 + tau_sq*psi_x0.dot(psi_x0))**(-0.5)
    
    elif method == 'hmc_ridge':
        s_0_hats =  []
        for tau_j in tau_sq:
            s_0_hatj = (1 + tau_j*psi_x0.dot(psi_x0))**(-0.5)
            s_0_hats.append(s_0_hatj)
        s_0_hat = np.mean(np.array(s_0_hats))

    elif method == 'va_horseshoe' or method == 'hmc_horseshoe':
        s_0_hat = (1 + (psi_x0*(Lambda**2)).dot(psi_x0))**(-0.5)
    
    if method == 'dnn_c':
        s_0_hat = (1 + psi_x0.dot(psi_x0))**(-0.5)

    part_0 = s_0_hat*f_eta_x0

    # compute the cdf of new ys
    term_1 = norm(0, 1).pdf((part_1- part_0) / s_0_hat)
    p_y_single_obs_whole_dens = (p_y_y0/phi_1_z)*(1/s_0_hat)*term_1

    return(p_y_single_obs_whole_dens)

def compute_coverage(confidence_intervals, true_y):
    coverage = []
    for i in range(0, len(confidence_intervals)):
        int_i = confidence_intervals[i]
        if  (true_y[i] >= int_i[0])  & (true_y[i] <= int_i[1]):
            covered = 1
        else:
            covered = 0
        coverage.append(covered)   
    return(np.mean(coverage))

def confidence_interval(densities, pred_y, density, confidence_level, grid):

    integrals = []

    for i in range(0, len(densities) - 1):
        integral = simps([max(0,densities[i]),max(0,densities[i + 1])], [grid[i], grid[i+1]])
        integrals.append(integral)
    Fy = np.cumsum(integrals)
    conf05 = grid[np.max((np.where(Fy <= 0.05)))]
    conf95 = grid[np.min((np.where(Fy >= 0.95)))]
    return([conf05, conf95])

def generate_fixed_terms(no_points, density):
    
    grid = np.linspace(min(density['axes']), max(density['axes']), no_points)
    density_y = density['axes']
    density_pdf = density['pdf']
    # compute these beforehand to save computation time
    p_y_y0 = [density_pdf[find_closest_element(y_i,density_y)] for y_i in grid]
    part_1 = np.array([norm.ppf(Fy(y_i, density)) for y_i in grid])
    phi_1_z = np.array([norm(0, 1).pdf(y_i) for y_i in part_1 ])

    return(p_y_y0, part_1, phi_1_z, grid)

def get_ci(B_zeta, grid, pred_y, density,  p_y_y0, part_1, phi_1_z, beta, tau_sq, p,  Lambda, method):
    densities_va = []
    for i in tqdm(range(0, B_zeta.shape[0])):
        dens = predict_single_density(B_zeta[i].reshape(p,), grid, p_y_y0, part_1, phi_1_z, beta, tau_sq, Lambda, method)
        densities_va.append(dens)
    confidence_intervals = []
    
    for i in tqdm(range(0, len(pred_y))):
        confidence_intervals.append(confidence_interval(densities_va[i], pred_y[i], density, 0.95, grid))
    conf_width = [element[1] - element[0] for element in confidence_intervals]
    return(confidence_intervals, conf_width)

def KL(P,Q):
    epsilon = 0.00001
    P = P.copy() +epsilon
    Q = Q.copy() +epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence
