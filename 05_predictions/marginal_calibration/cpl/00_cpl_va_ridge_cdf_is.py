import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from scipy.integrate import simps
import matplotlib.pyplot as plt
import imageio
import multiprocessing
from scipy import integrate
from helpers import Fy, find_closest_element, get_densities, get_cdf, predict_single_density
import statsmodels.api as sm
from compute_dens import comp_dens

# read in val data
B_zeta = np.load('../../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/Bzeta/B_zeta_val.npy')
true_y = np.load('../../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/Bzeta/labels_val.npy')
B_zeta = B_zeta[np.abs(true_y) < 50,:] 
true_y = true_y[np.abs(true_y) < 50]

density_path = '../../../../data/commaai/density/gaussian_density_filtered.csv'
density = pd.read_csv(density_path)

no_points = 750
grid = np.linspace(min(density['axes']), max(density['axes']), no_points)

# estimate density 
kde = sm.nonparametric.KDEUnivariate(true_y)
kde.fit()
support = kde.support
endog = kde.endog
density_gauss = kde.density

dens_pred = comp_dens(density, no_points, B_zeta)

hmc_horse = dens_pred.compute_dens('va_ridge')

np.save('../../../../data/commaai/calibration/cpl_dens/va_ridge_av_dens.npy', hmc_horse['av_density'])

# hmc_horse
densities = hmc_horse['densities']
cdf_va_is = []
j = 0
for supp in tqdm(densities):
    dens = densities[j]
    def f_integral(i):
        return(integrate.trapz(dens[i:(i+2)], grid[i:(i+2)]))
    with multiprocessing.Pool(20) as proc:
        probs = proc.map(f_integral, np.array([i for i in range(0, grid.shape[0])]))
    cdf_va_i = np.cumsum(np.array(probs))
    cdf_va_is.append(cdf_va_i)
    j += 1
cdf_va_is = np.array(cdf_va_is)
np.save('../../../../data/commaai/calibration/cpl_dens/va_ridge_cdf_is.npy', cdf_va_is)