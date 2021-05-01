import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from scipy import integrate
import pandas as pd
import ray
import numpy as np

# read in val data
density_path = '../../../../data/commaai/density/gaussian_density_filtered.csv'
density = pd.read_csv(density_path)

no_points = 750
grid = np.linspace(min(density['axes']), max(density['axes']), int(no_points))

density_ext_list = np.load('../../../../data/commaai/predictions/density_dropout_mdn.npy')

j = 0
cdf_mdn = [] 
for supp in tqdm(density_ext_list):
    dens = density_ext_list[j,:]
    def f_integral(i):
        return(integrate.trapz(dens[i:(i+2)], grid[i:(i+2)]))
    with multiprocessing.Pool(20) as proc:
        probs = proc.map(f_integral, np.array([i for i in range(0, grid.shape[0])]))
    cdf_i = np.cumsum(np.array(probs))
    cdf_mdn.append(cdf_i)
    j += 1

np.save('../../../../data/commaai/predictions/cdf_is_mdn.npy', np.array(cdf_mdn))