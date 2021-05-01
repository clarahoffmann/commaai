import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from scipy import integrate
import pandas as pd
import ray
import numpy as np

samples = np.load('../../../../data/commaai/predictions/mdn/cil/samples.npy').reshape(-1,1000)

# read in val data
density_path = '../../../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)

no_points = 750
grid = np.linspace(min(density['axes']), max(density['axes']), int(no_points))

densities = []
supports = []
for i in tqdm(range(0, samples.shape[0])):
    kde = sm.nonparametric.KDEUnivariate(samples[i,:])
    kde.fit() # Estimate the densities
    support = kde.support
    endog = kde.endog
    density = kde.density
    supports.append(support)
    densities.append(density) 
    
density_ext_list = []
density_ext = np.array(np.repeat(0, 750), dtype=float)
for j in tqdm(range(0, len(densities))):
    density_ext = np.array(np.repeat(0, 750), dtype=float)
    for i in range(0, supports[j].shape[0]):
        density_ext[np.where(np.abs(grid - supports[j][i]) == min(np.abs(grid - supports[j][i])))] = densities[j][i]
    density_ext_list.append(density_ext)
density_ext_list = np.array(density_ext_list)

np.save('../../../../data/commaai/predictions/density_cil_mdn.npy', density_ext_list)