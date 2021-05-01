
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from scipy import integrate
from utils import  find_closest_element, Fy
import pandas as pd
from scipy.stats import norm

density_path = '../../../../data/commaai/density/gaussian_density.csv'
density = pd.read_csv(density_path)

x1 = np.load('../../../../data/commaai/predictions/mc_preds_cil_1_neu.npy').reshape(-1,1000)
x2 = np.load('../../../../data/commaai/predictions/mc_preds_cil_2_neu.npy').reshape(-1,1000)
x3 = np.load('../../../../data/commaai/predictions/mc_preds_cil_3_neu.npy').reshape(-1,1000)
x4 = np.load('../../../../data/commaai/predictions/mc_preds_cil_4_neu.npy').reshape(-1,1000)

mc_preds = np.append(np.append(np.append(x1, x2, axis = 0), x3, axis = 0), x4, axis = 0)

#mc_preds_y = []
#for i in tqdm(range(0,mc_preds_z.shape[0])):
#    tr = [density.loc[find_closest_element(norm.cdf(float(x)), density['cdf']),'axes'] for x in mc_preds_z[i,:,:]]
#    mc_preds_y.append(tr)

#mc_preds = np.array(mc_preds_y)
#np.save('../../../../data/commaai/predictions/mc_dropout/mc_preds_y_cil.npy', mc_preds)

#mc_preds = np.append(np.load('../../../../data/commaai/predictions/mc_dropout/mc_preds_cil_y_1.npy'),
           #np.load('../../../../data/commaai/predictions/mc_dropout/mc_preds_cil_y_2.npy')).reshape(-1,1000)

no_points = 750
grid = np.linspace(min(density['axes']), max(density['axes']), int(no_points))

densities = []
supports = []
for i in tqdm(range(0, mc_preds.shape[0])):
    kde = sm.nonparametric.KDEUnivariate(mc_preds[i,:])
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

np.save('../../../../data/commaai/predictions/mc_dropout/density_dropout_cil.npy', np.array(density_ext_list))
#density_ext_list = list(np.load('../../../../data/commaai/predictions/mc_dropout/density_dropout_cil.npy'))

j = 0
cdf_mc_dropout = [] 
for supp in tqdm(range(0,len(density_ext_list))):
    grid = grid
    dens = density_ext_list[j]
    def f_integral(i):
        return(integrate.trapz(dens[i:(i+2)], grid[i:(i+2)]))
    with multiprocessing.Pool(20) as proc:
        probs = proc.map(f_integral, np.array([i for i in range(0, grid.shape[0])]))
    cdf_i = np.cumsum(np.array(probs))
    cdf_mc_dropout.append(cdf_i)
    j += 1
    if j % 1000 == 0:
        np.save('../../../../data/commaai/predictions/mc_dropout/cdf_mc_dropout_cil_is.npy', np.array(cdf_mc_dropout))
        
cdf_mc_dropout = np.array(cdf_mc_dropout)
np.save('../../../../data/commaai/predictions/mc_dropout/cdf_mc_dropout_cil_is.npy', cdf_mc_dropout)