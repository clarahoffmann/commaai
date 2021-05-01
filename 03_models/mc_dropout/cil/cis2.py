import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from scipy import integrate
import pandas as pd
from helpers import confidence_intervals

mc_preds = np.append(np.load('../../../../data/commaai/predictions/mc_dropout/mc_preds_cil_1.npy'), 
                     np.load('../../../../data/commaai/predictions/mc_dropout/mc_preds_cil_2.npy'), axis = 0)

densities = []
supports = []
for i in tqdm(range(0, mc_preds.shape[0])):
    kde = sm.nonparametric.KDEUnivariate(mc_preds[i,:,:])
    kde.fit() # Estimate the densities
    support = kde.support
    endog = kde.endog
    density = kde.density
    supports.append(support)
    densities.append(density) 
    
    
j = 0
cdf_mc_dropout = [] 
for supp in tqdm(0,len(densities))):
    grid = supports[j]
    dens = densities[j]
    def f_integral(i):
        return(integrate.trapz(dens[i:(i+2)], grid[i:(i+2)]))
    with multiprocessing.Pool(20) as proc:
        probs = proc.map(f_integral, np.array([i for i in range(0, grid.shape[0])]))
    cdf_i = np.cumsum(np.array(probs))
    cdf_mc_dropout.append(cdf_i)
    j += 1

cdf_mc_dropout = np.array(cdf_mc_dropout)
np.save('../../../../data/commaai/predictions/mc_dropout/cdf_mc_dropout_cil_is2.npy', cdf_mc_dropout)