import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from scipy import integrate
from helpers import Fy, find_closest_element, get_densities, get_cdf, predict_single_density
import multiprocessing

class comp_dens():
    
    def __init__(self, density, no_points, B_zeta):
        
        self.density = density
        self.no_points = no_points
        self.density_y = self.density['axes']
        self.density_pdf = self.density['pdf']
        self.grid = np.linspace(min(self.density['axes']), max(self.density['axes']), self.no_points) #support #
        # compute these beforehand to save computation time
        self.p_y_y0 = [self.density_pdf[find_closest_element(y_i,self.density_y)] for y_i in self.grid]
        self.part_1 = np.array([norm.ppf(Fy(y_i, self.density)) for y_i in self.grid])
        self.phi_1_z = np.array([norm(0, 1).pdf(y_i) for y_i in self.part_1 ])
        self.B_zeta = B_zeta
        
    def compute_dens(self, method):
        
        self.method = method
        
        # choose method and read in parameters
        if self.method == 'va_horseshoe':
            self.va_horse_dir = '../../../../data/commaai/va/unfiltered_gaussian_resampled/Horseshoe/'
            self.mu_t_va = np.load(self.va_horse_dir + 'mu_ts2_new_dev_1.npy')
            self.iteration = self.mu_t_va.shape[0]
            self.p = 10
            self.B_ts = np.mean(np.load(self.va_horse_dir + 'B_ts2_new_dev_1.npy')[int(0.9*self.iteration):,:,:], axis = 0)
            self.d_ts = np.mean(np.load(self.va_horse_dir + 'd_ts2_new_dev_1.npy')[int(0.9*self.iteration):,:,:], axis = 0)
            self.var = np.sqrt(np.diag(self.B_ts.dot(self.B_ts.T) + self.d_ts**2))
            self.beta_va = np.mean(self.mu_t_va[int(0.9*self.iteration):,0:10], axis = 0)
            self.Lambdas_log = np.mean(self.mu_t_va[int(0.9*self.iteration):,10:20], axis = 0)
            self.samples = np.exp(0.5*np.random.multivariate_normal(self.Lambdas_log.reshape(10,), np.diag(self.var[10:20]), 10000))
            
            self.Lambdas = np.mean(self.samples, axis = 0)
            self.beta = self.beta_va
            self.tau = None
        
        if self.method == 'hmc_horseshoe':
            self.hmc_thetas = np.load('../../../../data/commaai/mcmc/unfiltered_gaussian_resampled/Horseshoe/all_thetas_try.npy')
            self.hmc_thetas[:,10:20] = np.exp(0.5*self.hmc_thetas[:,10:20])
            self.hmc_thetas[:,20] = np.exp(self.hmc_thetas[:,20])
            self.means = np.mean(self.hmc_thetas[1000:], axis = 0)
            
            self.beta = self.means[0:10]
            self.Lambdas = self.hmc_thetas[1000:,10:20] 
            self.p = 10
            self.tau = None
            
        if self.method == 'va_ridge':
            
            self.va_ridge_dir = '../../../../data/commaai/va/unfiltered_gaussian_resampled/Ridge/'
            self.mu_t_va = np.load(self.va_ridge_dir + 'mu_ts23_factor_50.npy')
            self.iteration = self.mu_t_va.shape[0]
            self.beta = np.mean(self.mu_t_va[int(0.9*self.iteration):self.iteration,0:10], axis = 0)
            self.tau = np.exp(np.mean(self.mu_t_va[int(0.9*self.iteration):self.iteration,10], axis = 0))
            self.p = len(self.beta)
            self.Lambdas = None
            
        if self.method == 'hmc_ridge':
            self.hmc_ridge_dir = '../../../../data/commaai/mcmc/unfiltered_gaussian_resampled/Ridge/'
            self.hmc_thetas = np.load(str(self.hmc_ridge_dir + 'all_thetas_L100_5000.npy'))[1000:, :]
            self.beta = np.mean(self.hmc_thetas[:,0:10], axis = 0)
            self.tau = np.exp(self.hmc_thetas[:,10])
            self.p = 10
            self.Lambdas = None
            
        # compute posterior response densities
        self.densities = get_densities(self.B_zeta, self.grid, self.p_y_y0, self.part_1, self.phi_1_z, self.beta, self.tau, self.p,  self.method, self.Lambdas)
        
        # marginal posterior response density
        self.av_density = np.mean(np.array(self.densities), axis = 0)
            
        return({'densities': self.densities,
                'av_density': self.av_density})