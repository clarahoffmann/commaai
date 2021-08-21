from helpers import (Fy, find_closest_element,  compute_coverage, #predict_single_density,
confidence_interval, confidence_interval, generate_fixed_terms, get_ci)
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from scipy.integrate import trapz

class density_predictor():
    
    def __init__(self, B_zeta, true_y, density, no_points):
        # initialize
        self.B_zeta = B_zeta
        self.true_y = true_y
        self.density = density
        self.no_points = no_points
        self.p = self.B_zeta.shape[1]
        
        # compute fixed terms and grid
        self.p_y_y0,  self.part_1, self.phi_1_z, self.grid = generate_fixed_terms(self.no_points, self.density)
        
    def generate_fixed_terms(self):
    
        self.grid = np.linspace(min(self.density['axes']), max(self.density['axes']), self.no_points)
        self.density_y = self.density['axes']
        self.density_pdf = self.density['pdf']
        # compute these beforehand to save computation time
        self.p_y_y0 = [self.density_pdf[find_closest_element(y_i,density_y)] for y_i in self.grid]
        self.part_1 = np.array([norm.ppf(Fy(y_i, self.density)) for y_i in self.grid])
        self.phi_1_z = np.array([norm(0, 1).pdf(y_i) for y_i in self.part_1 ])
        
        return(self.p_y_y0,  self.part_1, self.phi_1_z, self.grid)
    
    def predict_single_density(self, x, grid, p_y_y0, part_1, phi_1_z, beta, tau_sq, Lambda, method):
    
    
        self.psi_x0 = x
        if self.method == 'dnn_c':
            self.f_eta_x0 = self.psi_x0.dot(self.beta[0:p]) + self.beta[p]
        else:
            self.f_eta_x0 = self.psi_x0.dot(self.beta)

        if self.method == 'va_ridge':
            self.s_0_hat = (1 + self.tau_sq*self.psi_x0.dot(self.psi_x0))**(-0.5)

        elif self.method == 'hmc_ridge':
            self.s_0_hats =  []
            for tau_j in self.tau_sq:
                self.s_0_hatj = (1 + tau_j*self.psi_x0.dot(self.psi_x0))**(-0.5)
                self.s_0_hats.append(self.s_0_hatj)
            self.s_0_hat = np.mean(np.array(self.s_0_hats))

        elif self.method == 'va_horseshoe':
            self.s_0_hat = (1 + (self.psi_x0*(self.Lambda**2)).dot(self.psi_x0))**(-0.5)

        elif self.method == 'hmc_horseshoe':
            self.s_0_hats =  []
            for Lambda_j in self.Lambda:
                self.s_0_hatj = (1 + (self.psi_x0*(Lambda_j**2)).dot(self.psi_x0))**(-0.5)
                self.s_0_hats.append(self.s_0_hatj)
            self.s_0_hat = np.mean(np.array(self.s_0_hats))
            
        if self.method == 'dnn_c':
            self.s_0_hat = (1 + self.psi_x0.dot(self.psi_x0))**(-0.5)

        self.part_0 = self.s_0_hat*self.f_eta_x0

        # compute the cdf of new ys
        self.term_1 = norm(0, 1).pdf((self.part_1- self.part_0) / self.s_0_hat)
        self.p_y_single_obs_whole_dens = (self.p_y_y0/self.phi_1_z)*(1/self.s_0_hat)*self.term_1

        return(self.p_y_single_obs_whole_dens)
    
    def get_density(self, method):
        
        self.method = method
        
        if self.method == 'va_ridge':
            
            self.va_ridge_dir = '../../../../data/commaai/va/unfiltered_gaussian_resampled/Ridge/'
            self.mu_t_va = np.load('../../../../data/commaai/va/unfiltered_gaussian_resampled/Ridge/mu_ts_new.npy')
            self.iterations = self.mu_t_va.shape[0]
            self.beta = np.mean(self.mu_t_va[int(0.9*self.iterations):self.iterations,0:10], axis = 0)
            self.beta_sd = np.std(self.mu_t_va[int(0.9*self.iterations):self.iterations,0:10], axis = 0)
            self.tau_sq = np.mean(np.exp(self.mu_t_va[int(0.9*self.iterations):self.iterations,10]), axis = 0)
            
            print('computing densities for each observation')
            self.densities_va = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.dens = self.predict_single_density(self.B_zeta[i].reshape(self.p,), self.grid, self.p_y_y0, 
                                                   self.part_1, self.phi_1_z, self.beta, self.tau_sq, None, 
                                                   self.method)
                self.densities_va.append(self.dens)
            
            print('computing mean prediction for each observation')
            # mean prediction & maximum density prediction
            self.pred_y_va_ridge = []
            self.max_dens = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.y_i = trapz(self.densities_va[i]*self.grid, self.grid)
                self.pred_y_va_ridge.append(self.y_i)
                self.max_dens.append(self.grid[self.densities_va[i] ==  max(self.densities_va[i])])
            self.pred_y_va_ridge = np.array(self.pred_y_va_ridge)
            self.max_dens = np.array(self.max_dens)
            
            print('computing variance prediction for each observation')
            # variance prediction
            self.pred_y_va_ridge_var = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.y_i = trapz(self.densities_va[i]*((self.grid - self.pred_y_va_ridge[i])**2), self.grid)
                self.pred_y_va_ridge_var.append(self.y_i)
            
            self.pred_y_va_ridge_var = np.array(self.pred_y_va_ridge_var)
            
            return({'densities': self.densities_va, 
                    'mean predictions': self.pred_y_va_ridge, 
                    'variance preditcion': self.pred_y_va_ridge_var,
                   'max dens prediction': np.array(self.max_dens)})
        
        if self.method == 'va_horseshoe':
        
            self.va_horse_dir = '../../../../data/commaai/va/unfiltered_gaussian_resampled/Horseshoe/'
            self.mu_t_va = np.load(self.va_horse_dir + 'mu_ts_new.npy').reshape(-1, 21)
            self.iter = self.mu_t_va.shape[0]
            self.beta = np.mean(self.mu_t_va[int(0.95*self.iter):,0:10], axis = 0)
            self.Lambda = np.mean(np.exp(0.5*self.mu_t_va[int(0.95*self.iter):,10:20]), axis = 0)
            self.tau_sq = np.exp(np.mean(self.mu_t_va[int(0.95*self.iter):,20], axis = 0))
            self.z_pred = self.B_zeta.reshape(self.B_zeta.shape[0], self.p).dot(self.beta)
            self.pred_y = [self.density.loc[find_closest_element(norm.cdf(z), self.density['cdf']), 'axes'] for z in self.z_pred]
            
            self.densities_va = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.dens = self.predict_single_density(self.B_zeta[i].reshape(self.p,), self.grid, self.p_y_y0, 
                                                   self.part_1, self.phi_1_z, self.beta, self.tau_sq, self.Lambda, 
                                                   'va_horseshoe')
                self.densities_va.append(self.dens)
               
            self.pred_y_va_horse = []
            self.max_dens = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.y_i = trapz(self.densities_va[i]*self.grid, self.grid)
                self.pred_y_va_horse.append(self.y_i)
                self.max_dens.append(self.grid[self.densities_va[i] ==  max(self.densities_va[i])])
            
            self.pred_y_va_horse = np.array(self.pred_y_va_horse)
            self.max_dens = np.array(self.max_dens)
            
            # variance prediction
            self.pred_y_va_horse_var = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.y_i = trapz(self.densities_va[i]*((self.grid - self.pred_y_va_horse[i])**2), self.grid)
                self.pred_y_va_horse_var.append(self.y_i)
                
            return({'densities': self.densities_va, 
                    'mean prediction': self.pred_y_va_horse, 
                    'variance prediction': self.pred_y_va_horse_var,
                   'max dens prediction': np.array(self.max_dens)})
        
        if self.method == 'hmc_ridge':
            
            self.hmc_ridge_dir = '../../../../data/commaai/mcmc/unfiltered_gaussian_resampled/Ridge/'
            self.mu_t_hmc = np.load(self.hmc_ridge_dir + 'all_thetas_new.npy')[500:,:]
            self.beta = np.mean(self.mu_t_hmc[:,0:10], axis = 0)
            self.tau_sq = np.exp(self.mu_t_hmc[:,10])
            self.z_pred = self.B_zeta.reshape(self.B_zeta.shape[0], self.p).dot(self.beta)
            self.pred_y = [self.density.loc[find_closest_element(norm.cdf(z), self.density['cdf']), 'axes'] for z in self.z_pred]
            
            self.densities_va = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.dens = self.predict_single_density(self.B_zeta[i].reshape(self.p,), self.grid, self.p_y_y0, 
                                              self.part_1, self.phi_1_z, self.beta, self.tau_sq, None, 'hmc_ridge')
                self.densities_va.append(self.dens)

            self.pred_y_hmc_ridge = []
            self.max_dens = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.y_i = trapz(self.densities_va[i]*self.grid, self.grid)
                self.pred_y_hmc_ridge.append(self.y_i)
                self.max_dens.append(self.grid[self.densities_va[i] ==  max(self.densities_va[i])])
            

            # variance prediction
            self.pred_y_hmc_ridge_var = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.y_i = trapz(self.densities_va[i]*((self.grid - self.pred_y_hmc_ridge[i])**2), self.grid)
                self.pred_y_hmc_ridge_var.append(self.y_i)

            return({'densities': self.densities_va, 
                    'mean prediction': np.array(self.pred_y_hmc_ridge), 
                    'variance prediction': np.array(self.pred_y_hmc_ridge_var),
                   'max dens prediction': np.array(self.max_dens)})

        if self.method == 'hmc_horseshoe':
            self.hmc_ridge_dir = '../../../../data/commaai/mcmc/unfiltered_gaussian_resampled/Horseshoe/'
            self.mu_t_hmc = np.load(self.hmc_ridge_dir + 'all_thetas_new.npy').reshape(-1, 21)
            self.beta = np.mean(self.mu_t_hmc[15000:,0:10], axis = 0)
            self.Lambda = np.exp(0.5*self.mu_t_hmc[15000:,10:20])
            self.tau_sq = np.exp(self.mu_t_hmc[15000:,20])
            self.z_pred = self.B_zeta.reshape(self.B_zeta.shape[0], self.p).dot(self.beta)
            self.pred_y = [self.density.loc[find_closest_element(norm.cdf(z), self.density['cdf']), 'axes'] for z in self.z_pred]
            
            self.densities_va = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.dens = self.predict_single_density(self.B_zeta[i].reshape(self.p,), self.grid, self.p_y_y0, 
                                              self.part_1, self.phi_1_z, self.beta, self.tau_sq, None, 'hmc_horseshoe')
                self.densities_va.append(self.dens)

            self.pred_y_hmc_ridge = []
            self.max_dens = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.y_i = trapz(self.densities_va[i]*self.grid, self.grid)
                self.pred_y_hmc_ridge.append(self.y_i)
                self.max_dens.append(self.grid[self.densities_va[i] ==  max(self.densities_va[i])])

            # variance prediction
            self.pred_y_hmc_ridge_var = []
            for i in tqdm(range(0, self.B_zeta.shape[0])):
                self.y_i = trapz(self.densities_va[i]*((self.grid - self.pred_y_hmc_ridge[i])**2), self.grid)
                self.pred_y_hmc_ridge_var.append(self.y_i)
       
            return({'densities': self.densities_va, 
                    'mean prediction': np.array(self.pred_y_hmc_ridge), 
                    'variance prediction': np.array(self.pred_y_hmc_ridge_var),
                   'max dens prediction': np.array(self.max_dens)})