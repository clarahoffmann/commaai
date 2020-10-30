import numpy as np
import random
import math
from scipy.stats import multivariate_normal, norm
import derivatives as dev
from tqdm import tqdm
import matplotlib.pyplot as plt
import lambda_sampler as ls
import tau_sampler as ts
from scipy.stats import halfcauchy

extracted_coefficients_directory = '../../../data/commaai/extracted_coefficients/20201021_unrestr_gaussian_resampled/'
B_zeta_path = str(extracted_coefficients_directory + 'Bzeta/B_zeta.npy')
beta_path = str(extracted_coefficients_directory + 'beta/beta.csv')
z_path = str(extracted_coefficients_directory + 'Bzeta/tr_labels.npy')
beta = np.genfromtxt(beta_path, delimiter=',')
B_zeta = np.load(B_zeta_path)
B_zeta = B_zeta.reshape(B_zeta.shape[0], beta.shape[0])
z = np.load(z_path) #[0:B_zeta.shape[0]]
n = B_zeta.shape[0]
p = B_zeta.shape[1]

Lambda_old = np.repeat(1,p)
BoB = B_zeta**2
log_tau_old = math.exp(1)
dS2_old, ddS2_old, S2_old, S_old = dev.generate_dS2_ddS2_S2_S(Lambda_old, BoB)

# initialize Lambda
betaBt = beta.dot(B_zeta.T)

num_iterations = 50000
lambdaacc = np.zeros((len(Lambda_old),num_iterations))
all_betas = []
all_lambdas = []
tauacc = []
all_taus = []
for j in tqdm(range(0, num_iterations)):
    
    beta = ts.sample_beta(p, B_zeta, Lambda_old, S_old, z)
    all_betas.append(beta)
    
    betaBt = beta.dot(B_zeta.T)
    
    # update tau
    log_tau_new, decision_criterion_tau = ts.sample_tau(log_tau_old, Lambda_old, p)
    
    # decision to accept/reject tau
    if math.log(random.random()) <= decision_criterion_tau:
        log_tau_old = log_tau_new
        tauacc.append(1)
        all_taus.append(log_tau_new)
    else:
        tauacc.append(0)
    
    # update lambda
    for m in range(0, len(Lambda_old)):
        log_sq_lambda_new, decision_criterion_lambda, dS2_new, ddS2_new, S2_new, S_new, ddlunew, Lambda_new = ls.sample_lambda(m, 
                                                                                               log_tau_old, 
                                                                                               Lambda_old, p, beta, 
                                                                                               B_zeta, dS2_old, ddS2_old, 
                                                                                              S2_old, S_old, z, BoB, betaBt)

        if math.log(random.random()) <= decision_criterion_lambda:
            if m < p - 1:
                Lambda_old = np.append(np.append(Lambda_old[0:m], math.exp(0.5*log_sq_lambda_new)), Lambda_new[m + 1:])
            elif m == p - 1:
                Lambda_old = np.append(Lambda_old[0:m],  math.exp(0.5*log_sq_lambda_new))
            lambdaacc[m,j] += 1
            all_lambdas.append(Lambda_old)
            dS2_old, ddS2_old, S2_old, S_old = dS2_new, ddS2_new, S2_new, S_new
        
        else:
            lambdaacc[m,j] += 0
            all_lambdas.append(Lambda_old)  
            
np.save('lambdas_train.npy', all_lambdas)
np.save('lambda_acc_train.npy', lambdaacc)
np.save('taus_train.npy', all_taus)
np.save('tau_acc_train.npy', tauacc)
np.save('betas_train.npy', all_betas)