import numpy as np
import math
import random
from scipy.stats import multivariate_normal, norm

def delta_1_log_tau(p, log_tau, Lambda):
    tau = math.exp(log_tau)
    tau2 = tau**2
    Lambda2 = Lambda**2
    return(- (p-1) - 2*(tau2/(1+tau2)) + 2*np.sum((Lambda2/tau2)*(1/(1+Lambda2/(tau2)))))
    
def delta_2_log_tau(log_tau, Lambda):
    tau = math.exp(log_tau)
    Lambda2 = Lambda**2
    tau2 = tau**2
    return(4*(tau2**2)/((1+tau2)**2) - 4*(tau2)/(1+tau2) + 4*np.sum(((Lambda2/tau**2)**2)/((1+Lambda2/(tau2))**(2))) - 4*np.sum(((Lambda2)/(tau2))/(1+ (Lambda2)/tau2)))

'''def initialize_tau(Lambda, p):
    # random tau as starting value
    tau = random.random()**2
    log_tau_old = random.random()**2
    variance_tau = - 1/delta_2_log_tau(log_tau_old, Lambda)
    mu_tau = variance_tau*delta_1_log_tau(p, log_tau_old, Lambda) + log_tau_old
    pdf_tau_old = multivariate_normal.pdf(log_tau_old, mean = mu_tau, cov = variance_tau)
    return(tau, log_tau_old, pdf_tau_old)'''

def sample_beta(p, B_zeta, Lambda, S, z):
    # update mean and variance of beta distribution
    variance_beta = np.linalg.inv(B_zeta.T.dot(B_zeta) + np.diag(1/(Lambda**2)))
    mean_beta = variance_beta.dot((B_zeta.T)*(1/S)).dot(z)
    
    # draw one beta from its distribution
    beta = np.random.multivariate_normal(mean_beta, variance_beta, 1).reshape(p,)
    return(beta)

def log_density_tau(log_tau, Lambda, p):
    tau2 = math.exp(log_tau)**2
    log_density = -log_tau*(p-1) - np.sum(np.log(1+Lambda**2/tau2)) - math.log(1 + tau2)
    return(log_density)
    
def sample_tau(log_tau_old, Lambda, p):
    # update variance and mean of lambda and tau distributions
    variance_tau = - 1/delta_2_log_tau(log_tau_old, Lambda)
    mu_tau = variance_tau*delta_1_log_tau(p, log_tau_old, Lambda) + log_tau_old
    
    # draw new tau
    log_tau_new = np.random.normal(0,1,1)*math.sqrt(variance_tau) + mu_tau
    
    # new sample has this distribution
    variance_tau_new = - 1/delta_2_log_tau(log_tau_new, Lambda)
    mu_tau_new = variance_tau*delta_1_log_tau(p, log_tau_new, Lambda) + log_tau_new
    
    # log density new tau
    log_dens_new = log_density_tau(log_tau_new, Lambda, p)
    # log density old tau
    log_dens_old = log_density_tau(log_tau_old, Lambda, p)
    
    # log density of proposal density
    proposalnew = -0.5*math.log(variance_tau) - 0.5*((log_tau_new - mu_tau)**2)/variance_tau
    # log density of old density
    proposal_old = -0.5*math.log(variance_tau_new) - 0.5*((log_tau_old - mu_tau_new)**2)/variance_tau_new
    
    decision_criterion = (log_dens_new - log_dens_old - proposalnew + proposal_old)
    return(log_tau_new, decision_criterion)