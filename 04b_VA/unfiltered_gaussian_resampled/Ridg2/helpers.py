import numpy as np
import math
from numpy.linalg import multi_dot

def generate_z(mean_epsilon,var_epsilon ):
    return np.random.multivariate_normal(mean_epsilon,var_epsilon, 1).T

def generate_epsilon(mean_z, var_z):
    return np.random.multivariate_normal(mean_z,var_z, 1).T

def delta_beta(z, u, B, S, beta, tBB):
    return((z*(1/S)).dot(B) - beta.T.dot(tBB) - beta/np.exp(u))

def delta_tau(u, B, S2, dS2, z, beta, theta, betaBt):
    p = B.shape[1]
    return( - 0.5*np.sum(dS2/S2) 
            - 0.5*np.sum((z**2)*(-dS2/(S2**2)))
            + np.sum(betaBt*((-0.5*dS2/(S2**1.5))*z)) 
            - (0.5*p - 0.5)
            + 0.5*(beta.T.dot(beta))/np.exp(u) 
            - 0.5*(np.exp(u)/theta)**0.5)

def compdS(tau2, W):
    tildeW = tau2*W
    S2 = 1/(1+tildeW)
    dS2 = -tildeW/((1+tildeW)**2)
    ddS2 = -tildeW/((1+tildeW)**2) + 2*(tildeW**2)/((1+tildeW)**3)
    
    S = np.sqrt(S2)
    return(S2, dS2, ddS2, S)

def Delta_theta(vartheta_t, B, n, z, p, tBB, betaBt, theta, W):
    vartheta_new = vartheta_t.copy()
    beta_t = vartheta_new[0:p].reshape(p,)
    u = vartheta_new[p]
    
    S2, dS2, ddS2, S = compdS(np.exp(u), W)
    
    # Gradient w.r.t. beta
    grad_beta = delta_beta(z, u, B, S, beta_t, tBB)
    
    # Gradient w.r.t. tau
    grad_tau = delta_tau(u, B, S2, dS2, z, beta_t, theta, betaBt).item()
    
    return(np.append(grad_beta, grad_tau))

def Delta_mu(gradient_h_t, BBD_inv, z_t, d_t, epsilon_t, B_t):
    return gradient_h_t 

def Delta_B(B_zeta,n,z, p, B_t, gradient_h_t, z_t,D_t, d_t, epsilon_t, BBD_inv):
    gradient_h = (gradient_h_t.reshape(p+1,1)).copy()
    return( gradient_h.dot(z_t.T) 
           + BBD_inv.dot(B_t.dot(z_t) 
           + (d_t*epsilon_t)).dot(z_t.T))

def Delta_D(gradient_h_t, epsilon_t, D_t, d_t,p, BBD_inv):
    gradient_h_diag = np.diag((gradient_h_t.reshape(p+1))).copy()
    return(
    gradient_h_diag.dot(epsilon_t)) 
    + (multi_dot([BBD_inv, (B.dot(z)  + (d_t*epsilon_t)), epsilon]))
    

def log_density(z, u,  beta, B, p,  n, S, S2, tBB, theta, betaBt):   
    return (
          - 0.5*sum(np.log(S2))
          - 0.5*z.dot((1/S2)*z) + betaBt.dot(z*(1/S)) 
          - 0.5*beta.T.dot(tBB).dot(beta) 
          - 0.5/np.exp(u)*np.sum(beta**2) 
          - 0.5*u*(p-1) 
          - np.sqrt(np.exp(u)/theta))


def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d*np.linalg.det(covariance))) * 
            np.exp(-((x_m.T).dot(np.linalg.inv(covariance)).dot(x_m)) / 2))

