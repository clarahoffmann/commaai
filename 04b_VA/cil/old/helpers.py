import numpy as np
import math
from numpy.linalg import multi_dot

def generate_z(mean_epsilon,var_epsilon ):
    return np.random.multivariate_normal(mean_epsilon,var_epsilon, 1).T

def generate_epsilon(mean_z, var_z):
    return np.random.multivariate_normal(mean_z,var_z, 1).T

def delta_beta(u, B, S2, dS2, z, beta, theta, betaBt, tBB):
    return(z.dot((1/S)*B) - beta.T.dot(tBB) - beta/np.exp(u))

def delta_tau(u, B, S2, dS2, z, beta, theta, betaBt):
    p = B.shape[1]
    return( - 0.5*np.sum(dS2/S2) 
            - 0.5*np.sum((z**2)*(-dS2/(S2**2)))
            + np.sum(betaBt*((-0.5*dS2/(S2**1.5))*z)) 
            - (0.5*p - 0.5)
            + 0.5*(beta.dot(beta))/np.exp(u) 
            - 0.5*(np.exp(u)/theta)**0.5)

def Delta_theta(vartheta_t, B, n, z, p, tBB, betaBt, theta, beta):
    vartheta_new = vartheta_t.copy()
    beta_t = vartheta_new[0:p].reshape(p,)
    u = vartheta_new[2*p]
    
    # Gradient w.r.t. beta
    grad_beta = delta_beta(u, B, S2, dS2, z, beta, theta, betaBt, tBB)
    
    # Gradient w.r.t. tau
    grad_tau = delta_tau(u, B, S2, dS2, z, beta, theta, betaBt)
    
    return(np.append(grad_beta, grad_tau))

def Delta_mu(gradient_h_t, BBD_inv, z_t, d_t, epsilon_t, B_t):
    return gradient_h_t 

def Delta_B(B_zeta,n,z, p, B_t, gradient_h_t, z_t,D_t, d_t, epsilon_t, BBD_inv):
    gradient_h = (gradient_h_t.reshape(2*p+1,1)).copy()
    return( gradient_h.dot(z_t.T) 
           + BBD_inv.dot(B_t.dot(z_t) 
           + (d_t*epsilon_t)).dot(z_t.T))

def Delta_D(gradient_h_t, epsilon_t, D_t, d_t,p, BBD_inv):
    gradient_h_diag = np.diag((gradient_h_t.reshape(2*p+1))).copy()
    return(
    gradient_h_diag.dot(epsilon_t)) 
    + (multi_dot([BBD_inv, (B.dot(z)  + (d_t*epsilon_t)), epsilon]))
    

def log_density(z, u,  beta, B_zeta, Lambda, p, tau, n, S, S2, tBB, theta):   
    return (
          - 0.5*sum(np.log(S2))
          - 0.5*z.dot((1/S2)*z) + beta.dot(B*(1/S)).dot(z) 
          - 0.5*beta.T.dot(tBB).dot(beta) 
          - 0.5/np.exp(u)*np.sum(beta**2) 
          - 0.5*u*(p-1) 
          - np.sqrt(np.exp(u)/theta))

def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-((x_m.T).dot(np.linalg.inv(covariance)).dot(x_m)) / 2))

