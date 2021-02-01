# load packages
import numpy as np
import math
from numpy.linalg import multi_dot
import ray
from random import random, seed
from scipy.stats import multivariate_normal

seed(679305)

def generate_z(mean_epsilon,var_epsilon, n):
    ''' 
    Generate new reparameterization variable epsilon:
    Input:  
        - mean_epsilon: mean of epsilon
        - var_epsilon: variance of epsilon
        - m: how many samples
    Output:
        - m samples from normal distribution with mean mean_epsilon
          and variance var_epsilon
        '''
    return np.random.multivariate_normal(mean_epsilon,var_epsilon, n)

def generate_epsilon(mean_z, var_z, n):
    ''' Generate new reparameterization variable epsilon:
    Input:  
        - mean_z: mean of epsilon
        - var_z: variance of epsilon
        - m: how many samples
    Output:
        - m samples from normal distribution with mean mean_z
          and variance var_z
        '''
    return np.random.multivariate_normal(mean_z,var_z, n)

def delta_beta(z, u, B, S, beta, tBB):
    '''
    Gradient of h(theta) w.r.t beta
    '''
    return((z*(1/S)).dot(B) - beta.T.dot(tBB) - beta/np.exp(u))

def delta_tau(u, B, S2, dS2, z, beta, theta, betaBt):
    '''
    Derivative of h(theta) w.r.t log tau^2
    '''
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

    S = np.sqrt(S2)
    return(S2, dS2, S)

def Delta_theta(vartheta_t, B, z, p, tBB, betaBt, theta, W):
    '''
    Gradient w.r.t. to beta, log(lambda^2), log(tau)
    '''
    vartheta_new = vartheta_t.copy()
    beta_t = vartheta_new[0:p].reshape(p,)
    u = vartheta_new[p]
    
    S2, dS2, S = compdS(np.exp(u), W)
    
    # Gradient w.r.t. beta
    grad_beta = delta_beta(z, u, B, S, beta_t, tBB)
    
    # Gradient w.r.t. tau
    grad_tau = delta_tau(u, B, S2, dS2, z, beta_t, theta, betaBt).item()
    
    return(np.append(grad_beta, grad_tau))

def Delta_mu_f(gradient_h_t, BBD_inv, z_t, d_t, epsilon_t, B_t):
    '''
    Gradient of ELBO w.r.t. mu
    '''
    return gradient_h_t 

def Delta_B_f(B_zeta, z, p, B_t, gradient_h_t, z_t, D_t, d_t, epsilon_t, BBD_inv):
    '''
    Gradient of ELBO w.r.t. B
    '''
    gradient_h = (gradient_h_t.reshape(p+1,1)).copy()
    return( gradient_h.dot(z_t.T)  + BBD_inv.dot(B_t.dot(z_t)  + (d_t*epsilon_t)).dot(z_t.T))

def Delta_D_f(gradient_h_t, epsilon_t, D_t, d_t,p, BBD_inv, z_t, B_t):
    '''
    Gradient of ELBO w.r.t. D
    '''
    gradient_h_t_r = gradient_h_t.copy().reshape(p+1,1)
    return(np.diag(gradient_h_t_r.dot(epsilon_t.T) + BBD_inv.dot(((B_t.dot(z_t) + (d_t*epsilon_t)).dot(epsilon_t.T)))))
    

def log_density(z, u,  beta, B, p,  n, S, S2, tBB, theta, betaBt):
    '''
    log density with parameters at hand
    '''
    return (
          - 0.5*sum(np.log(S2))
          - 0.5*z.dot((1/S2)*z) + betaBt.dot(z*(1/S)) 
          - 0.5*beta.T.dot(tBB).dot(beta) 
          - 0.5/np.exp(u)*np.sum(beta**2) 
          - 0.5*u*(p-1) 
          - np.sqrt(np.exp(u)/theta))

# ray remote function for parallelization
@ray.remote
def get_gradient(z_t, epsilon_t, mu_t, B_t, d_t, B_zeta, BBD_inv, i, v, k, p, z, tBB, theta, W, m):
    
    z_t_i = z_t[i,:].reshape(k,1)
    epsilon_t_i = epsilon_t[i,:].reshape(11,1)
    vartheta_t = mu_t + B_t.dot(z_t_i) +  (d_t*epsilon_t_i)
    beta_t = vartheta_t[0:p].reshape(p,)
    betaBt_t = beta_t.dot(B_zeta.T)

    # 3. Compute gradient of beta, lambda_j, and tau
    gradient_h_t = Delta_theta(vartheta_t, B_zeta, z, p, tBB, betaBt_t, theta, W)
    
    D_t = np.diag(d_t.reshape(m,))
    
    # Compute gradients for the variational parameters mu, B, D
    Delta_mu = Delta_mu_f(gradient_h_t, BBD_inv, z_t_i, d_t, epsilon_t_i, B_t)
    Delta_B = Delta_B_f(B_zeta, z, p, B_t, gradient_h_t, z_t_i, D_t, d_t, epsilon_t_i, BBD_inv)
    Delta_D = Delta_D_f(gradient_h_t, epsilon_t_i, D_t, d_t, p, BBD_inv, z_t_i, B_t).reshape(11,1)
    return(Delta_mu, Delta_B, Delta_D, vartheta_t)

@ray.remote
def get_lb(z_t, epsilon_t, mu_t, B_t, d_t, B_zeta, W, p, n, tBB, theta, i, v, m, k, z):
    
    z_t_i = z_t[i,:].reshape(k,1)
    epsilon_t_i = epsilon_t[i,:].reshape(11,1)
    vartheta_t = mu_t + B_t.dot(z_t_i) +  (d_t*epsilon_t_i)
    vartheta_t_transf = vartheta_t.copy()

    # 5. compute stopping criterion
    beta_t = vartheta_t_transf[0:p].reshape(p,)
    u_t = vartheta_t_transf[p]
    betaBt_t = beta_t.dot(B_zeta.T) 

    S = np.sqrt(1/(1 + W*np.exp(u_t)))
    S2 = S**2

    D_t = np.diag(d_t.reshape(m,))
    
    # Lower bound L(lambda) = E[log(L_lambda - q_lambda]
    log_h_t = log_density(z, u_t,  beta_t, B_zeta, p, n, S, S2, tBB, theta, betaBt_t)
    log_q_lambda_t = np.log(multivariate_normal.pdf(vartheta_t.reshape(11,), mu_t.reshape(11), (B_t.dot(B_t.T) + D_t**2)))

    return((log_h_t - log_q_lambda_t), beta_t)   

def adadelta_change(gradient, E_g2_t_1, E_delta_x_2_1, decay_rate = 0.99, constant = 10e-6):
    '''
    Adadelta upate for the learning rate
    '''
    E_g2_t = decay_rate*E_g2_t_1 + (1 - decay_rate)*(gradient**2)
    delta_x =  (np.sqrt(E_delta_x_2_1 + constant)/np.sqrt(E_g2_t + constant))*gradient
    E_delta_x_2 = decay_rate*E_delta_x_2_1 + (1 - decay_rate)*(delta_x**2)
    
    return(delta_x, E_g2_t, E_delta_x_2)

def init_adad(mu_t, B_t, d_t, m):
    '''
    Initialize variables for Adadelta
    '''
    E_g2_t_1 = 0
    E_delta_x_2_1 = 0

    E_g2_t_1_mu = np.repeat(0, len(mu_t))
    E_delta_x_2_1_mu = np.repeat(0, len(mu_t))
    E_g2_t_1_B = np.zeros(B_t.shape)
    E_delta_x_2_1_B = np.zeros(B_t.shape)
    E_g2_t_1_d = np.repeat(0, len(d_t)).reshape(m,1)
    E_delta_x_2_1_d = np.repeat(0, len(d_t)).reshape(m,1)
    
    return(E_g2_t_1, E_delta_x_2_1, E_g2_t_1_mu, E_delta_x_2_1_mu, E_g2_t_1_B, E_delta_x_2_1_B, E_g2_t_1_d, E_delta_x_2_1_d)

def init_mu_B_d(m, k):
    '''
    Initialize mu_t, B_t, d_t
    '''
    mu_t = np.array([random() for i in range(0,m)]).reshape(m,1)

    # B is a lower triangle m x k matrix and is the first component of the 
    # covariance matrix
    B_t = np.tril(np.random.rand(m,k))
    while not np.linalg.matrix_rank(B_t) == k:
        B_t = np.tril(np.random.rand(m,k))

    # D is a diagonal matrix of dimension m x m and is the second component of the 
    # covariance matrix
    D_t = np.diag(np.random.rand(m,))
    d_t = np.diag(D_t).reshape(m,1)
    
    return(mu_t, B_t, D_t, d_t)

def init_epsilon_z(m, k):
    ''' 
    Initalize reparameterization sample distribution
    '''
    mean_epsilon = np.repeat(0, m)
    mean_z = np.repeat(0, k)

    var_epsilon = np.diag(np.repeat(1,m))
    var_z = np.diag(np.repeat(1,k))
    
    return(mean_epsilon, mean_z, var_epsilon, var_z)
