# load packages
import numpy as np
import math
import ray
from random import random
from scipy.stats import multivariate_normal
from random import random, seed
seed(679305)

def generate_epsilon(mean_epsilon, var_epsilon, m):
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
    return np.random.multivariate_normal(mean_epsilon, var_epsilon, m)

def generate_z(mean_z, var_z, m):
    ''' Generate new reparameterization variable epsilon:
    Input:  
        - mean_z: mean of epsilon
        - var_z: variance of epsilon
        - m: how many samples
    Output:
        - m samples from normal distribution with mean mean_z
          and variance var_z
        '''
    return np.random.multivariate_normal(mean_z,var_z, m)

def delta_1_lambda(Lambda, beta, B_zeta, S2, S, z, tau):
    '''
    Derivative w.r.t. log(lambda^2)
    Input:
        - Lambda: local shrinkage parameters
        - beta: coeff of last hidden layer
        - Bzeta: basis functions
        - S2: S(x, theta)^2
        - S: S(x, theta)
        - z: transformed labels
        - tau: global shrinkage parameter
    
    Output: 
        - dlogFucj: first order derivative of h(theta) wrt lambda_js
    '''
    p = len(Lambda)
    Lambda2 = Lambda**2
    tau2 = tau**2
    dlogFucj = np.zeros(p)
    for lj in range(0, len(Lambda)):             
        dlogFucj[lj] = (0.5*(beta[lj]**2)/Lambda2[lj]
                         - (Lambda2[lj]/tau2)/(1 + Lambda2[lj]/tau2)
                         + 0.5
                         + 0.5*np.sum((B_zeta[:,lj]**2)*Lambda2[lj]*S2)
                         - 0.5*np.sum((z**2)*(B_zeta[:,lj]**2)*Lambda2[lj])
                        + 0.5*(beta.T.dot(B_zeta.T)*S*(B_zeta[:,lj]**2)*(Lambda2[lj])).dot(z))
    return(dlogFucj)

def generate_S2_S(Lambda, BoB):
    
    n, p = BoB.shape
    W = np.sum(BoB*(Lambda**2), axis = 1)
    S2 = (1/(1 + W))
    S = np.sqrt(S2)
    Lambda2 = Lambda**2

    return(S2, S)


def delta_beta(z, S, B_zeta, Lambda, beta):
    '''
    Gradient of h(theta) w.r.t beta
    '''
    return (B_zeta.T.dot(z*1/S) - (B_zeta.T).dot(B_zeta).dot(beta) - beta/(Lambda**2))

def delta_1_log_tau(p, log_tau, Lambda):
    '''
    Derivative of h(theta) w.r.t log tau
    '''
    tau = np.exp(log_tau)
    tau2 = tau**2
    Lambda2 = Lambda**2
    return(- (p-1) - (2*tau2)/(1+tau2) + 2*np.sum((Lambda2/tau2)/(1+Lambda2/(tau2))))

def Delta_theta(vartheta_t, B, z, p, tBB, betaBt, BoB):
    '''
    Gradient w.r.t. to beta, log(lambda^2), log(tau)
    '''
    vartheta_new = vartheta_t.copy()
    beta_t = vartheta_new[0:p].reshape(p,)
    Lambda_t = np.exp(0.5*vartheta_new[p:2*p].reshape(p,))
    log_tau_t = vartheta_new[2*p]

    S2, S = generate_S2_S(Lambda_t, BoB)
    
    # Gradient w.r.t. beta
    grad_beta = delta_beta(z, S, B, Lambda_t, beta_t)
    
    grad_lambda = delta_1_lambda(Lambda_t, beta_t, B, S2, S, z, np.exp(log_tau_t))
    # Gradient w.r.t. tau
    grad_tau = delta_1_log_tau(p, log_tau_t, Lambda_t)
    
    return(np.append(grad_beta, np.append(grad_lambda, grad_tau)))

def log_density(S, B, beta, Lambda, log_tau, z, p):
    '''
    log density with parameters at hand
    '''
    Lambda2 = Lambda**2
    tau2 = np.exp(log_tau)**2
    S2 = S**2
    square_term = (z - (np.array([B[:,i]*(S[i]) for i in range(0, B.shape[1])]).T.dot(beta)))
    term1 = - 0.5*np.sum(np.log(S2))  
    term2 = + 0.5*np.sum(np.log(Lambda2))  
    term3 = - 0.5*((square_term/S2).dot(square_term)) 
    term4 = -0.5*np.sum((beta**2)/(Lambda2)) 
    term5 = -(p-1)*log_tau 
    term6 = - np.sum(np.log(1+Lambda2/tau2))  
    term7 = - np.log(1 + tau2) 
    
    return(term1 + term2 + term3 + term4 + term5 + term6 + term7) 


def Delta_mu_f(gradient_h_t):
    '''
    Gradient of ELBO w.r.t. mu
    '''
    return gradient_h_t 

def Delta_B_f(z, p, B_t, gradient_h_t, z_t, d_t, epsilon_t, BBD_inv):
    '''
    Gradient of ELBO w.r.t. B
    '''
    gradient_h = (gradient_h_t.reshape(2*p+1,1)).copy()
    return( gradient_h.dot(z_t.T) 
           + BBD_inv.dot(B_t.dot(z_t) 
           + (d_t*epsilon_t)).dot(z_t.T))

def Delta_D_f(gradient_h_t, epsilon_t, D_t, d_t,p, BBD_inv, B_t, z_t):
    '''
    Gradient of ELBO w.r.t. D
    '''
    gradient_h_t_r = gradient_h_t.copy().reshape(2*p+1,1)
    return(np.diag(gradient_h_t_r.dot(epsilon_t.T) + BBD_inv.dot(((B_t.dot(z_t) + (d_t*epsilon_t)).dot(epsilon_t.T)))))

def adadelta_change(gradient, E_g2_t_1, E_delta_x_2_1, decay_rate = 0.99, constant = 10e-6):
    '''
    Adadelta upate for the learning rate
      '''
    E_g2_t = decay_rate*E_g2_t_1 + (1 - decay_rate)*(gradient**2)
    delta_x =  (np.sqrt(E_delta_x_2_1 + constant)/np.sqrt(E_g2_t + constant))*gradient
    E_delta_x_2 = decay_rate*E_delta_x_2_1 + (1 - decay_rate)*(delta_x**2)
    
    return(delta_x, E_g2_t, E_delta_x_2 )

# ray remote function for parallelization
@ray.remote
def get_lb(z_t, epsilon_t, mu_t, B_t, d_t, B_zeta, p, z, BoB, i, v, k, m):
    '''
    ELBO estimate based on reparametrizationb samples z_t and epsilon_t
    '''
    z_t_i = z_t[i,:].reshape(k,1)
    epsilon_t_i = epsilon_t[i,:].reshape(21,1)

    vartheta_t = mu_t + B_t.dot(z_t_i) + (d_t*epsilon_t_i)
    vartheta_t_transf = vartheta_t.copy()

    # 5. compute stopping criterion
    beta_t = vartheta_t_transf[0:p].reshape(p,)
    Lambda_t = np.exp(0.5*vartheta_t_transf[p:2*p].reshape(p,))
    log_tau_t = vartheta_t_transf[2*p]
    betaBt_t = beta_t.dot(B_zeta.T) 

    S2, S = generate_S2_S(Lambda_t, BoB)

    D_t = np.diag(d_t.reshape(m,))
    
    # Lower bound L(lambda) = E[log(L_lambda - q_lambda]
    log_h_t = log_density(S, B_zeta, beta_t, Lambda_t, log_tau_t, z, p)
    log_q_lambda_t = np.log(multivariate_normal.pdf(vartheta_t.reshape(21,), mu_t.reshape(21), (B_t.dot(B_t.T) + D_t**2)))

    return((log_h_t - log_q_lambda_t), beta_t)

@ray.remote 
def get_gradient(z_t, epsilon_t, mu_t, B_t, d_t, B_zeta, BBD_inv, tBB, BoB, i, v, k, p, z, m):
    '''
    Gradient estimates of ELBO based on reparametrizationb samples z_t and epsilon_t
    '''
    z_t_i = z_t[i,:].reshape(k,1)
    epsilon_t_i = epsilon_t[i,:].reshape(21,1)

    #2. Draw from vartheta, what we generate are log values
    #of lambda and tau -> have to transform them back to use them
    vartheta_t = mu_t + B_t.dot(z_t_i) + (d_t*epsilon_t_i)

    beta_t = vartheta_t[0:p].reshape(p,)
    betaBt_t = beta_t.dot(B_zeta.T)

    # 3. Compute gradient of beta, lambda_j, and tau
    gradient_h_t = Delta_theta(vartheta_t, B_zeta, z, p, tBB, betaBt_t, BoB)

    D_t = np.diag(d_t.reshape(m,))
    
    # Compute gradients for the variational parameters mu, B, D
    Delta_mu = Delta_mu_f(gradient_h_t)
    Delta_B = Delta_B_f(z, p, B_t, gradient_h_t, z_t_i, d_t, epsilon_t_i, BBD_inv)
    Delta_D = Delta_D_f(gradient_h_t, epsilon_t_i, D_t, d_t, p, BBD_inv, B_t, z_t_i).reshape(21,1)

    return(Delta_mu, Delta_B, Delta_D, vartheta_t)

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