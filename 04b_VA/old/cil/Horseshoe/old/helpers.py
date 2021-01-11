import numpy as np
import math

def generate_z(mean_epsilon,var_epsilon ):
    return np.random.multivariate_normal(mean_epsilon,var_epsilon, 1).T

def generate_epsilon(mean_z, var_z):
    return np.random.multivariate_normal(mean_z,var_z, 1).T

def delta_1_lambda(Lambda, beta, B_zeta, dS2, ddS2, S2, S, z, tau):
    p = len(Lambda)
    Lambda2 = Lambda**2
    tau2 = tau**2
    dlogFucj = np.zeros(p)
    for lj in range(0,p):
        dlogFucj[lj] = 0.5*(beta[lj]**2)/Lambda2[lj] - (Lambda2[lj]/tau2)/(1 + Lambda2[lj]/tau2) - 0.5*np.sum(dS2[:,lj]/S2) - 0.5*np.sum((z*z*(-dS2[:,lj]/(S2**2)))) + np.sum((beta.dot(B_zeta.T)*(-0.5)*(dS2[:,lj]/(S2**(1.5)))).dot(z))  
    #dlogFucj = (0.5*(beta**2)/Lambda2 - (Lambda2/tau2)/(1 + Lambda2/tau2) - 0.5*np.sum(dS2/(np.tile(S2.T, [p,1]).T), axis = 0) - 
    #           0.5*np.sum((np.tile(z*z, [p,1]).T*(-dS2/np.tile((S2**2), [p,1]).T)), axis = 0) + 
    #           (beta*((B_zeta*(-0.5)*(dS2/np.tile(S2**1.5, [p,1]).T)).T).dot(z)))
    return(dlogFucj)

def generate_dS2_ddS2_S2_S(Lambda, BoB):
    
    n, p = BoB.shape
    W = np.sum(BoB*(Lambda**2), axis = 1)
    S2 = (1/(1 + W))
    S = np.sqrt(S2)
    Lambda2 = Lambda**2
    
    dS2, ddS2 = np.zeros((n,p)), np.zeros((n,p))
    for lj in range(0, p):
        dS2[:,lj] = - BoB[:,lj]*Lambda2[lj]/((1+W)**2)
        ddS2[:,lj] = (-BoB[:,lj]*Lambda2[lj] + (BoB[:,lj]*(Lambda2[lj]**2)))/((1+W)**3)
    
    return(dS2, ddS2, S2, S)


def delta_beta(z, S, B_zeta, Lambda, beta):
    return ((z*(1/S).T).dot(B_zeta) - (beta.T).dot(B_zeta.T).dot(B_zeta) + np.sum(((1/Lambda)**2)*beta))

def delta_1_log_tau(p, log_tau, Lambda):
    tau = np.exp(log_tau)
    tau2 = tau**2
    Lambda2 = Lambda**2
    return(- (p-1) - (2*tau2)/(1+tau2) + 2*np.sum((Lambda2/tau2)/(1+Lambda2/(tau2))))

def Delta_theta(vartheta_t, B, n, z, p, tBB, betaBt, BoB):
    vartheta_new = vartheta_t.copy()
    beta_t = vartheta_new[0:p].reshape(p,)
    Lambda_t = np.exp(0.5*vartheta_new[p:2*p].reshape(p,))
    log_tau_t = vartheta_new[2*p]

    dS2, ddS2, S2, S = generate_dS2_ddS2_S2_S(Lambda_t, BoB)
    
    # Gradient w.r.t. beta
    grad_beta = delta_beta(z, S, B, Lambda_t, beta_t)
    
    grad_lambda = delta_1_lambda(Lambda_t, beta_t, B, dS2, ddS2, S2, S, z, np.exp(log_tau_t))
    # Gradient w.r.t. tau
    grad_tau = delta_1_log_tau(p, log_tau_t, Lambda_t)
    
    return(np.append(grad_beta, np.append(grad_lambda, grad_tau)))

def log_density(S, B, beta, Lambda, log_tau, z, p):
    Lambda2 = Lambda**2
    tau2 = np.exp(log_tau)**2
    S2 = S**2
    term1 = - 0.5*np.sum(np.log(S2))
    term2 = -0.5*np.sum(((z - S*(B.dot(beta)))**2)*(1/S2))
    term3 = + 0.5*np.sum(Lambda2)
    term4 = -0.5*np.sum((beta**2)/(Lambda2))
    term5 = -(p-1)*log_tau
    term6 = - np.sum(np.log(1+Lambda2/tau2))
    term7 = - np.log(1 + tau2)
    return(term1 + term2 + term3 + term4 + term5 + term6 + term7)

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
    
def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-((x_m.T).dot(np.linalg.inv(covariance)).dot(x_m)) / 2))