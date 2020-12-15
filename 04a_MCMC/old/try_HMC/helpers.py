import numpy as np
import math
from numpy.linalg import multi_dot

def generate_z(mean_epsilon,var_epsilon ):
    return np.random.multivariate_normal(mean_epsilon,var_epsilon, 1).T

def generate_epsilon(mean_z, var_z):
    return np.random.multivariate_normal(mean_z,var_z, 1).T

# gradient w.r.t. lambda_j
def grad_theta_h_lambda_j(Lambda, S, beta, B_zeta, z, tau,j):
    s_diag = np.diag(S)
    psi_j = B_zeta[:,j]
    lambda_j_sq = (np.diag(Lambda)[j])**2
    A = np.diag((B_zeta[:,j]**2)*lambda_j_sq)
    return(- (1/2)*lambda_j_sq*sum((psi_j**2)*(z**2))
           + (1/2)*lambda_j_sq*sum((psi_j**2)*(s_diag**2))
           + (1/2)*multi_dot([beta.T, B_zeta.T, A, S, z])
           + (beta[j]**2)/(2*lambda_j_sq)
           - (lambda_j_sq/(tau**2))/(1+lambda_j_sq/(tau**2)) + 1/2)

# gradient w.r.t. beta
def delta_theta_h_beta(z, S, B_zeta, Lambda, beta):
    return (z.T).dot(np.linalg.inv(S)).dot(B_zeta) - 2*(beta.T).dot(B_zeta.T).dot(B_zeta) + ((np.linalg.inv(Lambda**2))).dot(beta)

# gradient w.r.t. tau
def delta_tau(p, Lambda, tau):
    tau_sq = tau**2
    return -(p-1) + 2*sum([((Lambda[i,i])**2/tau_sq)*((1+((Lambda[i,i])**2/tau_sq))**(-1)) for i in range(0,Lambda.shape[0])]) - 2*(tau_sq/(1+tau_sq)) 

def Delta_theta(vartheta_t, B_zeta, n, z, p):
    vartheta_new = vartheta_t.copy()
    beta_t = vartheta_new[0:p].reshape(p,)
    lambda_js_t = np.exp(vartheta_new[p:2*p])
    tau_t = np.exp(vartheta_new[2*p])
    
    # Update values that are dependent on t
    # update lambda with lambda_js
    Lambda_t = np.diag(lambda_js_t)

    # update S for lambda values
    S_t = np.diag([(1 + ((B_zeta[i,:].T).dot(Lambda_t**2)).dot(B_zeta[i,:]))**(-1/2) for i in range(0,n)])
    
    # Gradient w.r.t. beta
    grad_beta = delta_theta_h_beta(z, S_t, B_zeta, Lambda_t, beta_t)
    
    # Gradient w.r.t. lambda_js
    grad_lambda_js = [(grad_theta_h_lambda_j(Lambda_t, S_t, beta_t, B_zeta, z, tau_t,j_i)) 
                      for j_i in range(0,Lambda_t.shape[0])]
    grad_lambda_js = np.array(grad_lambda_js).T
    
    # Gradient w.r.t. tau
    grad_tau = delta_tau(p, Lambda_t, tau_t)
    
    return(np.append(grad_beta, np.append(grad_lambda_js, grad_tau)))

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
    

def log_density(z, beta, B_zeta, Lambda, p, tau, n):   
    S =  np.diag([(1 + ((B_zeta[i,:].T).dot(Lambda**2)).dot(B_zeta[i,:]))**(-1/2) for i in range(0,n)])
    return (
          - (1/2)*sum(np.log(np.diag(S**2))) - multi_dot([(1/2)*(z - multi_dot([S, B_zeta, beta])).T, np.linalg.inv(S.dot(S)), (z - multi_dot([S, B_zeta, beta]))])
        +(1/2)*sum(np.log(np.diag(Lambda**2))) - (1/2)*sum([beta[i]**2/(Lambda[i,i]**2) for i in range(0,p)])
        -(p-1)*math.log(tau) - sum(np.log([1+(Lambda[i,i]**2)/(tau**2) for i in range(0,Lambda.shape[0])])) - (p-1)*math.log(1+tau**2) 
    )

def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-((x_m.T).dot(np.linalg.inv(covariance)).dot(x_m)) / 2))

