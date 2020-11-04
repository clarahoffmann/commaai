import numpy as np
import helpers as hlp
import math
from random import random, seed
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
from tqdm import tqdm


extracted_coefficients_path = '../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/'
B_zeta_path = str(extracted_coefficients_path + 'Bzeta/B_zeta.npy')
beta_path = str(extracted_coefficients_path + 'beta/beta.csv')
z_path = str(extracted_coefficients_path + 'Bzeta/tr_labels.npy')

beta = np.genfromtxt(beta_path, delimiter=',')
# B_zeta is a n x q matrix
B_zeta = np.load(B_zeta_path)
B_zeta = B_zeta.reshape(B_zeta.shape[0], beta.shape[0])
z = np.load(z_path) #[0:B_zeta.shape[0]]
# p is the number of beta coefficients in the last hidden layer
p = B_zeta.shape[1]


# Lambda is a diagonal matrix of dimension p
Lambda = np.diag(np.random.rand(p,))

seed(679305)
tau = random()

# Set iteration counter to 0
t = 0

n = B_zeta.shape[0]
# S(x, theta) is of dimension n x n
S = np.array([(1 + ((B_zeta[i,:].T).dot(Lambda)).dot(B_zeta[i,:]))**(-1/2) for i in range(0,n)])

# m is number of variational parameters, which is 
# 2p (for each lambda_j and each beta_j)
# plus the variational parameter for the prior on lambda
m = 2*p + 1

# number of factors in the factored covariance representation
k = m - 5

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

mean_epsilon = np.repeat(0, m)
mean_z = np.repeat(0, k)

var_epsilon = np.diag(np.repeat(1,m))
var_z = np.diag(np.repeat(1,k))

## Adadelta
decay_rate = 0.95
constant = 1e-7
E_g2_t_1 = 0
E_delta_x_2_1 = 0

E_g2_t_1_mu = np.repeat(0, len(mu_t))
E_delta_x_2_1_mu = np.repeat(0, len(mu_t))
E_g2_t_1_B = np.zeros(B_t.shape)
E_delta_x_2_1_B = np.zeros(B_t.shape)
E_g2_t_1_d = np.repeat(0, len(d_t)).reshape(m,1)
E_delta_x_2_1_d = np.repeat(0, len(d_t)).reshape(m,1)

def adadelta_change(gradient, E_g2_t_1, E_delta_x_2_1, decay_rate = 0.99, constant = 10e-6):
    # expected squared gradient for next iteration
    E_g2_t = decay_rate*E_g2_t_1 + (1 - decay_rate)*(gradient**2)
    # update for parameter
    # should there be a minus or plus here ?????
    delta_x =  (np.sqrt(E_delta_x_2_1 + constant)/np.sqrt(E_g2_t + constant))*gradient
    # expected update for next iteration
    E_delta_x_2 = decay_rate*E_delta_x_2_1 + (1 - decay_rate)*(delta_x**2)
    return(delta_x, E_g2_t, E_delta_x_2 )

lower_bounds = []
all_varthetas = []
t = 0
iterations = 80000
for i in tqdm(range(iterations)):
    
    # 1. Generate epsilon_t and z_t
    z_t = hlp.generate_z(mean_z,var_z)
    epsilon_t = hlp.generate_epsilon(mean_epsilon, var_epsilon)
    
    # 2. Draw from vartheta, what we generate are log values
    # of lambda and tau -> have to transform them back to use them
    vartheta_t = mu_t + B_t.dot(z_t) + (d_t*epsilon_t)
    
    # transform lambda and tau to log values
    # rather do in method to keep updates correct
    #vartheta_t[p:2*p] = np.sqrt(np.exp(vartheta_t[p:2*p]**2))
    #vartheta_t[2*p] = np.exp(abs(vartheta_t[2*p]))
    
    # 3. Compute gradient of beta, lambda_j, and tau
    gradient_h_t = hlp.Delta_theta(vartheta_t, B_zeta, n, z, p)
    
    # Compute inverse with Woodbury formula.
    inv = np.linalg.inv(D_t.dot(D_t))
    inv2 = np.linalg.inv(np.identity(k) + B_t.T.dot(inv).dot(B_t))
    BBD_inv = inv - multi_dot([inv, B_t, inv2, B_t.T, inv])
    
    # Compute gradients for the variational parameters mu, B, D
    Delta_mu = hlp.Delta_mu(gradient_h_t, BBD_inv, z_t, d_t, epsilon_t, B_t)
    Delta_B = hlp.Delta_B(B_zeta,n,z, p, B_t, gradient_h_t, z_t, D_t, d_t, epsilon_t, BBD_inv)
    Delta_D = hlp.Delta_D(gradient_h_t, epsilon_t,D_t, d_t,p, BBD_inv)
    
    # 4. Adadelta Updates
    update_mu, E_g2_t_1_mu, E_delta_x_2_1_mu = adadelta_change(Delta_mu, E_g2_t_1_mu, E_delta_x_2_1_mu, decay_rate = decay_rate, constant = constant)
    update_B, E_g2_t_1_B, E_delta_x_2_1_B  = adadelta_change(Delta_B, E_g2_t_1_B, E_delta_x_2_1_B, decay_rate = decay_rate, constant = constant)
    update_d, E_g2_t_1_d, E_delta_x_2_1_d = adadelta_change(Delta_D, E_g2_t_1_d, E_delta_x_2_1_d, decay_rate = decay_rate, constant = constant)
    
    # Update variables
    '''rho = 0.9
    mu_t = mu_t + rho*Delta_mu.reshape(m,1)
    B_t = B_t + rho*Delta_B
    B_t *= np.tri(*B_t.shape)
    d_t = (d_t + rho*Delta_D)
    D_t = np.diag(d_t.reshape(m,))'''
    mu_t = mu_t + update_mu.reshape(m,1)
    B_t = B_t + update_B
    # set upper triangular elements to 0
    B_t *= np.tri(*B_t.shape)
    d_t = (d_t + update_d)
    D_t = np.diag(d_t.reshape(m,))
    
    vartheta_t = mu_t + B_t.dot(z_t) + (d_t*epsilon_t)
    vartheta_t_transf = vartheta_t.copy()
    # 5. compute stopping criterion
    beta_t = vartheta_t_transf[0:p].reshape(p,)
    Lambda_t = np.diag(np.sqrt(np.exp(vartheta_t_transf[p:2*p].reshape(p,))))
    tau_t = np.exp(vartheta_t_transf[2*p])
    
    # Lower bound L(lambda) = E[log(L_lambda - q_lambda]
    log_h_t = hlp.log_density(z, beta_t, B_zeta, Lambda_t, p, abs(tau_t),n)
    log_q_lambda_t = np.log(hlp.multivariate_normal(vartheta_t, m, mu_t, (B_t.dot(B_t.T) + D_t**2)))
    # evidence lower bound
    L_lambda = log_h_t - log_q_lambda_t
    lower_bounds.append(L_lambda.item())
    all_varthetas.append(L_lambda.item())
    
    # increase time count
    t = t+1
    
    # can also set lambda as the value over the last 10 steps
    
np.savetxt('lower_lambda_va.csv', lower_bounds, delimiter=",")
np.savetxt('vartheta_final.csv', vartheta_t, delimiter=",")

last_10_percent = iterations*0.01
vartheta_hat = mean(all_varthetas[last_10_percent:])

np.savetxt('vartheta_hat.csv', vartheta_hat, delimiter=",")