# load packages
import helpers_ridge as hlp
import numpy as np
import numpy as np
import math
from random import random, seed
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy.stats import multivariate_normal
import ray
#ray.init()
#import ray
ray.init()

extracted_coefficients_path = '../../data/commaai/extracted_coefficients/copula_cil/'
B_zeta_path = str(extracted_coefficients_path + 'Bzeta/B_zeta.npy')
beta_path = str(extracted_coefficients_path + 'beta/beta.csv')
z_path = str(extracted_coefficients_path + 'Bzeta/tr_labels.npy')


beta = np.genfromtxt(beta_path, delimiter=',')
# B_zeta is a n x q matrix
B_zeta = np.load(B_zeta_path)
B_zeta = B_zeta.reshape(B_zeta.shape[0], beta.shape[0])
tBB = B_zeta.T.dot(B_zeta)
z = np.load(z_path)

# p is the number of beta coefficients in the last hidden layer
p = B_zeta.shape[1]


# Lambda is a diagonal matrix of dimension p
Lambda = np.diag(np.random.rand(p,))

seed(679305)
tau_start = 0.01

# Set iteration counter to 0
t = 0

theta = 2.5

n = B_zeta.shape[0]

# S(x, theta) is of dimension n x n
W = np.array([B_zeta[i,:].dot(B_zeta[i,:]) for i in range(0, n)])
S = np.sqrt(1/(1 + W*tau_start))
S2 = S**2

# m is number of variational parameters, which is 
# 2p (for each lambda_j and each beta_j)
# plus the variational parameter for the prior on lambda
m = p + 1

# number of factors in the factored covariance representation
k = 3

mu_t, B_t, D_t, d_t = hlp.init_mu_B_d(m, k)

mean_epsilon, mean_z, var_epsilon, var_z = hlp.init_epsilon_z(m, k)

## Adadelta
decay_rate = 0.95
constant = 1e-7
E_g2_t_1, E_delta_x_2_1, E_g2_t_1_mu, E_delta_x_2_1_mu, E_g2_t_1_B, E_delta_x_2_1_B, E_g2_t_1_d, E_delta_x_2_1_d = hlp.init_adad(mu_t, B_t, d_t, m)

lower_bounds = []
all_varthetas = []
d_ts = []
mu_ts = []
d_ts = []
B_ts = []
t = 0


v = 50
iterations = 19000
for i in tqdm(range(iterations)):
    
    # 1. Generate epsilon_t and z_t
    z_t = hlp.generate_z(mean_z,var_z, v)
    epsilon_t = hlp.generate_epsilon(mean_epsilon, var_epsilon, v)
    
    # Compute inverse with Woodbury formula.
    inv = np.diag(1/(np.diag(D_t**2)))
    inv2 = np.linalg.inv(np.identity(k) + B_t.T.dot(inv).dot(B_t))
    BBD_inv = inv - multi_dot([inv, B_t, inv2, B_t.T, inv])
    
    # 2. Get gradients from reparameterization samples
    result = ray.get([hlp.get_gradient.remote(z_t, epsilon_t, mu_t, B_t, d_t, B_zeta, 
                                              BBD_inv, i, v, k, p, z, tBB, theta, W, m) for i in range(0,v)])
    Delta_mu_mean = np.sum(np.array([result[i][0] for i in range(0,v)]), axis = 0)
    Delta_B_mean = np.sum(np.array([result[i][1] for i in range(0,v)]), axis = 0)
    Delta_D_mean = np.sum(np.array([result[i][2] for i in range(0,v)]), axis = 0)
    vartheta_t = np.mean(np.array([result[i][3] for i in range(0,v)]), axis = 0)
    
    # 3. Adadelta Updates
    update_mu, E_g2_t_1_mu, E_delta_x_2_1_mu = hlp.adadelta_change(Delta_mu_mean, E_g2_t_1_mu, E_delta_x_2_1_mu, decay_rate = decay_rate, constant = constant)
    update_B, E_g2_t_1_B, E_delta_x_2_1_B  = hlp.adadelta_change(Delta_B_mean, E_g2_t_1_B, E_delta_x_2_1_B, decay_rate = decay_rate, constant = constant)
    update_d, E_g2_t_1_d, E_delta_x_2_1_d = hlp.adadelta_change(Delta_D_mean, E_g2_t_1_d, E_delta_x_2_1_d, decay_rate = decay_rate, constant = constant)
    
    # Update variables
    mu_t = mu_t + update_mu.reshape(m,1)
    B_t = B_t + update_B
    # set upper triangular elements to 0
    B_t *= np.tri(*B_t.shape)
    d_t = (d_t + update_d)
    D_t = np.diag(d_t.reshape(m,))
    
    # 4. Compute ELBO
    result = ray.get([hlp.get_lb.remote(z_t, epsilon_t, mu_t, B_t, d_t, B_zeta, W, p, n, tBB, theta, i, v, m, k, z) for i in range(0,v)])
    L_lambda_mean = np.mean(np.array([result[i][0] for i in range(0,v)]))
    beta_t = np.mean(np.array([result[i][1] for i in range(0,v)]), axis = 0)
    
    # evidence lower bound
    lower_bounds.append(L_lambda_mean.item())
    all_varthetas.append(vartheta_t)
    mu_ts.append(mu_t)
    d_ts.append(d_t)
    B_ts.append(B_t)
    
    # increase time count
    t = t+1

np.save('../../data/commaai/va/unfiltered_gaussian_resampled/Ridge/lower_bounds_new.npy', lower_bounds)
np.save('../../data/commaai/va/unfiltered_gaussian_resampled/Ridge/vartheta_new.npy', np.array(all_varthetas))
np.save('../../data/commaai/va/unfiltered_gaussian_resampled/Ridge/mu_ts_new.npy', mu_ts)
np.save('../../data/commaai/va/unfiltered_gaussian_resampled/Ridge/d_ts_new.npy', d_ts)
np.save('../../data/commaai/va/unfiltered_gaussian_resampled/Ridge/B_ts_new.npy', B_ts)