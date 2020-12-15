import numpy as np
import pystan

extracted_coefficients_directory = '../../../../data/commaai/extracted_coefficients/20201021_unrestr_gaussian_resampled/'
B_zeta_path = str(extracted_coefficients_directory + 'Bzeta/B_zeta.npy')
beta_path = str(extracted_coefficients_directory + 'beta/beta.csv')
z_path = str(extracted_coefficients_directory + 'Bzeta/tr_labels.npy')
beta = np.genfromtxt(beta_path, delimiter=',')
B_zeta = np.load(B_zeta_path)
B_zeta = B_zeta.reshape(B_zeta.shape[0], beta.shape[0])
z = np.load(z_path)
tBB = B_zeta.T.dot(B_zeta)
n = B_zeta.shape[0]
p = B_zeta.shape[1]

X = B_zeta

hmc_code = '''


functions {
    vector S_xtheta(vector lambda, matrix X, int p, int n) {
        vector[n] s_is;
        for (N in 1:n) {
          s_is[N] = sqrt(1 + sum((row(X, N).*square(to_row_vector(lambda))).*(row(X, N)))); 
        } 
        return(s_is);
    }
    }
data {
  int<lower=1> n; // Number of data
  int<lower=1> p; // Number of covariates
  matrix[n,p] X;  // n-by-p design matrix
  real y[n];      // n-dimensional response vector
}


parameters {
  vector[p] beta;
  vector<lower=0>[p] lambda;
  real<lower=0> tau;
}

transformed parameters {
  vector[n] theta ;
  vector[n] S ;
  vector[n] Var ;
  S = S_xtheta(lambda, X, p, n);
  theta = S .* (X * beta);
  Var = square(S);
}

model {
  lambda ~ cauchy(0, 1);
  tau ~ cauchy(0, 1);
  beta ~ normal(0, tau * lambda); 
  y ~ normal(theta, Var);
}'''

hmc_dat = {'n': n,
           'p': p,
           'X': X,
           'y': z}
           
sm = pystan.StanModel(model_code = hmc_code, verbose = True)
print('finished compiling')
print('start sampling')
fit = sm.sampling(data = hmc_dat,
                  iter = 1000, 
                  chains = 4, verbose = True)
print('finished sampling')

samples = fit.extract(permuted=True)

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

with open('py_with_sigma.p', 'wb') as fp:
    pickle.dump(samples, fp, protocol=pickle.HIGHEST_PROTOCOL)