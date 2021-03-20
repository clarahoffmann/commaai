import numpy as np

def delta_beta(z, u, B, S, beta, tBB):
    '''
    Gradient w.r.t. beta
    '''
    return(B.T.dot(z*(1/S)) - beta.T.dot(tBB) - beta/np.exp(u))

def delta_tau(u, B, S2, dS2, z, beta, theta_prior, betaBt):
    '''
    Gradient w.r.t. log tau
    '''
    p = B.shape[1]
    return(- 0.5*np.sum(dS2/S2) 
           - 0.5*np.sum((z**2)*(-dS2/(S2**2)))
           + np.sum(betaBt*((-0.5*dS2/(S2**1.5))*z)) 
           - (0.5*p - 0.5) 
           + 0.5*(beta.T.dot(beta))/np.exp(u) 
           - 0.5*(np.exp(u)/theta_prior)**0.5 )

def generate_dS2_ddS2_S2_S(tau2, W):
    '''
    Generate S and S2
    '''
    tildeW = tau2*W
    S2 = 1/(1+tildeW)
    dS2 = -tildeW/((1+tildeW)**2)
    ddS2 = -tildeW/((1+tildeW)**2) + 2*(tildeW**2)/((1+tildeW)**3)
    
    S = np.sqrt(S2)
    return(S2, dS2, ddS2, S)

def Delta_theta(vartheta_t, B, n, z, p, tBB, betaBt, theta, W, theta_prior):
    '''
    Gradient w.r.t theta
    '''
    vartheta_new = vartheta_t.copy()
    beta_t = vartheta_new[0:p].reshape(p,)
    u = vartheta_new[p]

    S2, dS2, ddS2, S = generate_dS2_ddS2_S2_S(np.exp(u), W)
    
    # Gradient w.r.t. beta
    grad_beta = delta_beta(z, u, B, S, beta_t, tBB)
    # Gradient w.r.t. tau
    grad_tau = delta_tau(u, B, S2, dS2, z, beta_t, theta_prior, betaBt)
    
    return(np.append(grad_beta, grad_tau))

def log_density(z, u,  beta, B, p,  n, S, S2, tBB, theta, betaBt):   
    '''
    log density up to normalizing constant
    '''
    term1 = - 0.5*np.sum(np.log(S2))
    term2 =  - 0.5*z.dot((1/S2)*z)
    term3 = + betaBt.dot(z*(1/S)) 
    term4 = - 0.5*beta.T.dot(tBB).dot(beta)
    term5 =  - 0.5/np.exp(u)*np.sum(beta**2)
    term6 =  - 0.5*u*(p-1)  - np.sqrt(np.exp(u)/theta)
    return (term1 + term2 + term3 +term4 + term5 + term6  )

def Leapfrog(theta, r, epsilon, n, z, p, B, tBB, betaBt, beta, W, theta_prior):
    '''
    Leapfrog iterator
    '''

    # compute gradient with theta
    Delta_theta_t = Delta_theta(theta, B, n, z, p, tBB, betaBt, theta, W, theta_prior)

    # update momentum
    r_tilde = r + (epsilon/2)*Delta_theta_t

    # update theta
    theta_tilde = (theta + epsilon*r_tilde).reshape(p + 1,)
    beta_t = theta_tilde[0:p]
    betaBt_t = beta_t.dot(B.T)
    
    # compute updated gradient
    Delta_theta_tilde = Delta_theta(theta_tilde, B, n, z, p, tBB, betaBt_t, theta, W, theta_prior)
    
    # update momentum again
    r_tilde = r_tilde + (epsilon/2)*Delta_theta_tilde
    
    return(theta_tilde, r_tilde)

