import numpy as np
import ray

@ray.remote
def delta_beta(z, S, B_zeta, Lambda, beta):
    '''
    Gradient w.r.t. beta
    '''
    return (B_zeta.T.dot(z*1/S) - (B_zeta.T).dot(B_zeta).dot(beta) - beta/(Lambda**2))

@ray.remote
def delta_1_log_tau(p, log_tau, Lambda):
    '''
    Gradient w.r.t. log tau
    '''
    tau = np.exp(log_tau)
    tau2 = tau**2
    Lambda2 = Lambda**2
    return(- (p-1) - (2*tau2)/(1+tau2) + 2*np.sum((Lambda2/tau2)/(1+Lambda2/(tau2))))

def generate_S2_S(Lambda, BoB):
    '''
    Generate S and S2
    '''
    n, p = BoB.shape
    W = np.sum(BoB*(Lambda**2), axis = 1)
    S2 = (1/(1 + W))
    S = np.sqrt(S2)

    return(S2, S)

@ray.remote
def delta_1_lambda(Lambda, beta, B_zeta, S2, S, z, tau):
    '''
    Gradient w.r.t lambda
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

def Delta_theta(vartheta_t, B, n, z, p, tBB, betaBt, BoB):
    '''
    Gradient w.r.t theta
    '''
    vartheta_new = vartheta_t.copy()
    beta_t = vartheta_new[0:p].reshape(p,)
    Lambda_t = np.exp(0.5*vartheta_new[p:2*p])
    log_tau_t = vartheta_new[2*p]

    S2, S = generate_S2_S(Lambda_t, BoB)

    ret_id1 = delta_beta.remote(z, S, B, Lambda_t, beta_t)
    ret_id2 = delta_1_lambda.remote(Lambda_t, beta_t, B, S2, S, z, np.exp(log_tau_t))
    ret_id3 = delta_1_log_tau.remote(p, log_tau_t, Lambda_t)
    grad_beta, grad_lambda, grad_tau = ray.get([ret_id1, ret_id2, ret_id3])

    return(np.append(grad_beta, np.append(grad_lambda, grad_tau)))

def log_density(S, B, beta, Lambda, log_tau, z, p):
    '''
    log density up to normalizing constant
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
    term7 = - np.log(1 + tau2) #
    return(term1 + term2 + term3 + term4 + term5 + + term6 + term7 ) 

def Leapfrog(theta, r, epsilon, n, z, p, B, tBB, betaBt, i, L, B_zeta, BoB):
    '''
    Leapfrog iterator
    '''
    # update theta
    theta_tilde = (theta + epsilon*r).reshape(2*p + 1,)
    beta_t = theta_tilde[0:p]
    betaBt_t = beta_t.dot(B_zeta.T)
    
    # compute updated gradient
    Delta_theta_tilde = Delta_theta(theta_tilde, B, n, z, p, tBB, betaBt_t, BoB)
    
    if i != L:
        # update momentum again
        r_tilde = r + (epsilon)*Delta_theta_tilde 
    
    
    return(theta_tilde, r_tilde)