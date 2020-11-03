import numpy as np
import math
import random

def dlogFCuj(lj, lambdaj, beta, B_zeta, dS2, ddS2, S2, S, z, tau, p, betaBt):
    Lambdaj2 = lambdaj**2
    tau2 = tau**2
    #term1 = 0.5*(beta[lj]**2)/Lambdaj2 - (Lambdaj2/tau2)/(1 +Lambdaj2/tau2) 
    #term2 = - 0.5*np.sum(dS2/S2)
    #term3 = - 0.5*np.sum((z*z*(-dS2/(S2**2))))
    #term4 = np.sum(betaBt*(-0.5*(dS2/(S2**(1.5))))*z)
    #dlogFucj = term1 + term2 + term3 + term4
    return(0.5*(beta[lj]**2)/Lambdaj2 - (Lambdaj2/tau2)/(1 +Lambdaj2/tau2)  - 0.5*np.sum(dS2/S2)
          - 0.5*np.sum((z*z*(-dS2/(S2**2)))) + np.sum(betaBt*(-0.5*(dS2/(S2**(1.5))))*z))

   

def ddlogFCuj(lj, lambdaj, beta, B_zeta, dS2, ddS2, S2, S, z, tau, p, betaBt):

    lambdaj2 = lambdaj**2
    tau2 = tau**2
    # to the power of two or 1? different in code and paper...
    #term1 = -0.5*(beta[lj]**2)/(lambdaj2) - (lambdaj2/(tau2))/((1+ lambdaj2/(tau2))**2)
    #term2 =  - 0.5*np.sum(ddS2/S2 - (dS2**2/S2**2))
    #term3 = -0.5*np.sum((z**2)*(2*(dS2**2)/(S2**3) - ddS2/(S2**2)))
    #term4 = np.sum(betaBt*(0.75*(dS2**2)/(S2**2.5) - 0.5*(ddS2/(S2**1.5)))*z)
    #ddlogFCuj = term1 + term2 + term3 + term4
    return (-0.5*(beta[lj]**2)/(lambdaj2) - (lambdaj2/(tau2))/((1+ lambdaj2/(tau2))**2) - 0.5*np.sum(ddS2/S2 - (dS2**2/S2**2))
           -0.5*np.sum((z**2)*(2*(dS2**2)/(S2**3) - ddS2/(S2**2))) + np.sum(betaBt*(0.75*(dS2**2)/(S2**2.5) - 0.5*(ddS2/(S2**1.5)))*z))

def generate_dS2_ddS2_S2_S_lj(lj, lambda_new, BoB):
    
    n, p = BoB.shape
    W = BoB.dot(lambda_new**2)
    S2 = (1/(1 + W))
    S = np.sqrt(S2)
    
    dS2 = - BoB[:,lj]*(lambda_new[lj]**2)/((1+W)**2)
    ddS2 = - BoB[:,lj]*(lambda_new[lj]**2)/((1+W)**2) + (BoB[:,lj]*(lambda_new[lj]**2)**2)/((1+W)**3)
    
    return(dS2, ddS2, S2, S)


def log_density_lambda(lj, lambda_j, beta, tau, S2, betaBt, z):
    lambda2 = lambda_j**2
    betaj = beta[lj]
    return(-0.5*(betaj*betaj)/lambda2 - math.log(1+ lambda2/tau**2) - 0.5*np.sum(np.log(S2))
           + betaBt.dot(z/np.sqrt(S2)) - 0.5*np.sum(z**2/S2)) - 0.5*math.log(lambda2)

def sample_lambda(lj, log_tau_old, Lambda, p, beta, B_zeta, dS2_old, ddS2_old, S2_old, S_old, z, BoB, betaBt):

    Lambda_new = np.copy(Lambda)
    p = len(Lambda_new)
    log_sq_lambda_old = math.log(Lambda[lj]**2)
    lambdaj_old = Lambda[lj]

    dlu = dlogFCuj(lj, lambdaj_old, beta, B_zeta, dS2_old[:,lj], ddS2_old[:,lj], S2_old, S_old, z, math.exp(log_tau_old), p, betaBt)
    ddlu = ddlogFCuj(lj, lambdaj_old, beta, B_zeta, dS2_old[:,lj], ddS2_old[:,lj], S2_old, S_old, z, math.exp(log_tau_old), p, betaBt)
    
    sigma2u = -1/ddlu
    if sigma2u < np.finfo(float).eps:
        sigma2u = np.finfo(float).eps
        #accept = True

    # sample new lambda
    muu = sigma2u*dlu + log_sq_lambda_old
    unew = np.random.normal(0,1,1)*math.sqrt(sigma2u) + muu
    lambdajnew = math.exp(0.5*unew)

    if lj < p - 1:
        Lambda_new = np.append(np.append(Lambda_new[0:lj], lambdajnew), Lambda_new[lj + 1:])
    elif lj == p - 1:
        Lambda_new = np.append(Lambda_new[0:lj], lambdajnew)
    
    dS2_new_j, ddS2_new_j, S2_new, S_new = generate_dS2_ddS2_S2_S_lj(lj, Lambda_new, BoB)
    dS2_new, ddS2_new = np.copy(dS2_old), np.copy(ddS2_old)
    dS2_new[:,lj], ddS2_new[:,lj] = dS2_new_j, ddS2_new_j
    
    dlunew = dlogFCuj(lj, lambdajnew, beta, B_zeta, dS2_new[:,lj], ddS2_new[:,lj], S2_new, S_new, z, math.exp(log_tau_old), p, betaBt)
    ddlunew = ddlogFCuj(lj, lambdajnew, beta, B_zeta, dS2_new[:,lj], ddS2_new[:,lj], S2_new, S_new, z, math.exp(log_tau_old), p, betaBt)
    
    
    sigma2unew = -1/ddlunew
    if sigma2unew < np.finfo(float).eps:
        sigma2unew = np.finfo(float).eps
    muunew = sigma2unew*dlunew + unew
    #log_sq_lambda_new = np.random.normal(1)*math.sqrt(variance_lambda) + mu_lambda
    #Lambda_new[lj] = math.exp(0.5*log_sq_lambda_new)

    tau_old = math.exp(log_tau_old)
    log_dens_new = log_density_lambda(lj, lambdajnew, beta, tau_old, S2_new, betaBt, z)
    log_dens_old = log_density_lambda(lj, lambdaj_old, beta, tau_old, S2_old, betaBt, z)


    #ddlunew = ddlogFCuj(lj, Lambda_new, beta, B_zeta, dS2_new, ddS2_new, S2_new, S_new, z, math.exp(log_tau_old))
    #variance_lambda_new = -1/ddlunew
    #mu_lambda_new = variance_lambda*dlogFCuj(lj, Lambda_new, beta, B_zeta, dS2_new, ddS2_new, S2_new, S_new, z, math.exp(log_tau_old)) + log_sq_lambda_new


    proposal_new = -0.5*math.log(sigma2u) - 0.5*((unew - muu)**2)/sigma2u
    proposal_old = -0.5*math.log(sigma2unew) - 0.5*((log_sq_lambda_old - muunew)**2)/sigma2unew

    decision_criterion = (log_dens_new - log_dens_old - proposal_new + proposal_old)
    
    return(unew, decision_criterion, dS2_new, ddS2_new, S2_new, S_new, ddlunew, Lambda_new)