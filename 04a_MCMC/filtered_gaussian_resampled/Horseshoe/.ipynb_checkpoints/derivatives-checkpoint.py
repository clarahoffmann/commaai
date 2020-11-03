import numpy as np
import math

def generate_dS2_ddS2_S2_S(Lambda, BoB):
    
    n, p = BoB.shape
    W = np.sum(BoB*(Lambda**2), axis = 1)
    S2 = (1/(1 + W))
    S = np.sqrt(S2)
    
    dS2, ddS2 = np.zeros((n,p)), np.zeros((n,p))
    for lj in range(0, p):
        dS2[:,lj] = - BoB[:,lj]*(Lambda[lj]**2)/((1+W)**2)
        ddS2[:,lj] = (-BoB[:,lj]*(Lambda[lj]**2) + (BoB[:,lj]*(Lambda[lj]**2)**2))/((1+W)**3)
    
    return(dS2, ddS2, S2, S)

def dlogFCuj(lj, lambdaj, beta, B_zeta, dS2, ddS2, S2, S, z, tau, p, betaBt):
    Lambdaj2 = lambdaj**2
    tau2 = tau**2
    return(0.5*(beta[lj]**2)/Lambdaj2 - (Lambdaj2/tau2)/(1 +Lambdaj2/tau2) 
    - 0.5*np.sum(dS2/S2)
    - 0.5*np.sum((z*z*(-dS2/(S2**2))))
    + np.sum(betaBt*(-0.5*(dS2/(S2**(1.5))))*z))


def ddlogFCuj(lj, lambdaj, beta, B_zeta, dS2, ddS2, S2, S, z, tau, p, betaBt):

    lambdaj2 = lambdaj**2
    tau2 = tau**2
    # to the power of two or 1? different in code and paper...
    return(-0.5*(beta[lj]**2)/(lambdaj2) - (lambdaj2/(tau2))/((1+ lambdaj2/(tau2))**2) -
        0.5*np.sum(ddS2/S2 - ((dS2**2)/(S2**2))) -
         0.5*np.sum((z**2)*(2*(dS2**2)/(S2**3) - ddS2/(S2**2))) + 
     np.sum(betaBt*(0.75*(dS2**2)/(S2**2.5) - 0.5*(ddS2/(S2**1.5)))*z))

def delta_1_log_tau(p, log_tau, Lambda):
    tau = math.exp(log_tau)
    tau2 = tau**2
    Lambda2 = Lambda**2
    return(- (p-1) - 2*(tau2)/(1+tau2) + 2*np.sum((Lambda2/tau2)*((1+Lambda2/(tau2))**(-1))))
    
def delta_2_log_tau(log_tau, Lambda):
    tau = math.exp(log_tau)
    Lambda2 = Lambda**2
    tau2 = tau**2
    return(4*(tau2**2)/((1+tau2)**2) - 4*(tau2**2)/(1+tau2) + 4*np.sum(((Lambda2/tau**2)**2)*((1+Lambda2/(tau2))**(-2))) - 4*np.sum(((Lambda2)/(tau2))*(1/(1+ (Lambda2)/tau2))))

def logFCuj(Lambdaj, beta, B_zeta, S2, z, betaj, tau):
    Lambdaj2 = Lambdaj**2
    tau2 = tau**2
    return(-0.5*(betaj**2)/Lambdaj2 - math.log(1+Lambdaj2/tau2) - 0.5*np.sum(np.log(S2)) + beta.dot(B_zeta.T).dot(z/np.sqrt(S2)) - 0.5*np.sum(z**2/S2))