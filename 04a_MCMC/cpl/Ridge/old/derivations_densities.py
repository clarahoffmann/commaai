import numpy as np

def compdS(tau, W):
    tildeW = tau*W
    S2 = 1/(1+tildeW)
    dS2 = -tildeW/((1+tildeW)**2)
    ddS2 = -tildeW/((1+tildeW)**2) + 2*(tildeW**2)/((1+tildeW)**3)
    
    S = np.sqrt(S2)
    return(S2, dS2, ddS2, S)

def dlogFcu(u, B, S2, dS2, z, beta, theta, betaBt):
    p = B.shape[1]
    return( - 0.5*np.sum(dS2/S2) 
            - 0.5*np.sum((z**2)*(-dS2/(S2**2)))
            + np.sum(betaBt*((-0.5*dS2/(S2**1.5))*z)) 
            - (0.5*p - 0.5)
            + 0.5*(beta.dot(beta))/np.exp(u) 
            - 0.5*(np.exp(u)/theta)**0.5)

def ddlogFcu(u, B, S2, dS2, ddS2, z, beta, theta, betaBt):
    return( - 0.5*np.sum(ddS2/S2 - (dS2**2)/(S2**2)) 
           - 0.5*np.sum((z**2)*(2*(dS2**2)/(S2**3)
           - ddS2/(S2**2)))
           + np.sum(betaBt*(0.75*(dS2**2)/(S2**(2.5)) - 0.5*ddS2/(S2**1.5))*z)
           - 0.5*(beta.dot(beta)/np.exp(u))  
           - 0.25*(np.exp(u)/theta)**(0.5))

def logFCu(u,B,S2,z,beta,theta, betaBt):
    p = B.shape[1]
    return(  - 0.5*np.sum(np.log(S2)) 
           - 0.5*np.sum(z**2/S2)
           + np.sum(B.dot(beta)*z/np.sqrt(S2)) 
           - u*(0.5*p - 0.5) 
           - 0.5*(beta.dot(beta))/np.exp(u) 
           - (np.exp(u)/theta)**0.5)

def gen_beta(z,u,B,tBB,S):
    p = tBB.shape[0]
    Sigmabeta = np.linalg.inv(tBB + np.identity(p)/np.exp(u))
    mubeta = Sigmabeta.dot(B.T).dot(S*z)
    #Sigmabetainv = tBB + np.identity(p)/np.exp(u)
    #L = np.linalg.cholesky(Sigmabetainv)
    #v = L.T.dot(1/np.random.normal(0,1,p))
    #eta = B.T.dot(z/np.sqrt(S2))
    #w = L.dot(1/eta)
    #mubeta = L.T.dot(1/w)
    #beta = mubeta + v
    beta = np.random.multivariate_normal(mubeta, Sigmabeta, 1).reshape(B.shape[1],)
    return(beta)

def gen_tau(tau_old,W,B,z,beta,theta, betaBt):
    S2,dS2,ddS2,S = compdS(tau_old,W)
    dlu = dlogFcu(np.log(tau_old), B, S2, dS2, z, beta, theta, betaBt)
    ddlu = ddlogFcu(np.log(tau_old), B, S2, dS2, ddS2, z, beta, theta, betaBt)
    sigma2u = -1/ddlu

    if sigma2u < np.finfo(float).eps:
        decision_criterion = float('-inf')
        return(0, decision_criterion , S2, dS2, ddS2, S)  
    else:
        muu = sigma2u*dlu + np.log(tau_old)
        unew = np.random.normal(0,1,1)*np.sqrt(sigma2u) + muu
        S2_new, dS2_new, ddS2_new, S_new = compdS(np.exp(unew),W)
        dlunew = dlogFcu(unew, B, S2_new, dS2_new, z, beta, theta, betaBt)
        ddlunew = ddlogFcu(unew, B, S2_new, dS2_new, ddS2_new, z, beta, theta, betaBt)
        sigma2unew = -1/ddlunew
        muunew = sigma2unew*dlunew + unew
        fcnew = logFCu(unew, B, S2_new, z, beta, theta, betaBt)
        fcold = logFCu(np.log(tau_old), B, S2, z, beta, theta, betaBt)
        proposalnew = -0.5*np.log(sigma2u) - 0.5*((unew - muu)**2)/sigma2u
        proposalold = -0.5*np.log(sigma2unew) - 0.5*((np.log(tau_old) - muunew)**2)/sigma2unew
        decision_criterion = fcnew - fcold - proposalnew + proposalold
        return(np.exp(unew), decision_criterion , S2_new, dS2_new, ddS2_new, S_new)
