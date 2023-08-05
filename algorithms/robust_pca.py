import numpy as np
from algorithms.utility import *

def stable_pcp(data, λ, μ, step_size=1, convergence_threshold=.001, debug=False, up_M = -1, up_A = -1):
    num_iter = 0
    
    X = data.X
    Ω = data.Ω

    M_old = soft_impute(X,Ω,1/μ)
    A_old = l1_prox(X-M_old,Ω,μ/λ/2)
    f = 1e20
    while (True):
        num_iter += 1
        G = μ*(M_old+A_old-X)
        M_new = soft_impute(M_old-step_size*G,Ω,step_size)
        A_new = l1_prox(A_old-step_size*G,Ω,1/2/λ/step_size)
        if (up_M > 0):
            M_new = np.maximum(np.minimum(M_new, up_M), -up_M)
            A_new = np.maximum(np.minimum(A_new, up_A), -up_A)
        if np.max(np.abs(M_old-M_new)) < convergence_threshold and np.max(np.abs(A_old-A_new)) < convergence_threshold:
            break
        else:
            M_old, A_old = M_new, A_new
            if (num_iter % 1000 == 0):
                break
            if (num_iter % 100 == 0):
                f_new = np.linalg.norm(M_new, 'nuc') + λ*np.sum(np.abs(A_new)) + 0.5*μ*np.sum((Ω*(data.X-A_new-M_new))**2)
                if (np.abs(f_new - f) < 1e-3):
                    break
                f = f_new
            
            if (num_iter % 10 == 0):
                if (np.linalg.norm(M_new, 'nuc') + λ*np.sum(np.abs(A_new)) + 0.5*μ*np.sum((Ω*(data.X-A_new-M_new))**2) > 1e20):
                    step_size = step_size * 0.5
                    M_old = soft_impute(X,Ω,1/μ)
                    A_old = l1_prox(X-M_old,Ω,μ/λ/2)
    return M_new,A_new

def Robust_PCA(data, r_constraint = -1, up_M = -1, up_A = -1):
    
    gamma = np.minimum(data.n1, data.n2)
    flag = 0
    while (True):
        #hyper-parameters are optimized manually
        M_est,A_est = stable_pcp(data, 1/gamma*data.mean_value,1/gamma,step_size=50,convergence_threshold=1e-4, debug=False, up_M=up_M, up_A=up_A)
        if (r_constraint < 0):
            break
        if (np.linalg.matrix_rank(M_est) < r_constraint / 2):
            if (flag == -1):
                break
            gamma = gamma / 1.1     
        else: 
            if (np.linalg.matrix_rank(M_est) > r_constraint):
                gamma = gamma + 0.2
                flag = -1
            else:
                break
    return M_est, (A_est != 0)