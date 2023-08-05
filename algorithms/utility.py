import numpy as np

def noise_to_signal(X, M, Ω):
    '''
        Suppose M is the underlying matrix, X is the observed matrix. Noise to signal ratio is denoted by
        $$
            \sqrt{\frac{\|P_{\Omega}(X-M)\|_{F}^2}{\|P_{\Omega}(M)\|_{F}^2}} 
        $$
    '''
    return np.sqrt(np.sum((Ω*X - Ω*M)**2) / np.sum((Ω*M)**2))

def abs_mean(X, M, Ω):
    return np.sum(np.abs((X-M)*Ω)) / np.sum(Ω)

def error_Frobenious(Mhat, M):
    return np.sqrt(np.sum((Mhat - M)**2))

def error_max(Mhat, M):
    return np.max(np.abs(Mhat - M))

def count_AUC(FPR, TPR):
    return np.sum((TPR[1:] + TPR[:-1])*(FPR[1:]-FPR[:-1])/2) #trianglur method

def svd_fast(M):
    is_swap = False
    if M.shape[0] > M.shape[1]:
        is_swap = True
        M = M.T

    A = M @ M.T
    u, ss, uh = np.linalg.svd(A, full_matrices=False)
    s = np.sqrt(ss)
    sinv = 1.0 / (s + 1e-9)
    vh = sinv.reshape(M.shape[0], 1) * (uh @ M)

    if is_swap:
        return vh.T, s, u.T
    else:
        return u, s, vh

## least-squares solved via single SVD
def SVD(M, r): #input matrix M, approximating with rank r
    u, s, vh = svd_fast(M)
    s[r:] = 0
    return (u * s).dot(vh)
    
def l1_prox(X,Ω,γ):
    return Ω * np.sign(X) * np.maximum(0,np.abs(X)-.5/γ)

def soft_threshold(X,γ):
    U,s,VT = svd_fast(X)
    S_threshold = np.diag(np.maximum(0,s-γ))
    return U.dot(S_threshold).dot(VT)

def soft_impute(X,Ω,γ,convergence_threshold=.001, debug=False):
    M_old = np.zeros(X.shape)
    M_new = soft_threshold(Ω*X + (1-Ω)*M_old,γ)
    while  np.sum((M_old-M_new)**2) >= convergence_threshold * np.sum(M_old**2):
        M_old = M_new
        M_new = soft_threshold(Ω*X + (1-Ω)*M_old,γ)
        if (np.sum(M_old**2)<1e-6):
            break
        if (debug):
            print(np.linalg.matrix_rank(M_new))
    return M_new

def hard_impute(O, Ω, r=1):
    M = np.zeros_like(O)
    for T in range(2000):
        M_new = SVD(O * Ω + (1-Ω) * M , r)
        #print(np.linalg.norm(M-M_new) / np.linalg.norm(M))
        if (np.linalg.norm(M-M_new) < np.linalg.norm(M)*1e-3):
            break
        M = M_new
    return M

def cost_compute(data, anomaly_decision):
    if data.data_type == "synthetic":
        # we have an anomaly model in this case  
        mean_cost = np.sum(data.Ω * (data.a * anomaly_decision + data.b)) / np.sum(data.Ω)
        return mean_cost
    else:
        raise Exception() 

def ideal_cost(data):
    return cost_compute(data, data.a <= 0)
