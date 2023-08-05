from algorithms.utility import *

def EW_algorithm(data, gamma=1.0, r_constraint = 1e9, debug=False, do_SVD=False, using_ideal_parameter=False, hard_impute_yes=True):
        
    if (do_SVD):
        M = SVD(data.Ω*data.X, r_constraint)
    else:
        if (hard_impute_yes):
            M = hard_impute(data.X, data.Ω, r = r_constraint)
        else:
            while (True):
                M = soft_impute(data.X, data.Ω, gamma, convergence_threshold=1e-4, debug=False)
                if (np.linalg.matrix_rank(M) > r_constraint):
                    gamma = gamma *1.1
                    if (debug):
                        print('gamma update with the rank ', np.linalg.matrix_rank(M))
                else:
                    break

        #M = hard_impute(self.X, self.Ω, r = r_constraint)
        
    M = np.maximum(M, 1e-8)
    
    #p_est, exp_est = self.noise.counting_match_estimate(M, self.X, self.Ω, debug)
    p_est, exp_est = data.anomaly_model.MLE_estimate(M, data.X, data.Ω)

    if (not using_ideal_parameter):
        data.posterior_anomaly_est = data.anomaly_model.posterior_anomaly_processing(M / (1 - p_est + p_est/exp_est), data.X, data.Ω)
    else:
        data.posterior_anomaly_est = data.anomaly_model.posterior_anomaly_processing(M / (1 - data.anomaly_model.p + data.anomaly_model.p/data.anomaly_model.exp_rate), data.X, data.Ω)

    t = ((data.cost11 - data.cost01 + (data.cost10 + data.cost01 - data.cost11 - data.cost00)*(1 - data.posterior_anomaly_est)) <= 0)
        
    return M / (1 - p_est + p_est/exp_est), t

def DRMF(data, r, rate_anomalies):
    X = data.X
    Ω = data.Ω
    S = np.zeros_like(X)
    L = np.zeros_like(X)
    mm = np.sum(Ω)
    for t in range(40):
        L = SVD(Ω*(X - S - L) + L, r)
        thres = np.sort((np.abs(X - L)*Ω + (1-Ω)*1e8).reshape(-1))[int(mm*(1-rate_anomalies))]
        S = (X-L)*(np.abs(X-L) >= thres)

    return L, S != 0

