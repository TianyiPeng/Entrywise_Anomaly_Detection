import numpy as np
import scipy

class AnomalyModel:
    def __init__():
        pass
    def add_anomaly():
        pass

class ExponentialAnomaly(AnomalyModel):

    def __init__(self, p, exp_rate, p_range = (1e-5, 0.4), one_over_exp_range = (1e-5, 0.5)):
        self.p = p
        self.exp_rate = exp_rate
        self.p_range = p_range
        self.one_over_exp_range = one_over_exp_range

    def add_anomaly(self, M0):

        anomaly_set = np.random.binomial(1, self.p, M0.shape)
        
        A = anomaly_set * np.random.exponential(scale=1/self.exp_rate, size = M0.shape) + (1-anomaly_set)

        X = np.random.poisson(M0*A)

        return anomaly_set, X

    def counting_match_estimate(self, M, X, Ω, debug=False):
        def f(x):
            p = x[0]
            exp_rate = 1/x[1]
            M_star = M / (1- p+p/exp_rate)

            n0 = np.sum(Ω*(exp_rate / (M_star + exp_rate) * p + (1-p) * np.exp(-M_star))) / np.sum(Ω)
            n1 = np.sum(Ω*(exp_rate*M_star / (M_star + exp_rate)**2 * p + (1-p) * np.exp(-M_star)*M_star)) / np.sum(Ω) + n0
            n2 = np.sum(Ω*(exp_rate*(M_star**2) / ((M_star + exp_rate)**3) * p + (1-p) * np.exp(-M_star)*(M_star**2)/2)) / np.sum(Ω) + n1

            real_n0 = np.sum(Ω*(X==0)) / np.sum(Ω)
            real_n1 = np.sum(Ω*(X==1))/ np.sum(Ω) + real_n0
            real_n2 = np.sum(Ω*(X==2))/ np.sum(Ω) + real_n1
            if (debug):
                print(n0, real_n0, n1, real_n1, n2, real_n2)
            return np.sqrt((n0-real_n0)**2 + (n1-real_n1)**2 + (n2 - real_n2)**2)
        res = scipy.optimize.minimize(f, (self.p, 1/self.exp_rate), bounds = (self.p_range, self.one_over_exp_range))
        print(res)
        print('estimate hyper-parameter is p = {}, real_p is {}, exp_rate = {}, real_exp_rate is {}'.format(res.x[0], self.p, 1/res.x[1], self.exp_rate))
        return res.x[0], 1/res.x[1]

    def solve_MLE(self, x, M, X, Ω):
        p = x[0]
        exp_rate = 1/x[1]
        M_star = M / (1- p+p/exp_rate)
        A = (X+1)*np.log(M_star+exp_rate) - M_star - scipy.special.gammaln(X+1)
        logA = (A<=30)*np.log(p*exp_rate + (1-p)*np.exp(np.minimum(A, 30))) + (A>30)*(A+np.log(1-p))
        return - (np.sum(Ω*logA) + np.sum(Ω*X*np.log(M_star/(M_star+exp_rate))) - np.sum(Ω*np.log(M_star+exp_rate)))

        
    def MLE_estimate(self, M, X, Ω):
        res = scipy.optimize.minimize(lambda x: self.solve_MLE(x, M, X, Ω), (0.1, 0.1), bounds = (self.p_range, self.one_over_exp_range))
        return res.x[0], 1/res.x[1]

    def posterior_compute(self, M, p, exp_rate, X, Ω):
        A = (X+1)*np.log(M+exp_rate) - M - scipy.special.gammaln(X+1)
        #some precision trick, we require p*exp_rate / ((1-p)*1e20 + p*exp_rate) \approx 0
        T1 = (A<=50)*(1-p)*np.exp(np.minimum(A, 50)) + (A>50)*1e30
        return Ω * p*exp_rate / (T1 + p*exp_rate)

    def posterior_anomaly_processing(self, M, X, Ω, p=None, exp_rate=None):
        M = np.maximum(M, 0) + 1e-9

        if (p == None):
            p = self.p
        if (exp_rate == None):
            exp_rate = self.exp_rate

        return self.posterior_compute(M, p, exp_rate, X, Ω)
        

class ExponentialAnomalyRow(ExponentialAnomaly):
    def __init__(self, p, exp_rate, p_range = (1e-5, 0.4), one_over_exp_range = (1e-5, 0.5)):
        self.p = p # p is a vector (n1, 1)
        self.exp_rate = exp_rate # exp_rate is a vector (n1, 1)
        self.p_range = p_range
        self.one_over_exp_range = one_over_exp_range

    def add_anomaly(self, M0):

        assert(M0.shape[0] == self.p.shape[0])

        anomaly_set = np.random.binomial(1, self.p @ np.ones((1, M0.shape[1])), M0.shape)
        
        A = anomaly_set * np.random.exponential(scale = 1/(self.exp_rate @ np.ones((1, M0.shape[1]))), size = M0.shape) + (1-anomaly_set)

        X = np.random.poisson(M0*A)

        return anomaly_set, X
    
    def MLE_estimate(self, M, X, Ω):
        p = np.zeros((M.shape[0], 1))
        exp_rate = np.zeros((M.shape[0], 1))
        for i in range(M.shape[0]):
            res = scipy.optimize.minimize(lambda x: self.solve_MLE(x, M[i:i+1, :], X[i:i+1, :], Ω[i:i+1, :]), (0.1, 0.1), 
                                        bounds = (self.p_range, self.one_over_exp_range))
            p[i] = res.x[0]
            exp_rate[i] = 1 / res.x[1]
        return p, exp_rate
    
    def posterior_anomaly_processing(self, M, X, Ω, p=None, exp_rate=None):
        M = np.maximum(M, 0) + 1e-9

        if (p == None):
            p = self.p
        if (exp_rate == None):
            exp_rate = self.exp_rate

        T = np.zeros_like(M)
        for i in range(M.shape[0]):
            T[i:i+1, :] = self.posterior_compute(M[i:i+1, :], p[i], exp_rate[i], X[i:i+1, :], Ω[i:i+1, :])
        return T