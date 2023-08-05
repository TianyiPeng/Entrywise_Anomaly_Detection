import numpy as np 
import pandas as pd


class DataInput():

    def __init__(self):
        pass

    def synthetic_data(self, n1=1, n2=1, r=1, mean_value=1, prob_observing=1):

        self.n1 = n1
        self.n2 = n2
        self.r = r
        self.mean_value = mean_value
        self.prob_observing = prob_observing
        U = np.random.gamma(1, 2, (n1, r))
        V = np.random.gamma(1, 2, (n2, r))
        M0 = U.dot(V.T)
        self.M0 = M0 / np.mean(M0) * mean_value
        self.Ω = np.random.binomial(1, prob_observing, (n1,n2))
        self.data_type = 'synthetic'

    def real_data_sales_tensor(self):

        filtered = pd.read_csv('data/geo_prod_week.csv')

        weeks = np.sort(filtered.retail_week_id.unique())
        ts = []
        for week in weeks:
            df = filtered.loc[filtered.retail_week_id == week]
            table = df.pivot_table(values = 'unit_sales', index = 'geography_id', columns = 'product_id').to_numpy()
            if (table.shape[0] >= 28 and table.shape[1] == 300):
                table[np.isnan(table)] = 0
                ts.append(table)
        return ts

    def real_data_with_X(self, X, B, mask):
        self.B = B
        self.X = X
        self.Ω = mask
        self.n1 = self.X.shape[0]
        self.n2 = self.X.shape[1]
        self.M0 = X
        
        self.mean_value = np.sum(self.X*self.Ω) / np.sum(self.Ω)
        self.prob_observing = np.sum(self.Ω) / (self.n1*self.n2)

    def add_anomaly(self, anomaly_model):
        self.anomaly_set, self.X = anomaly_model.add_anomaly(self.M0)
        self.anomaly_model = anomaly_model
    

    def cost_generation(self, c11 = [0, 0], c01 = [0, 1], c10 = [0, 1], c00 = [0, 0]):
        self.cost11 = np.random.uniform(low=c11[0], high=c11[1], size=(self.n1, self.n2))

        self.cost01 = np.random.uniform(low=c01[0], high=c01[1], size=(self.n1, self.n2))

        self.cost10 = np.random.uniform(low=c10[0], high=c10[1], size=(self.n1, self.n2))

        self.cost00 = np.random.uniform(low=c00[0], high=c00[1], size=(self.n1, self.n2))

    def compute_posterior(self):
        self.true_posterior = 1 - self.anomaly_model.posterior_anomaly_processing(self.M0, self.X, self.Ω)
        self.b = self.cost01 * (1-self.true_posterior) + self.cost00 * self.true_posterior
        self.a = self.cost11 - self.cost01 + (self.cost10 + self.cost01 - self.cost11 - self.cost00)*self.true_posterior