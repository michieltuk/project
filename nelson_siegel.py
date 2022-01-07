import numpy as np

from sklearn.linear_model import LinearRegression

class NelsonSiegel:

    def __init__(self, maturities, shape_parameter=1.0):
        self.maturities = maturities
        self.shape_parameter = shape_parameter

        self.x_1 = [np.exp(-x/self.shape_parameter) for x in self.maturities]
        self.x_2 = [x/self.shape_parameter*np.exp(-x/self.shape_parameter) for x in self.maturities]

        self.X = np.array([self.x_1, self.x_2]).T

    def factor_estimates(self, yield_curves):

        self.yhat = np.empty(yield_curves.shape)
        self.factors = np.empty((yield_curves.shape[0], 3))

        for t, yield_curve in enumerate(yield_curves):
            model = LinearRegression().fit(self.X, yield_curve)
            self.factors[t] = np.insert(model.coef_, 0, model.intercept_)

        return self.factors

    def yield_estimates(self, factors):

        X = np.insert(self.X, 0, 1, axis=1)

        yhat = X@factors.T

        return yhat.T

