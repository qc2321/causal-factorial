import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split


class FactorialModel():
    def __init__(self, n, p_t=0.5, k=2, degree=2, sigma=0.1, sparsity=0.5, seed=None):
        self.n = n
        self.p_t = p_t
        self.k = k
        self.sigma = sigma
        self.sparsity = sparsity
        self.seed = seed
                
        # initialize interaction expansion transformation
        self.pf = preprocessing.PolynomialFeatures(
            degree=degree, interaction_only=True, include_bias=True
        )
        _ = self.pf.fit_transform(np.zeros((1, self.k), dtype="float64"))
        
        # initialize beta
        rng_beta = np.random.default_rng(self.seed)
        self.beta = rng_beta.normal(0, 1, self.pf.n_output_features_).astype("float64")

        # impose sparsity on beta
        zero_indices = rng_beta.choice(
            self.pf.n_output_features_,
            size=int(self.pf.n_output_features_ * sparsity),
            replace=False,
        )
        self.beta[zero_indices] = 0.0

        # normalize beta
        norm = np.linalg.norm(self.beta)
        self.beta = self.beta / norm


    def sample_and_split_data(self, contrast_coding=False, test_size=0.2):
        # sample treatment array
        rng = np.random.default_rng(self.seed)
        t = rng.binomial(1, self.p_t, (self.n, self.k)).astype("float64")
        t = t * 2 - 1 if contrast_coding else t

        # expand treatment array
        self.T = self.pf.fit_transform(t)

        # sample outcome with noise
        mu = self.T @ self.beta
        eps = rng.normal(0, self.sigma, size=self.n)
        self.y = mu + eps

        list_of_splits = train_test_split(self.T, self.y, test_size=test_size, random_state=self.seed)
        self.T_train, self.T_test, self.y_train, self.y_test = list_of_splits


    def fit_lasso(self, alphas=None):
        self.lasso = LassoCV(alphas=alphas, max_iter=10000, cv=5)
        self.lasso.fit(self.T_train, self.y_train)
        self.beta_hat = self.lasso.coef_
        beta_mask = self.pf.fit_transform(self.pf.powers_)
        self.expected_outcomes = beta_mask @ self.beta_hat


    def predict(self):
        self.y_pred = self.lasso.predict(self.T_test)
    

    def compute_mse(self):
        self.mse = np.mean((self.y_pred - self.y_test) ** 2)

