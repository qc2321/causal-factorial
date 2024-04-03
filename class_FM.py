import numpy as np
from sklearn import preprocessing


class FactorialModel(object):
    def __init__(
        self,
        n,
        p_t=0.5,
        k=2,
        degree=2,
        sigma=0.1,
        sparsity=0.5,
        beta_seed=42,
        heredity = False
    ) -> None:
        self.n = n
        self.p_t = p_t
        self.k = k
        self.degree = degree
        self.sigma = sigma
        self.sparsity = sparsity
        self.beta_seed = beta_seed
        self.heredity = heredity
        # initialize beta random number generator
        self.rng_beta = np.random.default_rng(self.beta_seed)
        # initialize interaction expansion transformation
        self.xfm = preprocessing.PolynomialFeatures(
            degree=self.degree, interaction_only=True, include_bias=True
        )
        _ = self.xfm.fit_transform(np.zeros((1, self.k), dtype="float32"))
        # sample ground truth betas
        if heredity:
            self.beta = self.rng_beta.normal(0, 1, self.xfm.n_output_features_).astype(
            "float32"
            )
            self.mask = np.ones(self.beta.shape,dtype='float32')
            
            zero_indices = self.rng_beta.choice(
                self.k,
                size=int(self.k * self.sparsity),
                replace=False,
            )
            self.mask[zero_indices] = 0.0
            self.beta = self.beta * self.mask

        else:
            self.beta = self.rng_beta.normal(0, 1, self.xfm.n_output_features_).astype(
                "float32"
            )
            zero_indices = self.rng_beta.choice(
                self.xfm.n_output_features_,
                size=int(self.xfm.n_output_features_ * self.sparsity),
                replace=False,
            )
            self.beta[zero_indices] = 0.0

    def sample(self, seed=None):
        self.rng = np.random.default_rng(seed)
        # sample treatment array
        t = self.rng.binomial(1, self.p_t, (self.n, self.k)).astype("float32")
        # expand treatment array
        T = self.xfm.fit_transform(t)
        # build response surface
        self.mu = T @ self.beta
        # sample outcome
        self.eps = self.rng.normal(0, self.sigma, size=self.n)
        y = self.mu + self.eps
        return t, y
    



