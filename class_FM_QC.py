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
        self.sigma = sigma
        self.sparsity = sparsity
        
        # initialize beta random number generator
        rng_beta = np.random.default_rng(beta_seed)
        
        # initialize interaction expansion transformation
        self.xfm = preprocessing.PolynomialFeatures(
            degree=degree, interaction_only=True, include_bias=True
        )
        _ = self.xfm.fit_transform(np.zeros((1, self.k), dtype="float32"))
        
        # sample ground truth betas
        if heredity:
            # assume interactions only exist if the main effects are present
            self.beta = rng_beta.normal(0, 1, self.xfm.n_output_features_).astype(
            "float32"
            )
            mask = np.ones(self.k, dtype='float32')

            zero_indices = rng_beta.choice(
                self.k,
                size=int(self.k * sparsity),
                replace=False,
            )
            mask[zero_indices] = 0.0
            mask = self.xfm.transform(mask)
            self.beta *= mask

        else:
            self.beta = rng_beta.normal(0, 1, self.xfm.n_output_features_).astype(
                "float32"
            )
            zero_indices = rng_beta.choice(
                self.xfm.n_output_features_,
                size=int(self.xfm.n_output_features_ * sparsity),
                replace=False,
            )
            self.beta[zero_indices] = 0.0


    def sample(self, seed=0):
        rng = np.random.default_rng(seed)
        # sample treatment array
        t = rng.binomial(1, self.p_t, (self.n, self.k)).astype("float32")
        # expand treatment array
        T = self.xfm.fit_transform(t)
        # build response surface
        self.mu = T @ self.beta
        # sample outcome
        self.eps = rng.normal(0, self.sigma, size=self.n)
        y = self.mu + self.eps
        return t, y
    



