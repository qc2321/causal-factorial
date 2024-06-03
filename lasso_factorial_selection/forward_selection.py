import numpy as np
import itertools as it
import statsmodels.api as sm
from scipy.special import comb
from factorial_model import FactorialModel


class ForwardSelection:
    def __init__(self, T, y, max_order, alpha=0.05, strong_heredity=False):
        self.T = T
        self.y = y
        self.D = max_order
        self.alpha = alpha
        self.strong_heredity = strong_heredity
        self.n = T.shape[0]
        self.num_coeffs = T.shape[1]
        self.k = int(np.log2(self.num_coeffs))
        assert np.log2(self.num_coeffs) % 1 == 0, "Number of coeffs must be a power of 2"
        assert self.D <= self.k, "Maximum order must be less than or equal to number of factors (k)"
        assert np.all(np.isin(np.unique(self.T), [-1, 1])), "Input must be contrast coded"

    
    def forward_selection(self, logistic=False):
        self.selected = np.zeros(self.num_coeffs, dtype=int)
        self.selected[0] = 1
        parent_idx = 0
        for d in range(1, self.D + 1):
            child_idx = parent_idx + int(comb(self.k, d - 1))
            self.include_d_order_terms(d, child_idx)
            self.impose_heredity(d, parent_idx, child_idx)
            self.drop_interactions_by_pvalues(child_idx, logistic)
            parent_idx = child_idx
        if logistic:
            model = sm.Logit(self.y, self.T * self.selected)
            self.results = model.fit(method='bfgs', skip_hessian=True)
        else:
            model = sm.OLS(self.y, self.T * self.selected)
            self.results = model.fit()

    
    def include_d_order_terms(self, d, child_idx):
        num_new_indices = int(comb(self.k, d))
        for i in range(num_new_indices):
            self.selected[i + child_idx] = 1

    
    def impose_heredity(self, d, parent_idx, child_idx):
        parent_mask = self.selected[parent_idx:child_idx]
        parents = np.array(list(it.combinations(np.arange(1, self.k + 1), d - 1)))
        parents = parents[parent_mask.astype(bool)] if d > 1 else parents
        children = np.array(list(it.combinations(np.arange(1, self.k + 1), d)))

        if self.strong_heredity:
            for i, child in enumerate(children):
                count = 0
                for parent in parents:
                    if set(parent).issubset(child):
                        count += 1
                    if count == d:
                        break
                else:
                    self.selected[child_idx + i] = 0
        else:
            for i, child in enumerate(children):
                for parent in parents:
                    if set(parent).issubset(child):
                        break
                else:
                    self.selected[child_idx + i] = 0


    def drop_interactions_by_pvalues(self, child_idx, logistic):
        adjusted_T = self.T[:, self.selected.astype(bool)]
        if logistic:
            model = sm.Logit(self.y, adjusted_T)
            results = model.fit(method='bfgs', skip_hessian=True)
        else:
            model = sm.OLS(self.y, adjusted_T)
            results = model.fit()

        selected_indices = np.where(self.selected)[0]
        mask_indices = np.where(selected_indices >= child_idx)[0]
        mask = results.pvalues[mask_indices] < self.alpha
        self.selected[mask_indices] *= mask


    def predict(self, T_test):
        self.y_pred = self.results.predict(T_test)
    
    def compute_mse(self, y_test):
        self.mse = np.mean((self.y_pred - y_test) ** 2)

    def compute_r2(self):
        self.r2 = self.results.rsquared
    
    