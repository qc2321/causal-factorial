import numpy as np
import itertools as it
import statsmodels.api as sm
from scipy.special import comb
from factorial_model import FactorialModel


class ForwardSelection:
    def __init__(self, T, y, max_order, alpha=0.05, strong_heredity=True):
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

    
    def forward_selection(self):
        self.selected = np.zeros(self.num_coeffs, dtype=int)
        self.selected[0] = 1
        parent_idx = 0
        for d in range(1, self.D + 1):
            child_idx = parent_idx + int(comb(self.k, d - 1))
            self.include_d_order_terms(d, child_idx)
            self.impose_heredity(d, parent_idx, child_idx)
            # compute p values and drop non-significant interactions [TODO]
            parent_idx = child_idx

    
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
                for parent in parents:
                    if set(parent).issubset(child):
                        break
                else:
                    self.selected[child_idx + i] = 0
        else:
            for i, child in enumerate(children):
                count = 0
                for parent in parents:
                    if set(parent).issubset(child):
                        count += 1
                    if count == d:
                        break
                else:
                    self.selected[child_idx + i] = 0
