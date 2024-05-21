import numpy as np
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

    
    def forward_selection(self):
        self.selected_indices = [0]
        start_index = 0
        for d in range(1, self.D + 1):
            start_index += int(comb(self.k, d - 1))
            self.include_interaction(d, start_index)
            self.impose_heredity()

    
    def include_interaction(self, d, start_index):
        num_new_indices = int(comb(self.k, d))
        for i in range(num_new_indices):
            self.selected_indices.append(i + start_index)

    
    def impose_heredity(self):
        pass

