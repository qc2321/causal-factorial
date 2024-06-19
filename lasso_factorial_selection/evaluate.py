from forward_selection import ForwardSelection


def sample_beta(fm, sampling_method=1, min_degree=2):
    """Sample beta according to the beta_sampling_method"""
    match sampling_method:
        case 1:
            return fm.sample_normal_beta()
        case 2:
            return fm.sample_skewed_beta(min_degree=min_degree)
        case 3:
            return fm.sample_interactive_beta_clusters(fm.k // 2)
        case _:
            raise ValueError(f"Invalid beta_sampling_method: {sampling_method}")
        

def evaluate_lasso(fm, logistic=False):
    """Evaluate Lasso model from FactorialModel instance"""
    fm.fit_lasso(logistic=logistic)
    fm.predict()
    fm.compute_mse()
    if not logistic:
        fm.compute_r2()


def evaluate_forward_selection(fs, T_test, y_test, logistic=False):
    """Evaluate ForwardSelection model from given test data"""
    fs.forward_selection(logistic=logistic)
    fs.predict(T_test)
    fs.compute_mse(y_test)
    if not logistic:
        fs.compute_r2()


def evaluate(fm, num_trials, seed=None, strong_heredity=False):
    """Evaluate Lasso and ForwardSelection models"""
    lasso_mses = []
    fs_mses = []
    lasso_r2s = []
    fs_r2s = []
    lasso_betas = []
    fs_betas = []
    lasso_expected_outcomes = []
    fs_expected_outcomes = []

    for _ in range(num_trials):
        fm.sample_and_split_data(seed=seed)

        evaluate_lasso(fm)
        lasso_mses.append(fm.mse)
        lasso_r2s.append(fm.r2)
        lasso_betas.append(fm.beta_hat)
        lasso_expected_outcomes.append(fm.expected_outcomes)

        fs = ForwardSelection(fm.T_train, fm.y_train, fm.k, fm.degree, strong_heredity=strong_heredity)
        evaluate_forward_selection(fs, fm.T_test, fm.y_test)
        fs_mses.append(fs.mse)
        fs_r2s.append(fs.r2)
        fs_betas.append(fs.results.params)
        beta_mask = fm.pf.fit_transform(fm.pf.powers_)
        fs_expected_outcome = beta_mask @ fs.results.params
        fs_expected_outcomes.append(fs_expected_outcome)

    return lasso_mses, fs_mses, lasso_r2s, fs_r2s, lasso_betas, fs_betas, \
           lasso_expected_outcomes, fs_expected_outcomes

