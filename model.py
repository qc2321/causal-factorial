import numpy as np
import matplotlib.pyplot as plt


"""
Generate a treatment matrix, represented by an n x k matrix with random Bernoulli
entries parameterized by p.
"""
def generate_treatment_matrix(n, k, p):
    return np.random.binomial(1, p, size=(n, k))


"""
Generate an outcome vector, represented by an n-vector with random zero-mean
Gaussian entries.
"""
def generate_outcome_vector(n):
    return np.random.normal(0, 1, size=n)


"""
Fit a factorial model to the data, where the outcome vector is regressed on the
treatment matrix and the interaction terms. The interaction terms can include
up to k treatments concurrently. The function returns the estimated coefficients
and the residuals.
"""
def fit_factorial_model(treatment_matrix, outcome_vector):
    n, k = treatment_matrix.shape
    interaction_matrix = np.zeros((n, 2**k))
    
    # Add bias term to interaction matrix
    interaction_matrix[:, 0] = 1
    
    # Generate interaction terms with permutations of the treatment matrix
    for i in range(1, 2**k):
        interaction_matrix[:, i] = np.prod(treatment_matrix[:, 
                np.where(np.flip(np.array(list(bin(i)[2:])) == 1))[0]], axis=1)

    # Regress outcome vector on interaction matrix  
    return np.linalg.lstsq(interaction_matrix, outcome_vector, rcond=None)


if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate data
    n = 1000
    k = 5
    p = 0.5
    treatment_matrix = generate_treatment_matrix(n, k, p)
    outcome_vector = generate_outcome_vector(n)
    
    # Fit factorial model
    model_params = fit_factorial_model(treatment_matrix, outcome_vector)
    coefficients = model_params[0]
    residuals = model_params[1]

    print("Coefficients:", coefficients)
    print("Mean squared error:", residuals[0] if residuals.size > 0 else residuals)
