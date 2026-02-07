import numpy as np

def rouwenhorst(rho, sigma, n):
    """
    Discretize AR(1) process using Rouwenhorst method.
    y_t = rho * y_{t-1} + epsilon_t, epsilon_t ~ N(0, sigma^2)
    
    Args:
        rho: Persistence
        sigma: Standard deviation of shock
        n: Number of grid points
        
    Returns:
        grid: State values (log income)
        Pi: Transition matrix
    """
    # 1. Grid Construction
    # The asymptotic standard deviation of y
    sigma_y = sigma / np.sqrt(1 - rho**2)
    
    # Max/Min bounds for Rouwenhorst
    psi = np.sqrt(n - 1) * sigma_y
    y_min = -psi
    y_max = psi
    grid = np.linspace(y_min, y_max, n)
    
    # 2. Transition Matrix Construction
    p = (1 + rho) / 2
    q = p
    
    # Base case for n=2
    Pi = np.array([[p, 1-p],
                   [1-q, q]])
    
    # Recursively build for n > 2
    for i in range(2, n):
        len_prev = i
        Pi_old = Pi
        
        # Construct blocks
        # Top-Left block
        z1 = np.zeros((len_prev + 1, len_prev + 1))
        z1[:len_prev, :len_prev] = p * Pi_old
        
        # Top-Right block
        z2 = np.zeros((len_prev + 1, len_prev + 1))
        z2[:len_prev, 1:] = (1 - p) * Pi_old
        
        # Bottom-Left block
        z3 = np.zeros((len_prev + 1, len_prev + 1))
        z3[1:, :len_prev] = (1 - q) * Pi_old
        
        # Bottom-Right block
        z4 = np.zeros((len_prev + 1, len_prev + 1))
        z4[1:, 1:] = q * Pi_old
        
        Pi = z1 + z2 + z3 + z4
        
        # Normalize rows (essential for Rouwenhorst)
        Pi[1:-1, :] /= 2
        
    return grid, Pi

def stationary_distribution(Pi):
    """Find ergodic distribution of Markov chain."""
    n = Pi.shape[0]
    eigenvals, eigenvecs = np.linalg.eig(Pi.T)
    
    # Find eigenvector with eigenvalue 1
    idx = np.argmin(np.abs(eigenvals - 1))
    dist = np.real(eigenvecs[:, idx])
    return dist / np.sum(dist)

def get_income_grid(params):
    """
    Generate income grid (levels) and transition matrix.
    Returns:
        y_grid: Array of income levels (exp(log_y))
        Pi: Transition matrix
        stat_dist: Stationary distribution of income states
    """
    log_y, Pi = rouwenhorst(params.rho_e, params.sigma_e, params.n_e)
    y_grid = np.exp(log_y)
    
    # Normalize mean income to 1 for convenience
    stat_dist = stationary_distribution(Pi)
    mean_y = np.dot(stat_dist, y_grid)
    y_grid = y_grid / mean_y
    
    return y_grid, Pi, stat_dist
