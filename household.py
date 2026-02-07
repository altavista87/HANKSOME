import numpy as np
from scipy.interpolate import interp1d

def util(c, sigma):
    if sigma == 1.0:
        return np.log(c)
    else:
        return (c**(1-sigma))/(1-sigma)

def marg_util(c, sigma):
    return c**(-sigma)

def inv_marg_util(u_prime, sigma):
    return u_prime**(-1.0/sigma)

def solve_household_egm(r, w, params, y_grid, Pi):
    """
    Solve household problem using Endogenous Grid Method.
    
    Args:
        r: Real interest rate
        w: Wage rate
        params: Parameters object
        y_grid: Income states
        Pi: Transition matrix
        
    Returns:
        a_pol: Policy function for assets a'(a, y) [n_a, n_e]
        c_pol: Policy function for consumption c(a, y) [n_a, n_e]
    """
    beta = params.beta
    sigma = params.sigma
    n_a = params.n_a
    n_e = params.n_e
    
    # Asset Grid (exogenous)
    # Use exponential spacing for more points near 0
    a_grid = params.a_min + (params.a_max - params.a_min) * np.linspace(0, 1, n_a)**2
    
    # Initial Guess: consume everything (c = (1+r)a + wy)
    # Or guess consume interest + income
    # c_guess [n_a, n_e]
    c_next = np.zeros((n_a, n_e))
    for j in range(n_e):
        c_next[:, j] = r * a_grid + w * y_grid[j]
        # Ensure positive
        c_next[:, j] = np.maximum(c_next[:, j], 1e-4)

    # Iteration
    tol = 1e-6
    max_iter = 1000
    dist = 1.0
    
    # Pre-compute gross return
    R = 1 + r
    
    for it in range(max_iter):
        # 1. Expected Marginal Utility tomorrow
        # mu_next [n_a, n_e]
        mu_next = marg_util(c_next, sigma)
        
        # E_mu [n_a, n_e] = mu_next * Pi.T (Matrix mult over income states)
        # Expected value over y' for each current a'
        # Pi is [n_e_current, n_e_next]
        # We want E[mu(a', y') | y]
        E_mu = np.dot(mu_next, Pi.T) 
        
        # 2. Euler Equation
        # u'(c) = beta * R * E_mu
        # c_endo = inv_u'(beta * R * E_mu)
        rhs = beta * R * E_mu
        c_endo = inv_marg_util(rhs, sigma) # [n_a, n_e]
        
        # 3. Endogenous Grid for assets
        # Budget: c + a' = R*a + w*y
        # Implied a_current = (c_endo + a_grid - w*y) / R
        # Note: a_grid here represents a' (the choice for tomorrow)
        
        a_pol = np.zeros((n_a, n_e))
        c_pol = np.zeros((n_a, n_e))
        
        for j in range(n_e):
            # For each income state y_j
            # y_current = w * y_grid[j]
            # a_endo = (c_endo[:, j] + a_grid - w * y_grid[j]) / R
            
            # Since a_endo is not defined on fixed a_grid, we interpolate
            # We want to find c(a) on the fixed a_grid
            
            y_curr = y_grid[j]
            a_endo = (c_endo[:, j] + a_grid - w * y_curr) / R
            
            # Handle borrowing constraint
            # If a_grid[0] (min asset choice) implies a_endo > a_min, then for a < a_endo, constraint binds.
            # But typically we interpolate a_endo -> c_endo to get c(fixed_a)
            
            f_interp = interp1d(a_endo, c_endo[:, j], kind='linear', fill_value="extrapolate")
            c_pol[:, j] = f_interp(a_grid)
            
            # Check constraints
            # If constraint binds: a' = a_min.
            # c = R*a + w*y - a_min
            c_constrained = R * a_grid + w * y_curr - params.a_min
            
            # Policy is min of constrained and unconstrained?
            # EGM naturally handles this: if computed a_endo > a_grid, it means to choose that a', you need more assets than you have.
            # Actually, standard EGM interpolation:
            # We have pairs (a_endo, c_endo).
            # For a given fixed 'a', if 'a' < a_endo[0], constraint binds.
            
            # Correct logic:
            # a_endo are the asset levels TODAY that would lead to choosing a_grid (a') TOMORROW unconstrained.
            # If my actual assets 'a' < a_endo[0], I want to save LESS than a_grid[0] (the min).
            # But I can't. So I save a_grid[0] (a_min).
            # So for a < a_endo[0], c = R*a + w*y - a_min.
            
            mask_constrained = a_grid < a_endo[0]
            c_pol[mask_constrained, j] = c_constrained[mask_constrained]
            
            # Update policy for assets
            # a' = R*a + w*y - c
            a_pol[:, j] = R * a_grid + w * y_curr - c_pol[:, j]
            
            # Enforce bounds
            a_pol[:, j] = np.maximum(a_pol[:, j], params.a_min)

        # 4. Check convergence
        dist = np.max(np.abs(c_pol - c_next))
        if dist < tol:
            # print(f"Converged in {it} iterations")
            break
            
        c_next = c_pol
        
    return a_pol, c_pol, a_grid

import income

def compute_distribution(a_pol, Pi, params, n_sim=1000):
    """
    Compute stationary distribution of agents over (a, y).
    Using eigenvector method on histogram approximation or simulation.
    Since we need histograms for Gini, let's use a non-stochastic simulation or histogram iteration.
    Here: Histogram iteration (Young's method).
    """
    n_a = params.n_a
    n_e = params.n_e
    a_grid = params.a_min + (params.a_max - params.a_min) * np.linspace(0, 1, n_a)**2

    # Initialize distribution D [n_a, n_e]
    # Recompute stat_y for safety
    _, _, stat_y = income.get_income_grid(params)
    
    D = np.zeros((n_a, n_e))
    # Initial mass on low assets
    D[0, :] = stat_y 
    
    # Iterate
    max_iter = 1000
    tol = 1e-7
    
    for it in range(max_iter):
        D_new = np.zeros_like(D)
        
        # For each state (i, j) i: asset, j: income
        # Agents move to a' = a_pol[i, j]
        # Find indices in grid
        
        for j in range(n_e): # Current income
            # Where do they go in asset grid?
            a_next_val = a_pol[:, j]
            
            # Find bracket
            # Use searchsorted
            idx_low = np.searchsorted(a_grid, a_next_val, side='right') - 1
            idx_low = np.clip(idx_low, 0, n_a - 2)
            idx_high = idx_low + 1
            
            # Weights for interpolation
            w_high = (a_next_val - a_grid[idx_low]) / (a_grid[idx_high] - a_grid[idx_low])
            w_low = 1.0 - w_high
            
            # Clamp weights (if out of bounds)
            w_high = np.clip(w_high, 0.0, 1.0)
            w_low = 1.0 - w_high
            
            # Transition of income j -> k
            for k in range(n_e):
                prob_y = Pi[j, k]
                
                # Mass moves from (., j) to (idx_low, k) and (idx_high, k)
                # Use add.at to handle duplicate indices
                mass_low = D[:, j] * w_low * prob_y
                mass_high = D[:, j] * w_high * prob_y
                
                np.add.at(D_new, (idx_low, k), mass_low)
                np.add.at(D_new, (idx_high, k), mass_high)
                
        dist = np.max(np.abs(D_new - D))
        D = D_new
        if dist < tol:
            break
            
    return D

