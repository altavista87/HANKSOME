import numpy as np
from scipy.optimize import brentq
import calibration
import income
import household

def get_mixture_asset_supply(r, params, y_grid, Pi):
    """
    Solve two-type mixture HH problem and return aggregate asset supply.
    """
    # 1. Prices
    # Firms pay the Elite rate (r + premium)
    r_firm = r + params.premium
    K_dem = (params.alpha / (r_firm + params.delta))**(1.0 / (1.0 - params.alpha))
    w = (1.0 - params.alpha) * K_dem**params.alpha
    
    # 2. Worker HH Problem (Mass omega)
    params.beta = params.beta # Locked beta_worker
    a_pol_low, _, a_grid = household.solve_household_egm(r, w, params, y_grid, Pi)
    D_low = household.compute_distribution(a_pol_low, Pi, params)
    A_low = np.sum(D_low * a_grid[:, None])
    
    # 3. Elite HH Problem (Mass 1-omega)
    original_beta = params.beta
    params.beta = params.beta_high
    r_elite = r + params.premium
    a_pol_high, _, _ = household.solve_household_egm(r_elite, w, params, y_grid, Pi)
    D_high = household.compute_distribution(a_pol_high, Pi, params)
    A_high = np.sum(D_high * a_grid[:, None])
    params.beta = original_beta # Restore
    
    # 4. Aggregation
    A_total = params.omega * A_low + (1 - params.omega) * A_high
    D_total = params.omega * D_low + (1 - params.omega) * D_high
    
    return A_total - K_dem, A_total, K_dem, D_total, w, a_grid, D_low, D_high

def solve_steady_state(params):
    """
    Find equilibrium r for the Mixture Model.
    """
    y_grid, Pi, stat_y = income.get_income_grid(params)
    
    def market_clearing(r):
        diff, _, _, _, _, _, _, _ = get_mixture_asset_supply(r, params, y_grid, Pi)
        return diff
    
    # Bounds for r based on Elite patience
    r_max = (1.0 / params.beta_high) - 1.0 - params.premium - 0.001
    r_min = -0.04
    
    try:
        r_ss = brentq(market_clearing, r_min, r_max, xtol=1e-5)
    except:
        print("Equilibrium search failed. Checking signs...")
        f_min = market_clearing(r_min)
        f_max = market_clearing(r_max)
        r_ss = r_min if abs(f_min) < abs(f_max) else r_max
            
    # Compute final SS objects
    _, A_ss, K_ss, D_ss, w_ss, a_grid, D_low, D_high = get_mixture_asset_supply(r_ss, params, y_grid, Pi)
    
    return {
        'r': r_ss,
        'w': w_ss,
        'K': K_ss,
        'Y': K_ss**params.alpha,
        'A': A_ss,
        'D': D_ss,
        'D_low': D_low,
        'D_high': D_high,
        'a_grid': a_grid,
        'y_grid': y_grid
    }

