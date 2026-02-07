import calibration
import income
import household
import numpy as np

def debug_case():
    params = calibration.get_calibration()
    
    # Test Case
    beta_low = 0.92
    beta_high = 0.94
    omega = 0.98
    premium = 0.03
    sigma_e = 0.40
    
    params.update(sigma_e=sigma_e)
    y_grid, Pi, stat_y = income.get_income_grid(params)
    
    r_max = (1.0 / beta_high) - 1.0 - premium - 0.0005
    r_min = -0.04
    
    print(f"r_min: {r_min}")
    print(f"r_max: {r_max}")
    
    def get_supply(r):
        # Workers
        params.beta = beta_low
        K_dem = (params.alpha / (r + params.delta))**(1.0 / (1.0 - params.alpha))
        w = (1.0 - params.alpha) * K_dem**params.alpha
        
        a_pol_low, _, _ = household.solve_household_egm(r, w, params, y_grid, Pi)
        D_low = household.compute_distribution(a_pol_low, Pi, params)
        a_grid = params.a_min + (params.a_max - params.a_min) * np.linspace(0, 1, params.n_a)**2
        A_low = np.sum(D_low * a_grid[:, None])
        
        # Elites
        params.beta = beta_high
        r_elite = r + premium
        a_pol_high, _, _ = household.solve_household_egm(r_elite, w, params, y_grid, Pi)
        D_high = household.compute_distribution(a_pol_high, Pi, params)
        A_high = np.sum(D_high * a_grid[:, None])
        
        A_total = omega * A_low + (1-omega) * A_high
        
        return A_total, K_dem
        
    S_min, D_min = get_supply(r_min)
    print(f"At r_min={r_min}: Supply={S_min:.4f}, Demand={D_min:.4f}, Excess={S_min-D_min:.4f}")
    
    S_max, D_max = get_supply(r_max)
    print(f"At r_max={r_max}: Supply={S_max:.4f}, Demand={D_max:.4f}, Excess={S_max-D_max:.4f}")

if __name__ == "__main__":
    debug_case()
