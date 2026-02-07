import numpy as np
import household
import calibration
import income
import steady_state

def compute_mixture_jacobian(ss, params, T=50):
    """
    Computes the Jacobian for the Mixture Model (Workers + Elites).
    """
    n_a = params.n_a
    n_e = params.n_e
    y_grid, Pi, _ = income.get_income_grid(params)
    a_grid = ss['a_grid']
    r_ss = ss['r']
    w_ss = ss['w']
    D_ss = ss['D'] # Aggregate SS Dist
    
    # Track Separate Segment Responses
    J = {'Agg': np.zeros((T, T)), 'B40': np.zeros((T, T)), 'T20': np.zeros((T, T))}
    
    # 1. Define Percentile Masks on Aggregate Dist
    D_a = np.sum(D_ss, axis=1)
    cum_w = np.cumsum(D_a)
    idx_40 = np.searchsorted(cum_w, 0.40)
    idx_80 = np.searchsorted(cum_w, 0.80)
    
    mask_b40 = np.zeros(n_a, dtype=bool); mask_b40[:idx_40] = True
    mask_t20 = np.zeros(n_a, dtype=bool); mask_t20[idx_80:] = True
    
    def agg_segment(D, c_pol, mask):
        return np.sum(D[mask, :] * c_pol[mask, :])

    C_ss_agg = np.sum(D_ss * (r_ss * a_grid[:, None] + w_ss * y_grid[None, :])) # Simplified SS check
    # Actually use proper SS consumption policy from EGM
    # To be precise, re-solve SS policies
    params.beta = params.beta # Worker
    _, c_ss_low, _ = household.solve_household_egm(r_ss, w_ss, params, y_grid, Pi)
    params.beta = params.beta_high # Elite
    _, c_ss_high, _ = household.solve_household_egm(r_ss + params.premium, w_ss, params, y_grid, Pi)
    
    C_ss_agg = params.omega * np.sum(ss['D_low'] * c_ss_low) + (1-params.omega) * np.sum(ss['D_high'] * c_ss_high)
    
    eps = 1e-4
    print(f"Computing Mixture Jacobian for T={T}...")
    
    for s in range(T):
        # 1. Path of r
        r_path = np.full(T, r_ss)
        r_path[s] += eps
        
        # 2. Backward Iteration for both types
        c_path_low = [None] * T
        c_path_high = [None] * T
        
        c_next_low = c_ss_low
        c_next_high = c_ss_high
        
        for t in reversed(range(T)):
            if t > s:
                c_path_low[t] = c_ss_low
                c_path_high[t] = c_ss_high
            else:
                # Solve Worker (r)
                params.beta = params.beta
                _, c_path_low[t], _ = household.solve_household_egm(r_path[t], w_ss, params, y_grid, Pi)
                # Solve Elite (r + premium)
                params.beta = params.beta_high
                _, c_path_high[t], _ = household.solve_household_egm(r_path[t] + params.premium, w_ss, params, y_grid, Pi)
            
        # 3. Forward Iteration for Distribution
        D_t_low = ss['D_low']
        D_t_high = ss['D_high']
        
        for t in range(T):
            C_t_low = np.sum(D_t_low * c_path_low[t])
            C_t_high = np.sum(D_t_high * c_path_high[t])
            C_t_agg = params.omega * C_t_low + (1 - params.omega) * C_t_high
            
            J['Agg'][t, s] = (C_t_agg - C_ss_agg) / eps
            
            # Update Distributions (simplified transition)
            # D_t_low = household.compute_distribution(a_pol_low_t, Pi, params)
            # ... For Jacobian we often assume fixed distribution (Direct Effect)
            # or update. Let's do fixed distribution for speed in this demo.
            
        if s % 10 == 0: print(f"Progress: {s}/{T}")

    return J

def simulate_opr_shock(ss, params, J_dict, shock_size=0.0025, persistence=0.7):
    """
    Simulates a 25bps shock.
    """
    T = J_dict['Agg'].shape[0]
    dr = np.zeros(T)
    for t in range(T):
        dr[t] = shock_size * (persistence**t)
        
    responses = {}
    for key, J in J_dict.items():
        responses[key] = np.dot(J, dr)
    
    return dr, responses

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Loading Malaysia HANK Steady State...")
    params = calibration.get_calibration()
    params.beta = 0.935 
    ss = steady_state.solve_steady_state(params)
    
    # Compute Jacobian
    T_sim = 20
    J_dict = compute_household_jacobian(ss, params, T=T_sim)
    
    # Simulate
    dr, dC_dict = simulate_opr_shock(ss, params, J_dict, shock_size=0.0025)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(T_sim), dr * 10000, 'r-o', label='OPR Shock (bps)')
    plt.title('Monetary Policy Shock (25bps Hike)')
    plt.ylabel('Basis Points')
    plt.xlabel('Quarters')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Calculate Group Steady States for Normalization
    D_a = np.sum(ss['D'], axis=1)
    cum_w = np.cumsum(D_a)
    idx_40 = np.searchsorted(cum_w, 0.40)
    idx_80 = np.searchsorted(cum_w, 0.80)
    
    # Masks
    n_a = params.n_a
    mask_b40 = np.zeros(n_a, dtype=bool); mask_b40[:idx_40] = True
    mask_m40 = np.zeros(n_a, dtype=bool); mask_m40[idx_40:idx_80] = True
    mask_t20 = np.zeros(n_a, dtype=bool); mask_t20[idx_80:] = True
    
    # Helper to get SS Consumption for groups
    def get_C_ss(D, c_pol, mask):
        return np.sum(D[mask, :] * c_pol[mask, :])
        
    C_b40_ss = get_C_ss(ss['D'], ss['c_pol'], mask_b40)
    C_m40_ss = get_C_ss(ss['D'], ss['c_pol'], mask_m40)
    C_t20_ss = get_C_ss(ss['D'], ss['c_pol'], mask_t20)
    C_agg_ss = np.sum(ss['D'] * ss['c_pol'])

    # Plot normalized responses (% deviation from GROUP steady state)
    plt.plot(range(T_sim), dC_dict['B40'] / C_b40_ss * 100, 'r-', label='B40 (Poor)', linewidth=2.5)
    plt.plot(range(T_sim), dC_dict['M40'] / C_m40_ss * 100, color='orange', label='M40 (Middle)', linewidth=2, linestyle='--')
    plt.plot(range(T_sim), dC_dict['T20'] / C_t20_ss * 100, 'g-', label='T20 (Rich)', linewidth=2)
    plt.plot(range(T_sim), dC_dict['Agg'] / C_agg_ss * 100, 'k:', label='National Avg', linewidth=1, alpha=0.6)
    
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.title('Living Standards Impact (Relative to Pre-Shock)')
    plt.ylabel('% Change in Group Consumption')
    plt.xlabel('Quarters after Hike')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('malaysia_hank/shock_response_segmented.png')
    print("Saved shock_response_segmented.png")
