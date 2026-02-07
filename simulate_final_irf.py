import numpy as np
import matplotlib.pyplot as plt
import calibration
import steady_state
import dynamics
import os

def run_final_simulation():
    print("Simulating 25bps OPR Hike (Rank 4 Calibration)...")
    params = calibration.get_calibration()
    ss = steady_state.solve_steady_state(params)
    
    # 1. Compute Jacobian for Mixture Model
    T_sim = 20
    J_dict = dynamics.compute_mixture_jacobian(ss, params, T=T_sim)
    
    # 2. Simulate OPR Shock
    # dr_t = 0.0025 * 0.7^t
    shock_size = 0.0025
    persistence = 0.7
    dr, dC_dict = dynamics.simulate_opr_shock(ss, params, J_dict, shock_size=shock_size, persistence=persistence)
    
    # 3. Aggregate Consumption for normalization
    C_ss = np.sum(ss['D'] * (ss['r'] * ss['a_grid'][:, None] + ss['w'] * ss['y_grid'][None, :]))
    
    # Define Segment steady states for plotting
    # Helper to aggregate by mask
    D_a = np.sum(ss['D'], axis=1)
    cum_w = np.cumsum(D_a)
    idx_40 = np.searchsorted(cum_w, 0.40)
    idx_80 = np.searchsorted(cum_w, 0.80)
    n_a = params.n_a
    mask_b40 = np.zeros(n_a, dtype=bool); mask_b40[:idx_40] = True
    mask_t20 = np.zeros(n_a, dtype=bool); mask_t20[idx_80:] = True
    
    # Re-calculate segment J for plotting (B40 / T20)
    # Dynamics.py compute_mixture_jacobian only tracks 'Agg' in the simplified return
    # Let's fix dynamics.py return or manually compute here
    # For now, let's plot Aggregate IRF
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(T_sim), dC_dict['Agg'] / C_ss * 100, 'b-o', linewidth=2, label='National Consumption')
    plt.axhline(0, color='black', alpha=0.3)
    plt.title('Impact of 25bps OPR Hike on Malaysian Consumption\n(Rank 4 Calibrated Mixture Model)')
    plt.ylabel('% Deviation from Steady State')
    plt.xlabel('Quarters after Shock')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs('malaysia_hank/outputs', exist_ok=True)
    plt.savefig('malaysia_hank/outputs/final_mixture_irf.png')
    print("Saved malaysia_hank/outputs/final_mixture_irf.png")

if __name__ == "__main__":
    run_final_simulation()
