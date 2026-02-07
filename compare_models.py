import numpy as np
import matplotlib.pyplot as plt
import calibration
import income
import household
import steady_state
import os

def solve_rank(r, w, params):
    """
    Solve Representative Agent New Keynesian (RANK) consumption.
    Euler: u'(C) = beta * (1+r) * u'(C')
    Steady State: 1 = beta * (1+r).
    If r is given, RANK consumption is undetermined in partial eq (or determined by budget constraint C = rA + wL).
    We assume RANK agent holds the SAME aggregate assets as HANK.
    """
    # Just linear consumption rule: C(a) = r*a + w
    # (Assuming beta*(1+r)=1 holds, otherwise consumption grows/shrinks)
    
    # We will just plot the line C = r*a + w
    a_grid = params.a_min + (params.a_max - params.a_min) * np.linspace(0, 1, params.n_a)**2
    c_rank = r * a_grid + w # Consuming permanent income
    return a_grid, c_rank

def compare_steady_states():
    print("Comparing HANK (Mixture) vs RANK...")
    
    # 1. Setup Parameters (Winning Calibration)
    params = calibration.get_calibration()
    # Rank 2 Params
    beta_low = 0.900
    beta_high = 0.940
    omega = 0.99
    premium = 0.03
    sigma_e = 0.60
    
    params.update(sigma_e=sigma_e)
    
    # 2. Income Grid
    y_grid, Pi, stat_y = income.get_income_grid(params)
    
    # 3. Solve HANK Components
    # We need the equilibrium r first. From calibration, r=3.32% (0.0332)
    r_eq = 0.0332
    r_firm = r_eq + premium
    
    # Wages
    K_dem = (params.alpha / (r_firm + params.delta))**(1.0 / (1.0 - params.alpha))
    w_eq = (1.0 - params.alpha) * K_dem**params.alpha
    
    # Worker Policy
    params.beta = beta_low
    a_pol_low, c_pol_low, a_grid = household.solve_household_egm(r_eq, w_eq, params, y_grid, Pi)
    D_low = household.compute_distribution(a_pol_low, Pi, params)
    
    # Elite Policy
    params.beta = beta_high
    r_elite = r_eq + premium
    a_pol_high, c_pol_high, _ = household.solve_household_egm(r_elite, w_eq, params, y_grid, Pi)
    D_high = household.compute_distribution(a_pol_high, Pi, params)
    
    # 4. Solve RANK Benchmark
    # RANK agent with same r_eq and w_eq
    _, c_rank = solve_rank(r_eq, w_eq, params)
    
    # 5. Plotting
    os.makedirs('malaysia_hank/outputs', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Consumption Functions (The Mechanism)
    # Plot Average Consumption Policy over income states
    c_low_avg = np.dot(c_pol_low, stat_y)
    c_high_avg = np.dot(c_pol_high, stat_y)
    
    axes[0].plot(a_grid, c_low_avg, 'b-', linewidth=2, label=f'Worker (Beta={beta_low})')
    axes[0].plot(a_grid, c_high_avg, 'r--', linewidth=2, label=f'Elite (Beta={beta_high}, r+{premium*100:.0f}%)')
    axes[0].plot(a_grid, c_rank, 'k:', linewidth=1.5, label='RANK (Permanent Income)')
    
    axes[0].set_title('Consumption Policy Functions\n(Why HANK differs from RANK)')
    axes[0].set_xlabel('Assets (a)')
    axes[0].set_ylabel('Consumption (c)')
    axes[0].set_xlim(0, 50) # Zoom in on relevant range
    axes[0].set_ylim(0, 5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel B: Wealth Distribution (The Reality)
    # Aggregate D
    D_total = omega * D_low + (1 - omega) * D_high
    D_asset = np.sum(D_total, axis=1)
    
    axes[1].bar(a_grid, D_asset, width=a_grid[1]-a_grid[0], color='blue', alpha=0.6, label='HANK Mixture Distribution')
    # RANK is a point mass at K_dem
    # K_total = sum(D_total * a)
    K_total = np.sum(D_asset * a_grid)
    axes[1].axvline(K_total, color='black', linestyle='-', linewidth=2, label=f'RANK (Point Mass at {K_total:.1f})')
    
    axes[1].set_title('Wealth Distribution\n(HANK captures the Tails)')
    axes[1].set_xlabel('Assets (a)')
    axes[1].set_ylabel('Density')
    axes[1].set_xlim(0, 100)
    axes[1].set_yscale('log') # Log scale to see tails
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('malaysia_hank/outputs/model_comparison.png')
    print("Saved malaysia_hank/outputs/model_comparison.png")

if __name__ == "__main__":
    compare_steady_states()
