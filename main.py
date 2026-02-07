import numpy as np
import matplotlib.pyplot as plt
import calibration
import steady_state

def gini(x, w=None):
    # The array x contains the values (assets)
    # The array w contains the weights (population mass)
    # Sort x
    idx = np.argsort(x)
    x = x[idx]
    if w is not None:
        w = w[idx]
        w = w / np.sum(w)
    else:
        w = np.ones_like(x) / len(x)
        
    cum_w = np.cumsum(w)
    cum_x = np.cumsum(w * x)
    
    # Lorenz curve
    # L(p) = cum_x(p) / Total
    return 1 - 2 * np.trapz(cum_x / cum_x[-1], cum_w)

def compute_wealth_stats(ss, params):
    D = ss['D']
    a_grid = ss['a_grid']
    
    # Marginal distribution of assets
    D_a = np.sum(D, axis=1)
    
    # Total mass check
    print(f"Total Mass: {np.sum(D_a):.6f}")
    
    # Gini
    g = gini(a_grid, D_a)
    
    # Top Shares
    # Sort
    idx = np.argsort(a_grid)
    a_sorted = a_grid[idx]
    w_sorted = D_a[idx]
    
    cum_pop = np.cumsum(w_sorted)
    cum_wealth = np.cumsum(w_sorted * a_sorted)
    total_wealth = cum_wealth[-1]
    
    # Function to find share
    def get_share(percentile):
        # find index where cum_pop >= percentile
        cutoff = np.searchsorted(cum_pop, percentile)
        cutoff = np.clip(cutoff, 0, len(cum_wealth) - 1)
        wealth_below = cum_wealth[cutoff]
        return 1 - wealth_below / total_wealth
        
    top_10 = get_share(0.90)
    top_1 = get_share(0.99)
    bottom_50_wealth = cum_wealth[np.searchsorted(cum_pop, 0.50)] / total_wealth
    
    return g, top_10, top_1, bottom_50_wealth

def main():
    print("Initializing Malaysia HANK Model...")
    params = calibration.get_calibration()
    
    print("Solving for Steady State (this may take a moment)...")
    ss = steady_state.solve_steady_state(params)
    
    print("\n--- Steady State Results ---")
    print(f"Interest Rate (r): {ss['r']*100:.2f}% (Target: ~3%)")
    print(f"Capital/Output (K/Y): {ss['K']/ss['Y']:.2f}")
    print(f"Wage (w): {ss['w']:.2f}")
    
    print("\n--- Distributional Statistics ---")
    g, top_10, top_1, bot_50 = compute_wealth_stats(ss, params)
    
    print(f"Wealth Gini:       {g:.3f} (Malaysia Target: ~0.40 Income / Higher for Wealth)")
    print(f"Top 10% Share:     {top_10*100:.1f}% (Malaysia WID: ~35-40%)")
    print(f"Top 1% Share:      {top_1*100:.1f}% (Malaysia WID: ~11%)")
    print(f"Bottom 50% Share:  {bot_50*100:.1f}%")
    
    # Validation against targets
    print("\n--- Calibration Check ---")
    if abs(ss['r'] - 0.03) < 0.005:
        print("[PASS] Interest rate close to target.")
    else:
        print("[FAIL] Interest rate deviation.")
        
    if 0.35 <= top_10 <= 0.60:
         print("[PASS] Top 10% share reasonable for EM.")
    else:
         print("[WARN] Top 10% share off target (Adjust sigma_e or borrowing constraint).")

    # Plotting
    print("\n--- Generating Plots ---")
    D = ss['D']
    a_grid = ss['a_grid']
    D_a = np.sum(D, axis=1) # Marginal asset dist
    
    # Sort for Lorenz
    idx = np.argsort(a_grid)
    a_sorted = a_grid[idx]
    w_sorted = D_a[idx]
    
    cum_w = np.cumsum(w_sorted)
    cum_w /= cum_w[-1]
    cum_wealth = np.cumsum(w_sorted * a_sorted)
    cum_wealth /= cum_wealth[-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(cum_w, cum_wealth, label='Model Lorenz Curve', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
    
    # Add Malaysia Data Points (Approximate from WID/World Bank)
    # Bottom 50% hold ~9-11% -> (0.5, 0.1)
    # Top 10% hold ~40% -> Bottom 90% hold 60% -> (0.9, 0.6)
    plt.scatter([0.5, 0.9], [0.11, 0.60], color='red', zorder=5, label='Malaysia Data Targets (WID)')
    
    plt.title('Wealth Distribution: Model vs Malaysia Data')
    plt.xlabel('Cumulative Share of Population')
    plt.ylabel('Cumulative Share of Wealth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('malaysia_hank/lorenz_curve.png')
    print("Saved lorenz_curve.png")

if __name__ == "__main__":
    main()
