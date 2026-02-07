import numpy as np
import matplotlib.pyplot as plt
import calibration
import steady_state
import household
import os

def gini(x, w=None):
    x = np.asarray(x)
    if w is None: w = np.ones_like(x)
    w = np.asarray(w)
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    X = np.concatenate(([0], np.cumsum(w)/np.sum(w)))
    Y = np.concatenate(([0], np.cumsum(x*w)/np.sum(x*w)))
    return 1.0 - 2.0 * np.trapz(Y, X)

def run_validation():
    print("Solving Baseline Malaysia HANK (Rank 4 Calibration)...")
    params = calibration.get_calibration()
    ss = steady_state.solve_steady_state(params)
    
    # 1. Wealth Stats
    D_a = np.sum(ss['D'], axis=1)
    a_grid = ss['a_grid']
    g = gini(a_grid, w=D_a)
    
    total_wealth = np.sum(D_a * a_grid)
    cum_pop = np.cumsum(D_a)
    cum_wealth = np.cumsum(D_a * a_grid)
    
    def get_share(percentile):
        idx = np.searchsorted(cum_pop, percentile)
        return (total_wealth - cum_wealth[idx]) / total_wealth

    top10 = get_share(0.90)
    top1 = get_share(0.99)
    
    print("\n--- BASELINE RESULTS ---")
    print(f"Real Interest Rate (r): {ss['r']*100:.2f}% (Target: 3.0%)")
    print(f"Wealth Gini:           {g:.3f} (Target: 0.755)")
    print(f"Top 10% Wealth Share:  {top10*100:.1f}% (Target: 60.7%)")
    print(f"Top 1% Wealth Share:   {top1*100:.1f}% (Target: 28.8%)")
    print("------------------------\n")
    
    # 2. Plotting
    os.makedirs('malaysia_hank/outputs', exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Lorenz Curve
    plt.subplot(1, 2, 1)
    X = np.concatenate(([0], np.cumsum(D_a)))
    Y = np.concatenate(([0], np.cumsum(a_grid * D_a) / total_wealth))
    plt.plot(X, Y, 'b-', label=f'Model Lorenz (Gini={g:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.title('Lorenz Curve (Wealth)')
    plt.xlabel('Cumulative Population')
    plt.ylabel('Cumulative Wealth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # PDF
    plt.subplot(1, 2, 2)
    plt.bar(a_grid, D_a, width=a_grid[1]-a_grid[0], color='red', alpha=0.6)
    plt.title('Wealth Distribution (Asset PDF)')
    plt.xlabel('Assets (a)')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('malaysia_hank/outputs/baseline_validation.png')
    print("Saved malaysia_hank/outputs/baseline_validation.png")

if __name__ == "__main__":
    run_validation()
