import numpy as np
import itertools
from scipy.optimize import brentq
import calibration
import income
import household
import os

# --- Helper Functions ---

def gini(x, w=None):
    x = np.asarray(x)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w)
    
    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    sorted_w = w[sorted_indices]
    
    cumw = np.cumsum(sorted_w)
    cumxw = np.cumsum(sorted_x * sorted_w)
    
    total_weight = cumw[-1]
    total_wealth = cumxw[-1]
    
    # Lorenz curve points: (0,0) + (cumw/TotalW, cumxw/TotalWealth)
    # Area under curve using trapezoids
    # Area = sum( 0.5 * (y_i + y_{i-1}) * (x_i - x_{i-1}) )
    
    # Prepend 0
    X = np.concatenate(([0], cumw / total_weight))
    Y = np.concatenate(([0], cumxw / total_wealth))
    
    # Area
    B = np.trapz(Y, X)
    
    # Gini = 1 - 2*B
    return 1.0 - 2.0 * B

def compute_wealth_stats(D_total, a_grid):
    D_a = np.sum(D_total, axis=1) # [n_a]
    g = gini(a_grid, w=D_a)
    
    total_wealth = np.sum(D_a * a_grid)
    cum_pop = np.cumsum(D_a)
    cum_wealth = np.cumsum(D_a * a_grid)
    
    idx_90 = np.searchsorted(cum_pop, 0.90)
    idx_99 = np.searchsorted(cum_pop, 0.99)
    
    wealth_bot_90 = cum_wealth[idx_90] if idx_90 < len(cum_wealth) else total_wealth
    wealth_top_10 = total_wealth - wealth_bot_90
    share_10 = wealth_top_10 / total_wealth if total_wealth > 0 else 0
    
    wealth_bot_99 = cum_wealth[idx_99] if idx_99 < len(cum_wealth) else total_wealth
    wealth_top_1 = total_wealth - wealth_bot_99
    share_1 = wealth_top_1 / total_wealth if total_wealth > 0 else 0
    
    return g, share_10, share_1

def solve_two_type_model(beta_low, beta_high, omega, return_premium, sigma_e, verbose=False):
    params = calibration.get_calibration()
    params.update(sigma_e=sigma_e)
    
    y_grid, Pi, stat_y = income.get_income_grid(params)
    
    def get_aggregate_supply(r):
        if r + params.delta <= 0: return -1e9
        if r + return_premium >= (1/beta_high - 1): return 1e9 
        
        # Firms Cost of Capital
        r_firm = r + return_premium
        K_dem = (params.alpha / (r_firm + params.delta))**(1.0 / (1.0 - params.alpha))
        w = (1.0 - params.alpha) * K_dem**params.alpha
        
        # Low Type
        params.beta = beta_low
        a_pol_low, _, _ = household.solve_household_egm(r, w, params, y_grid, Pi)
        D_low = household.compute_distribution(a_pol_low, Pi, params)
        a_grid = params.a_min + (params.a_max - params.a_min) * np.linspace(0, 1, params.n_a)**2
        A_low = np.sum(D_low * a_grid[:, None])
        
        # High Type
        params.beta = beta_high
        r_elite = r + return_premium
        a_pol_high, _, _ = household.solve_household_egm(r_elite, w, params, y_grid, Pi)
        D_high = household.compute_distribution(a_pol_high, Pi, params)
        A_high = np.sum(D_high * a_grid[:, None])
        
        A_total = omega * A_low + (1 - omega) * A_high
        return A_total - K_dem, D_low, D_high, a_grid

    r_max = (1.0 / beta_high) - 1.0 - return_premium - 0.0005
    r_min = -0.04 
    
    try:
        diff_min, _, _, _ = get_aggregate_supply(r_min)
        diff_max, _, _, _ = get_aggregate_supply(r_max)
    except:
        return None
        
    if diff_min * diff_max > 0:
        return None
        
    try:
        r_ss = brentq(lambda x: get_aggregate_supply(x)[0], r_min, r_max, xtol=1e-5)
    except:
        return None
        
    _, D_low, D_high, a_grid = get_aggregate_supply(r_ss)
    D_total = omega * D_low + (1 - omega) * D_high
    g, top10, top1 = compute_wealth_stats(D_total, a_grid)
    
    return {
        'r': r_ss,
        'gini': g,
        'top10': top10,
        'top1': top1
    }

def run_grid_search():
    # Final Feasible Ranges
    betas_low = np.linspace(0.90, 0.94, 3) 
    betas_high = np.linspace(0.93, 0.94, 2)
    omegas = np.linspace(0.95, 0.99, 3) 
    premiums = np.linspace(0.03, 0.05, 2)
    sigmas_e = np.linspace(0.40, 0.60, 2)
    
    target_r = 0.03
    target_gini = 0.755
    target_top10 = 0.607
    target_top1 = 0.288
    
    w1, w2, w3, w4 = 10.0, 5.0, 2.0, 5.0 
    
    results = []
    total_iter = len(betas_low) * len(betas_high) * len(omegas) * len(premiums) * len(sigmas_e)
    print(f"Starting Grid Search with {total_iter} iterations...")
    
    count = 0
    for b_l, b_h, om, prem, sig in itertools.product(betas_low, betas_high, omegas, premiums, sigmas_e):
        count += 1
        if count % 10 == 0:
            print(f"Progress: {count}/{total_iter}")
            
        res = solve_two_type_model(b_l, b_h, om, prem, sig)
        
        if res is not None:
            loss = (w1 * (res['r'] - target_r)**2 + 
                    w2 * (res['gini'] - target_gini)**2 + 
                    w3 * (res['top10'] - target_top10)**2 + 
                    w4 * (res['top1'] - target_top1)**2)
            
            entry = {
                'beta_low': b_l,
                'beta_high': b_h,
                'omega': om,
                'premium': prem,
                'sigma_e': sig,
                'r': res['r'],
                'gini': res['gini'],
                'top10': res['top10'],
                'top1': res['top1'],
                'loss': loss
            }
            results.append(entry)

    if not results:
        print("No solutions found.")
        return

    results.sort(key=lambda x: x['loss'])
    
    os.makedirs('malaysia_hank/outputs', exist_ok=True)
    with open('malaysia_hank/outputs/calibration_results.md', 'w') as f:
        f.write("# Malaysia HANK Calibration Results (Grid Search)\n\n")
        f.write(f"**Targets:** r={target_r}, Gini={target_gini}, Top10={target_top10}, Top1={target_top1}\n\n")
        f.write("| Rank | Loss | Beta_L | Beta_H | Omega | Premium | Sigma_e | r (%) | Gini | Top10 | Top1 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        
        for i, r in enumerate(results[:20]): 
            f.write(f"| {i+1} | {r['loss']:.5f} | {r['beta_low']:.3f} | {r['beta_high']:.3f} | {r['omega']:.4f} | {r['premium']:.3f} | {r['sigma_e']:.3f} | {r['r']*100:.2f}% | {r['gini']:.3f} | {r['top10']:.3f} | {r['top1']:.3f} |\n")
            
    print("Grid search complete. Results saved.")

if __name__ == "__main__":
    run_grid_search()