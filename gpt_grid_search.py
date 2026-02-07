import numpy as np
from itertools import product
from scipy.optimize import brentq
import calibration
import income
import household
import os

# ======================================================
# 1. Global Parameters & Helpers
# ======================================================

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
    
    # Prepend 0
    X = np.concatenate(([0], cumw / total_weight))
    Y = np.concatenate(([0], cumxw / total_wealth))
    
    # Area
    B = np.trapz(Y, X)
    
    return 1.0 - 2.0 * B

def compute_wealth_stats(D_total, a_grid):
    # D_total is (n_a, n_e) - we sum over income to get marginal asset dist
    D_a = np.sum(D_total, axis=1) # [n_a]
    
    total_wealth = np.sum(D_a * a_grid)
    cum_pop = np.cumsum(D_a)
    cum_wealth = np.cumsum(D_a * a_grid)
    
    # Interpolate for precise percentiles
    # Or just use searchsorted for discrete grid approximation
    def get_share(percentile):
        idx = np.searchsorted(cum_pop, percentile)
        if idx >= len(a_grid): return 0.0
        
        wealth_below = cum_wealth[idx]
        wealth_above = total_wealth - wealth_below
        return wealth_above / total_wealth

    top10 = get_share(0.90)
    top1 = get_share(0.99)
    g = gini(a_grid, w=D_a)
    
    return g, top10, top1

# ======================================================
# 2. General Equilibrium Block
# ======================================================

def solve_model_at_r(r, params, y_grid, Pi, a_grid):
    """
    Solve for A_total given r.
    """
    # 1. Prices
    # Firms pay Elite Rate (r + premium)
    r_firm = r + params.return_premium
    if r_firm + params.delta <= 0: return -1e9 # Invalid
    
    # Capital Demand
    K_dem = (params.alpha / (r_firm + params.delta))**(1.0 / (1.0 - params.alpha))
    w = (1.0 - params.alpha) * K_dem**params.alpha
    
    # 2. Household Problem (Worker)
    # Uses base r
    params.beta = params.beta_low # Set temp beta
    a_pol_low, _, _ = household.solve_household_egm(r, w, params, y_grid, Pi)
    D_low = household.compute_distribution(a_pol_low, Pi, params)
    A_low = np.sum(D_low * a_grid[:, None])
    
    # 3. Household Problem (Elite)
    # Uses Elite r
    params.beta = params.beta_high # Set temp beta
    r_elite = r + params.return_premium
    a_pol_high, _, _ = household.solve_household_egm(r_elite, w, params, y_grid, Pi)
    D_high = household.compute_distribution(a_pol_high, Pi, params)
    A_high = np.sum(D_high * a_grid[:, None])
    
    # 4. Aggregation
    A_total = params.omega * A_low + (1 - params.omega) * A_high
    D_total = params.omega * D_low + (1 - params.omega) * D_high
    
    return A_total - K_dem, D_total

def equilibrium_r(params):
    """
    Find equilibrium r using Brent's method.
    """
    # Get Income Grid based on current sigma_e
    y_grid, Pi, stat_y = income.get_income_grid(params)
    
    # Construct Asset Grid
    a_grid = params.a_min + (params.a_max - params.a_min) * np.linspace(0, 1, params.n_a)**2
    
    # Define Excess Demand Function
    def excess_demand(r):
        diff, _ = solve_model_at_r(r, params, y_grid, Pi, a_grid)
        return diff
        
    # Bounds
    # Low r constrained by -delta (or practical -0.05)
    # High r constrained by Elite patience: r_elite < 1/beta_high - 1
    # r + premium < 1/beta_high - 1 => r < 1/beta_high - 1 - premium
    r_max = (1.0 / params.beta_high) - 1.0 - params.return_premium - 0.001
    r_min = -0.04
    
    try:
        r_star = brentq(excess_demand, r_min, r_max, xtol=1e-5)
        # Recover distribution
        _, D_total = solve_model_at_r(r_star, params, y_grid, Pi, a_grid)
        return r_star, D_total, a_grid
    except:
        return None, None, None

# ======================================================
# 3. SMM Loss Function
# ======================================================

def loss_function(r, stats):
    gini, top10, top1 = stats
    
    # Targets
    t_r = 0.03
    t_gini = 0.755
    t_top10 = 0.607
    t_top1 = 0.288
    
    # Weights
    w1, w2, w3, w4 = 10.0, 5.0, 2.0, 5.0 
    
    loss = (w1 * (r - t_r)**2 + 
            w2 * (gini - t_gini)**2 + 
            w3 * (top10 - t_top10)**2 + 
            w4 * (top1 - t_top1)**2)
            
    return loss

# ======================================================
# 4. Grid Search Execution
# ======================================================

def run_gpt_grid_search():
    # Base Calibration Object
    base_params = calibration.get_calibration()
    
    # Grid Arrays (Adjusted for feasibility and scope)
    betas_low = np.linspace(0.90, 0.94, 3) 
    betas_high = np.linspace(0.935, 0.96, 3) # Adjusted up slightly for stability
    omegas = np.linspace(0.95, 0.99, 3)
    premiums = np.linspace(0.03, 0.05, 2)
    sigmas_e = np.linspace(0.40, 0.60, 2)
    
    results = []
    best = {"loss": np.inf}
    
    total_iter = len(betas_low) * len(betas_high) * len(omegas) * len(premiums) * len(sigmas_e)
    print(f"Starting GPT-style Grid Search with {total_iter} iterations...")
    
    count = 0
    for b_l, b_h, om, prem, sig in product(betas_low, betas_high, omegas, premiums, sigmas_e):
        count += 1
        if count % 10 == 0: print(f"Progress: {count}/{total_iter}")
        
        # Set parameters on object
        base_params.beta_low = b_l
        base_params.beta_high = b_h
        base_params.omega = om
        base_params.return_premium = prem
        base_params.sigma_e = sig
        base_params.update(sigma_e=sig) # Ensure income grid uses this
        
        # Solve
        r_eq, D_total, a_grid = equilibrium_r(base_params)
        
        if r_eq is not None:
            g, top10, top1 = compute_wealth_stats(D_total, a_grid)
            stats = (g, top10, top1)
            loss = loss_function(r_eq, stats)
            
            entry = {
                "loss": loss,
                "beta_low": b_l,
                "beta_high": b_h,
                "omega": om,
                "premium": prem,
                "sigma_e": sig,
                "r": r_eq,
                "gini": g,
                "top10": top10,
                "top1": top1
            }
            results.append(entry)
            
            if loss < best["loss"]:
                best = entry
                print(f"New Best! Loss={loss:.4f} | r={r_eq:.2%} | Gini={g:.3f}")

    # Save to Markdown
    results.sort(key=lambda x: x['loss'])
    os.makedirs('malaysia_hank/outputs', exist_ok=True)
    with open('malaysia_hank/outputs/gpt_calibration_results.md', 'w') as f:
        f.write("# GPT Grid Search Results\n\n")
        f.write("| Rank | Loss | Beta_L | Beta_H | Omega | Premium | Sigma_e | r (%) | Gini | Top10 | Top1 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(results[:20]):
            f.write(f"| {i+1} | {r['loss']:.5f} | {r['beta_low']:.3f} | {r['beta_high']:.3f} | {r['omega']:.4f} | {r['premium']:.3f} | {r['sigma_e']:.3f} | {r['r']*100:.2f}% | {r['gini']:.3f} | {r['top10']:.3f} | {r['top1']:.3f} |\n")

    print("\nFINAL BEST RESULT:")
    print(best)

if __name__ == "__main__":
    run_gpt_grid_search()
