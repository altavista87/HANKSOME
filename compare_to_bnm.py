"""
Comparison Script: Malaysia MoE HANK vs BNM Benchmarks
======================================================
Comparing model outputs after 3,000 epochs against typical 
Bank Negara Malaysia (BNM) and Department of Statistics (DOSM) figures.
"""

import torch
import numpy as np
import pandas as pd
from malaysia_deep_hank_architecture import MalaysiaExtendedParams, MalaysiaExtendedHANK

def calculate_gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        # Shift values so that they are positive (Gini is defined for non-negative values)
        array -= np.amin(array)
    array += 0.0000001  # Values cannot be 0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def get_mpc(model, state, prices, eps=0.1):
    """Compute MPC for a batch of households."""
    policies = model.compute_economic_policies(state, prices)
    c_base = policies['consumption']
    
    state_plus = {k: v.clone() for k, v in state.items()}
    state_plus['continuous'] = state['continuous'].clone()
    state_plus['continuous'][:, 0] += eps
    
    policies_plus = model.compute_economic_policies(state_plus, prices)
    c_plus = policies_plus['consumption']
    
    mpc = (c_plus - c_base) / eps
    return mpc.detach().cpu().numpy()

def run_comparison():
    # 1. Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    params = MalaysiaExtendedParams()
    model = MalaysiaExtendedHANK(params).to(device)
    
    try:
        checkpoint = torch.load("hank_checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: No checkpoint found! Train model first.")
        return

    # 2. Simulate a Population (10,000 households)
    N = 10000
    prices = {'r_liquid': 0.03, 'r_illiquid': 0.05, 'r_mortgage': 0.045, 'w': 1.0}
    
    # Create realistic distribution (log-normal income/assets)
    # This is a proxy for the actual ergodic distribution
    torch.manual_seed(42)
    wealth_dist = torch.exp(torch.randn(N, device=device) * 0.8 + 1.0) # Log-normal wealth
    
    # Sort to assign groups
    sorted_indices = torch.argsort(wealth_dist)
    b40_idx = sorted_indices[:int(0.4*N)]
    m40_idx = sorted_indices[int(0.4*N):int(0.8*N)]
    t20_idx = sorted_indices[int(0.8*N):]
    
    # Construct state tensor
    continuous = torch.zeros(N, 5, device=device)
    continuous[:, 0] = wealth_dist * 0.2  # 20% liquid
    continuous[:, 1] = wealth_dist * 0.8  # 80% illiquid
    continuous[:, 2] = 1.0  # Human capital (placeholder)
    continuous[:, 3] = torch.randint(20, 60, (N,), device=device).float() # Age
    
    state = {
        'continuous': continuous,
        'education': torch.randint(0, 3, (N,), device=device),
        'health': torch.zeros(N, dtype=torch.long, device=device),
        'housing': torch.randint(0, 3, (N,), device=device),
        'location': torch.zeros(N, dtype=torch.long, device=device),
        'sector': torch.zeros(N, dtype=torch.long, device=device)
    }
    
    # 3. Compute Metrics
    policies = model.compute_economic_policies(state, prices)
    consumption = policies['consumption'].detach().cpu().numpy().flatten()
    liquid_assets = policies['liquid_assets'].detach().cpu().numpy().flatten()
    illiquid_assets = policies['illiquid_assets'].detach().cpu().numpy().flatten()
    total_wealth = liquid_assets + illiquid_assets
    
    mpc_all = get_mpc(model, state, prices)
    
    # Group Averages
    b40_mpc = np.mean(mpc_all[b40_idx.cpu()])
    m40_mpc = np.mean(mpc_all[m40_idx.cpu()])
    t20_mpc = np.mean(mpc_all[t20_idx.cpu()])
    
    b40_cons = np.mean(consumption[b40_idx.cpu()])
    t20_cons = np.mean(consumption[t20_idx.cpu()])
    
    wealth_gini = calculate_gini(total_wealth)
    
    # 4. BNM Targets (Approximate from Annual Reports/WID)
    bnm_targets = {
        "MPC_B40": 0.65,
        "MPC_T20": 0.25,
        "Wealth_Gini": 0.65,
        "Cons_Ratio_T20_B40": 3.5  # T20 consumes 3.5x more than B40
    }
    
    # 5. Scorecard Output
    print("\n" + "="*60)
    print("MALAYSIA MOE HANK vs. BNM BENCHMARKS (Scorecard)")
    print("="*60)
    print(f"{ 'Metric':<25} | {'Model':<10} | {'BNM Target':<10} | {'Status'}")
    print("-" * 60)
    
    # MPC B40
    diff_b40 = abs(b40_mpc - bnm_targets['MPC_B40'])
    status_b40 = "✅ Good" if diff_b40 < 0.1 else ("⚠️ High" if b40_mpc > bnm_targets['MPC_B40'] else "⚠️ Low")
    print(f"{ 'MPC (B40)':<25} | {b40_mpc:.2f}       | {bnm_targets['MPC_B40']:.2f}       | {status_b40}")
    
    # MPC T20
    diff_t20 = abs(t20_mpc - bnm_targets['MPC_T20'])
    status_t20 = "✅ Good" if diff_t20 < 0.1 else ("⚠️ Low" if t20_mpc < bnm_targets['MPC_T20'] else "⚠️ High")
    print(f"{ 'MPC (T20)':<25} | {t20_mpc:.2f}       | {bnm_targets['MPC_T20']:.2f}       | {status_t20}")
    
    # Wealth Gini
    diff_gini = abs(wealth_gini - bnm_targets['Wealth_Gini'])
    status_gini = "✅ Good" if diff_gini < 0.05 else "⚠️ Off"
    print(f"{ 'Wealth Gini':<25} | {wealth_gini:.2f}       | {bnm_targets['Wealth_Gini']:.2f}       | {status_gini}")
    
    # Consumption Inequality
    cons_ratio = t20_cons / (b40_cons + 1e-6)
    diff_cons = abs(cons_ratio - bnm_targets['Cons_Ratio_T20_B40'])
    status_cons = "✅ Good" if diff_cons < 1.0 else "⚠️ Low"
    print(f"{ 'Cons. Ratio (T20/B40)':<25} | {cons_ratio:.2f}x      | {bnm_targets['Cons_Ratio_T20_B40']:.2f}x      | {status_cons}")
    print("="*60)
    
    # Expert Utilization Check
    print("\nExpert Utilization (Who is handling whom?)")
    encoded = model.policy_net.encoder(state['continuous'], state['education'], 
                                      state['health'], state['housing'], 
                                      state['location'], state['sector'])
    gate_weights = model.policy_net.backbone.gating_network(encoded).detach().cpu().numpy()
    
    # Average weight per group
    b40_weights = np.mean(gate_weights[b40_idx.cpu()], axis=0)
    m40_weights = np.mean(gate_weights[m40_idx.cpu()], axis=0)
    t20_weights = np.mean(gate_weights[t20_idx.cpu()], axis=0)
    
    print(f"B40 Avg Weights: Exp1={b40_weights[0]:.2f}, Exp2={b40_weights[1]:.2f}, Exp3={b40_weights[2]:.2f}")
    print(f"M40 Avg Weights: Exp1={m40_weights[0]:.2f}, Exp2={m40_weights[1]:.2f}, Exp3={m40_weights[2]:.2f}")
    print(f"T20 Avg Weights: Exp1={t20_weights[0]:.2f}, Exp2={t20_weights[1]:.2f}, Exp3={t20_weights[2]:.2f}")

if __name__ == "__main__":
    run_comparison()
