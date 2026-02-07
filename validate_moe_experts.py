"""
Validation Script for Malaysia HANK MoE Architecture
===================================================
Verifies if the specialized experts are correctly capturing the 
Marginal Propensity to Consume (MPC) across the wealth distribution.
"""

import torch
import numpy as np
from malaysia_deep_hank_architecture import MalaysiaExtendedParams, MalaysiaExtendedHANK

def calculate_mpc(model, state, prices, eps=0.1):
    """
    Calculates MPC by slightly increasing liquid assets.
    MPC = (C(a+eps) - C(a)) / eps
    """
    # Original policy
    policies = model.compute_economic_policies(state, prices)
    c_orig = policies['consumption']
    
    # State with slightly more liquid assets
    state_plus = {k: v.clone() for k, v in state.items()}
    state_plus['continuous'] = state['continuous'].clone()
    state_plus['continuous'][:, 0] += eps
    
    # New policy
    policies_plus = model.compute_economic_policies(state_plus, prices)
    c_plus = policies_plus['consumption']
    
    mpc = (c_plus - c_orig) / eps
    return mpc.mean().item(), c_orig.mean().item()

def validate():
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    params = MalaysiaExtendedParams()
    model = MalaysiaExtendedHANK(params).to(device)
    
    # Load checkpoint
    checkpoint = torch.load("hank_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded from checkpoint. Starting validation...")
    
    prices = {'r_liquid': 0.03, 'r_illiquid': 0.05, 'r_mortgage': 0.045, 'w': 1.0}
    
    # Define group scenarios
    # Continuous: (liquid, illiquid, human_capital, age, mortgage)
    groups = {
        "B40 (Liquidity Constrained)": {
            "continuous": [0.1, 0.5, 0.8, 25.0, 0.0],
            "education": 1, "health": 0, "housing": 0, "location": 0, "sector": 2
        },
        "M40 (Middle Class)": {
            "continuous": [10.0, 50.0, 1.2, 40.0, 45.0],
            "education": 2, "health": 0, "housing": 1, "location": 0, "sector": 0
        },
        "T20 (Asset Rich)": {
            "continuous": [100.0, 500.0, 2.5, 50.0, 0.0],
            "education": 2, "health": 0, "housing": 2, "location": 0, "sector": 0
        }
    }
    
    results = []
    
    for name, data in groups.items():
        # Prepare state tensor
        state = {
            "continuous": torch.tensor(data["continuous"], device=device, dtype=torch.float32).unsqueeze(0),
            "education": torch.tensor([data["education"]], device=device, dtype=torch.long),
            "health": torch.tensor([data["health"]], device=device, dtype=torch.long),
            "housing": torch.tensor([data["housing"]], device=device, dtype=torch.long),
            "location": torch.tensor([data["location"]], device=device, dtype=torch.long),
            "sector": torch.tensor([data["sector"]], device=device, dtype=torch.long)
        }
        
        # Calculate MPC and Consumption
        mpc, cons = calculate_mpc(model, state, prices)
        
        # Check Gating Network Weights
        encoded = model.policy_net.encoder(
            state['continuous'], state['education'], state['health'], 
            state['housing'], state['location'], state['sector']
        )
        gate_weights = model.policy_net.backbone.gating_network(encoded)
        weights = gate_weights.detach().cpu().numpy()[0]
        
        results.append({
            "Group": name,
            "Consumption": round(cons, 4),
            "MPC": round(mpc, 4),
            "Gating Weights (Exp 1, 2, 3)": [round(w, 3) for w in weights]
        })

    # Print results table
    print("\n" + "="*80)
    print(f"{ 'Group':<30} | { 'Cons.':<10} | { 'MPC':<10} | {'Expert Preference'}")
    print("-" * 80)
    for res in results:
        weights = res["Gating Weights (Exp 1, 2, 3)"]
        expert = np.argmax(weights) + 1
        print(f"{res['Group']:<30} | {res['Consumption']:<10} | {res['MPC']:<10} | Expert {expert} ({weights})")
    print("="*80)
    
    # Final Verdict
    b40_mpc = results[0]["MPC"]
    t20_mpc = results[2]["MPC"]
    
    print("\nEconomic Sanity Check:")
    if b40_mpc > t20_mpc:
        print(f"✓ SUCCESS: B40 MPC ({b40_mpc}) is higher than T20 MPC ({t20_mpc}).")
        print("  The 'Three Advisors' strategy is successfully capturing the B40 kink.")
    else:
        print(f"⚠ WARNING: B40 MPC ({b40_mpc}) is NOT higher than T20 MPC ({t20_mpc}).")
        print("  The model may need more training or adjustment to the Gating Network.")

if __name__ == "__main__":
    validate()
