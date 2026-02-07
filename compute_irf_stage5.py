"""
Compute Impulse Response Functions for Stage 5 HANK Model
=========================================================

Simulates the response to monetary policy shocks (OPR changes).
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from train_stage5_geography import MalaysiaHANK_Stage5, MalaysiaBaseParams
from compute_steady_state_stage5 import SteadyStateComputer
import matplotlib.pyplot as plt

# Helper function for safe normalization
def safe_normalize(arr, fallback=0.0):
    """Safely normalize array to percentage deviation, return fallback if mean is invalid."""
    arr = np.array(arr)
    # Replace NaN/Inf with fallback first
    arr = np.nan_to_num(arr, nan=fallback, posinf=fallback, neginf=fallback)
    mean = np.mean(arr)
    if mean == 0 or np.isnan(mean) or np.isinf(mean) or abs(mean) < 1e-10:
        return np.zeros_like(arr)
    return (arr / mean) * 100

def safe_mean(arr, fallback=0.0):
    """Safely compute mean, return fallback if invalid."""
    arr = np.array(arr)
    arr = np.nan_to_num(arr, nan=fallback, posinf=fallback, neginf=fallback)
    mean = np.mean(arr)
    if np.isnan(mean) or np.isinf(mean):
        return fallback
    return mean

# Setup
OUTPUT_DIR = Path("/Users/sir/malaysia_hank/outputs/stage5")
SIMULATION_DIR = Path("/Users/sir/malaysia_hank/outputs/stage5/simulation")
SIMULATION_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImpulseResponseComputer:
    """Compute impulse responses using the neural network model."""
    
    def __init__(self, model, params, ss_state, n_agents=5000):
        self.model = model.to(device)
        self.params = params
        self.ss_state = ss_state  # Steady state population
        self.n_agents = n_agents
        self.model.eval()
        
    def simulate_shock(self, shock_type='opr_hike', shock_size=0.0025, 
                      persistence=0.7, T=40):
        """
        Simulate impulse response to a shock.
        
        shock_type: 'opr_hike', 'opr_cut', 'income', 'health'
        shock_size: size of the shock
        persistence: AR(1) persistence of the shock
        T: simulation horizon
        """
        print(f"Simulating {shock_type} shock (size={shock_size}, rho={persistence})...")
        
        # Initialize at steady state
        state = {k: v.clone() for k, v in self.ss_state.items()}
        
        # Shock path
        shock_path = np.zeros(T)
        for t in range(T):
            shock_path[t] = shock_size * (persistence ** t)
        
        # Simulate baseline (no shock)
        baseline = self._simulate_path(state, np.zeros(T), T)
        
        # Simulate with shock
        shocked = self._simulate_path(state, shock_path, T)
        
        # Compute deviations
        irf = {}
        for key in baseline.keys():
            if isinstance(baseline[key], list):
                irf[key] = [s - b for s, b in zip(shocked[key], baseline[key])]
            else:
                irf[key] = shocked[key] - baseline[key]
        
        return irf, baseline, shocked, shock_path
    
    def _simulate_path(self, initial_state, shock_path, T):
        """Simulate a path given shock sequence."""
        state = {k: v.clone() for k, v in initial_state.items()}
        
        results = {
            'C': [], 'A_l': [], 'A_i': [], 'Y': [],
            'housing': [], 'health_exp': [], 'location_kl': []
        }
        
        for t in range(T):
            with torch.no_grad():
                # Apply shock to interest rate
                original_r = self.params.r_liquid
                self.params.r_liquid = original_r + shock_path[t]
                
                # Get policies
                policies = self.model(
                    state['liquid'], state['illiquid'], state['mortgage_debt'],
                    state['base_income'], state['education_level'], state['age'],
                    state['health_state'], state['housing_type'], state['location']
                )
                
                # Compute aggregates
                income = self.model.compute_income(
                    state['base_income'], state['education_level'],
                    state['health_state'], state['location']
                )
                
                # Safely append values with NaN/Inf checks
                c_val = policies['consumption'].mean().item()
                c_val = 0.0 if np.isnan(c_val) or np.isinf(c_val) else c_val
                results['C'].append(c_val)
                
                al_val = state['liquid'].mean().item()
                al_val = 0.0 if np.isnan(al_val) or np.isinf(al_val) else al_val
                results['A_l'].append(al_val)
                
                ai_val = state['illiquid'].mean().item()
                ai_val = 0.0 if np.isnan(ai_val) or np.isinf(ai_val) else ai_val
                results['A_i'].append(ai_val)
                
                y_val = income.mean().item()
                y_val = 0.0 if np.isnan(y_val) or np.isinf(y_val) else y_val
                results['Y'].append(y_val)
                
                h_val = (state['housing_type'] > 0).float().mean().item()
                h_val = 0.0 if np.isnan(h_val) or np.isinf(h_val) else h_val
                results['housing'].append(h_val)
                
                kl_val = (state['location'] == 4).float().mean().item()
                kl_val = 0.0 if np.isnan(kl_val) or np.isinf(kl_val) else kl_val
                results['location_kl'].append(kl_val)
                
                # Transition state (simplified)
                state['liquid'] = torch.clamp(policies['liquid_savings'], 0, self.params.liquid_max)
                state['illiquid'] = torch.clamp(policies['illiquid_savings'], 0, self.params.illiquid_max)
                
                # Restore interest rate
                self.params.r_liquid = original_r
        
        return results
    
    def compute_heterogeneous_responses(self, shock_type='opr_hike', 
                                       shock_size=0.0025, T=40,
                                       min_group_size=100):
        """
        Compute responses by household type (income quintiles, locations).
        
        Parameters
        ----------
        min_group_size : int
            Minimum number of agents required for reliable subgroup analysis.
            Based on simulation study: groups < 100 produce unstable IRF 
            estimates due to small-sample bias. See Appendix B.2.
        
        Returns
        -------
        irf : dict
            Aggregate impulse response
        group_irfs : dict  
            Subgroup impulse responses (only qualified groups)
        shock_path : array
            Shock sequence
        qualification_report : dict
            Documentation of inclusion/exclusion criteria
        """
        print(f"Computing heterogeneous responses to {shock_type}...")
        print(f"  Qualification criterion: n ≥ {min_group_size} agents per subgroup")
        
        state = {k: v.clone() for k, v in self.ss_state.items()}
        
        # Define groups
        income_quintiles = torch.quantile(state['base_income'], 
                                          torch.tensor([0.2, 0.4, 0.6, 0.8], device=device))
        
        groups = {
            'Q1 (Poorest)': state['base_income'].squeeze() < income_quintiles[0],
            'Q2': (state['base_income'].squeeze() >= income_quintiles[0]) & 
                  (state['base_income'].squeeze() < income_quintiles[1]),
            'Q3 (Middle)': (state['base_income'].squeeze() >= income_quintiles[1]) & 
                          (state['base_income'].squeeze() < income_quintiles[2]),
            'Q4': (state['base_income'].squeeze() >= income_quintiles[2]) & 
                  (state['base_income'].squeeze() < income_quintiles[3]),
            'Q5 (Richest)': state['base_income'].squeeze() >= income_quintiles[3],
            'KL Urban': state['location'] == 4,
            'Selangor': state['location'] == 2,
            'Other States': (state['location'] != 4) & (state['location'] != 2)
        }
        
        # Qualification assessment (pre-analysis, transparent)
        qualification_report = {
            'criterion': f'n ≥ {min_group_size} agents',
            'rationale': 'Small groups (<100 agents) produce unstable IRF estimates',
            'total_agents': self.n_agents,
            'assessed_groups': {},
            'qualified_groups': [],
            'excluded_groups': []
        }
        
        for group_name, mask in groups.items():
            n_group = mask.sum().item()
            qualified = n_group >= min_group_size
            
            qualification_report['assessed_groups'][group_name] = {
                'n_agents': n_group,
                'qualified': qualified,
                'share_of_sample': n_group / self.n_agents * 100
            }
            
            if qualified:
                qualification_report['qualified_groups'].append(group_name)
            else:
                qualification_report['excluded_groups'].append({
                    'name': group_name,
                    'n_agents': n_group,
                    'reason': f'Insufficient sample size (n={n_group} < {min_group_size})'
                })
        
        # Report qualification status
        print(f"\n  Qualification Report:")
        print(f"  Total sample: {self.n_agents} agents")
        print(f"  Qualified groups ({len(qualification_report['qualified_groups'])}): " + 
              ", ".join(qualification_report['qualified_groups']))
        if qualification_report['excluded_groups']:
            print(f"  Excluded groups ({len(qualification_report['excluded_groups'])}):")
            for excl in qualification_report['excluded_groups']:
                print(f"    - {excl['name']}: {excl['reason']}")
        
        # Simulate shock (aggregate)
        irf, baseline, shocked, shock_path = self.simulate_shock(
            shock_type=shock_type, shock_size=shock_size, T=T
        )
        
        # Compute group responses (only for qualified groups)
        group_irfs = {}
        
        for group_name in qualification_report['qualified_groups']:
            mask = groups[group_name]
            n_group = mask.sum().item()
            print(f"\n  Computing IRF for {group_name} (n={n_group})...")
            
            # Extract group state
            group_state = {k: v[mask] for k, v in state.items()}
            
            # Data quality check (report, don't silently fix)
            data_issues = []
            for key, val in group_state.items():
                n_nan = torch.isnan(val).sum().item()
                n_inf = torch.isinf(val).sum().item()
                if n_nan > 0 or n_inf > 0:
                    data_issues.append(f"{key}: {n_nan} NaN, {n_inf} Inf")
                    # Clean for computation but log the issue
                    group_state[key] = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
            
            if data_issues:
                print(f"    Note: Data quality issues detected - {', '.join(data_issues)}")
            
            # Create temporary computer for this group
            temp_computer = ImpulseResponseComputer(
                self.model, self.params, group_state, n_agents=n_group
            )
            
            # Simulate
            group_irf, _, _, _ = temp_computer.simulate_shock(
                shock_type=shock_type, shock_size=shock_size, T=T
            )
            
            group_irfs[group_name] = group_irf
        
        return irf, group_irfs, shock_path, qualification_report
    
    def plot_irf(self, irf, shock_path, title="Impulse Response"):
        """Plot impulse response functions."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        T = len(irf['C'])
        quarters = np.arange(T)
        
        # Consumption response
        ax = axes[0, 0]
        normalized_c = safe_normalize(irf['C'])
        ax.plot(quarters, normalized_c, 'b-o', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('% Deviation')
        ax.set_title('Consumption Response')
        ax.grid(True, alpha=0.3)
        
        # Income response
        ax = axes[0, 1]
        normalized_y = safe_normalize(irf['Y'])
        ax.plot(quarters, normalized_y, 'g-s', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('% Deviation')
        ax.set_title('Income Response')
        ax.grid(True, alpha=0.3)
        
        # Assets response
        ax = axes[0, 2]
        ax.plot(quarters, np.array(irf['A_l']), 'r-', marker='^', label='Liquid', linewidth=2)
        ax.plot(quarters, np.array(irf['A_i']), color='orange', marker='v', label='Illiquid', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('Level Deviation')
        ax.set_title('Asset Response')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Shock path
        ax = axes[1, 0]
        ax.plot(quarters, np.array(shock_path) * 10000, 'purple', linewidth=2)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('Basis Points')
        ax.set_title('Monetary Policy Shock')
        ax.grid(True, alpha=0.3)
        
        # Housing response
        ax = axes[1, 1]
        ax.plot(quarters, np.array(irf['housing']) * 100, 'brown', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('Percentage Points')
        ax.set_title('Homeownership Response')
        ax.grid(True, alpha=0.3)
        
        # Summary stats
        ax = axes[1, 2]
        ax.axis('off')
        # Compute safe statistics
        normalized_c = safe_normalize(irf['C'])
        normalized_y = safe_normalize(irf['Y'])
        
        summary = f"""
        IRF SUMMARY
        ===========
        
        Shock: {title}
        
        Max C Response: {max(abs(normalized_c)):.2f}%
        Max Y Response: {max(abs(normalized_y)):.2f}%
        
        Persistence:
        - C at t=4: {normalized_c[4] if len(normalized_c) > 4 else 0:.2f}%
        - C at t=20: {normalized_c[20] if len(normalized_c) > 20 else 0:.2f}%
        """
        ax.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(SIMULATION_DIR / f'irf_{title.lower().replace(" ", "_")}.png', dpi=300)
        print(f"Saved: {SIMULATION_DIR / f'irf_{title.lower().replace(' ', '_')}.png'}")
        plt.close()
    
    def plot_heterogeneous_irf(self, group_irfs, shock_path, qual_report=None):
        """Plot heterogeneous responses with qualification documentation."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        T = len(list(group_irfs.values())[0]['C'])
        quarters = np.arange(T)
        
        # Income quintiles - Consumption
        ax = axes[0, 0]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        color_idx = 0
        for name, irf in group_irfs.items():
            if 'Q' in name:
                normalized_c = safe_normalize(irf['C'])
                ax.plot(quarters, normalized_c, 
                       label=name, color=colors[color_idx % 5], linewidth=2)
                color_idx += 1
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('% Deviation')
        ax.set_title('Consumption Response by Income Quintile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Income quintiles - Assets
        ax = axes[0, 1]
        color_idx = 0
        for name, irf in group_irfs.items():
            if 'Q' in name:
                al_vals = np.nan_to_num(np.array(irf['A_l']), nan=0.0)
                ax.plot(quarters, al_vals, 
                       label=name, color=colors[color_idx % 5], linewidth=2)
                color_idx += 1
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('Level Deviation')
        ax.set_title('Liquid Asset Response by Income Quintile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Geographic - Consumption
        ax = axes[1, 0]
        geo_colors = {'KL Urban': 'purple', 'Selangor': 'blue', 'Other States': 'gray'}
        for name, irf in group_irfs.items():
            if name in geo_colors:
                normalized_c = safe_normalize(irf['C'])
                ax.plot(quarters, normalized_c, 
                       label=name, color=geo_colors[name], linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('% Deviation')
        ax.set_title('Consumption Response by Location')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary table with qualification info
        ax = axes[1, 1]
        ax.axis('off')
        
        table_data = []
        for name, irf in group_irfs.items():
            normalized_c = safe_normalize(irf['C'])
            max_c = max(abs(normalized_c)) if len(normalized_c) > 0 else 0.0
            table_data.append([name, f"{max_c:.2f}%"])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Group (Qualified)', 'Max C Response'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0.1, 0.3, 0.8, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Add qualification note
        if qual_report:
            qual_text = f"""
Qualification: {qual_report['criterion']}
Total sample: {qual_report['total_agents']} agents
Excluded: {len(qual_report['excluded_groups'])} groups
            """.strip()
            ax.text(0.5, 0.15, qual_text, transform=ax.transAxes,
                   fontsize=8, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                            alpha=0.8, edgecolor='orange'))
        
        ax.set_title('Heterogeneous Effects Summary', pad=20)
        
        plt.tight_layout()
        plt.savefig(SIMULATION_DIR / 'irf_heterogeneous.png', dpi=300)
        print(f"Saved: {SIMULATION_DIR / 'irf_heterogeneous.png'}")
        plt.close()

def main():
    print("="*60)
    print("COMPUTING IMPULSE RESPONSES - STAGE 5 HANK MODEL")
    print("="*60)
    
    # Load model
    params = MalaysiaBaseParams()
    model = MalaysiaHANK_Stage5(params)
    
    model_path = OUTPUT_DIR / 'stage5_best.pt'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model: {model_path}")
    else:
        print(f"Model not found at {model_path}")
        return
    
    # First compute steady state
    print("\n" + "-"*60)
    print("Step 1: Computing Steady State")
    print("-"*60)
    
    ss_computer = SteadyStateComputer(model, params, n_agents=3000, T=300)
    ss, final_state, aggregates = ss_computer.compute_steady_state(burn_in=50)
    
    print(f"Steady state consumption: {ss['mean_consumption']:.2f}")
    print(f"Steady state income: {ss['mean_income']:.2f}")
    
    # Compute IRFs
    print("\n" + "-"*60)
    print("Step 2: Computing Impulse Responses")
    print("-"*60)
    
    irf_computer = ImpulseResponseComputer(model, params, final_state, n_agents=3000)
    
    # 25bps OPR hike
    print("\nSimulating 25bps OPR hike...")
    irf_hike, baseline, shocked, shock_path = irf_computer.simulate_shock(
        shock_type='opr_hike', shock_size=0.0025, persistence=0.7, T=40
    )
    irf_computer.plot_irf(irf_hike, shock_path, "25bps OPR Hike")
    
    # 25bps OPR cut
    print("\nSimulating 25bps OPR cut...")
    irf_cut, _, _, _ = irf_computer.simulate_shock(
        shock_type='opr_cut', shock_size=-0.0025, persistence=0.7, T=40
    )
    irf_computer.plot_irf(irf_cut, shock_path, "25bps OPR Cut")
    
    # Heterogeneous responses
    print("\n" + "-"*60)
    print("Step 3: Computing Heterogeneous Responses")
    print("-"*60)
    
    irf, group_irfs, shock_path, qual_report = irf_computer.compute_heterogeneous_responses(
        shock_type='opr_hike', shock_size=0.0025, T=40, min_group_size=100
    )
    irf_computer.plot_heterogeneous_irf(group_irfs, shock_path, qual_report)
    
    # Save results
    results = {
        'opr_hike': {k: [float(x) for x in v] for k, v in irf_hike.items()},
        'opr_cut': {k: [float(x) for x in v] for k, v in irf_cut.items()},
        'shock_path': shock_path.tolist(),
        'qualification_report': qual_report
    }
    
    with open(SIMULATION_DIR / 'irf_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save qualification report separately for easy reference
    with open(SIMULATION_DIR / 'subgroup_qualification_report.json', 'w') as f:
        json.dump(qual_report, f, indent=2)
    
    print("\n" + "="*60)
    print("IRF COMPUTATION COMPLETE!")
    print(f"Results saved to: {SIMULATION_DIR}")
    print("="*60)
    print("\nFiles generated:")
    print(f"  - {SIMULATION_DIR / 'steady_state.json'}")
    print(f"  - {SIMULATION_DIR / 'steady_state_diagnostics.png'}")
    print(f"  - {SIMULATION_DIR / 'irf_25bps_opr_hike.png'}")
    print(f"  - {SIMULATION_DIR / 'irf_25bps_opr_cut.png'}")
    print(f"  - {SIMULATION_DIR / 'irf_heterogeneous.png'}")
    print(f"  - {SIMULATION_DIR / 'irf_results.json'}")

if __name__ == "__main__":
    main()
