"""
Compute Steady State for Stage 5 Neural Network HANK Model
===========================================================

Simulates the model forward to find the stationary distribution
and aggregate steady-state variables.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from train_stage5_geography import MalaysiaHANK_Stage5, MalaysiaBaseParams
import matplotlib.pyplot as plt

# Setup
OUTPUT_DIR = Path("/Users/sir/malaysia_hank/outputs/stage5")
SIMULATION_DIR = Path("/Users/sir/malaysia_hank/outputs/stage5/simulation")
SIMULATION_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SteadyStateComputer:
    """Compute steady state by simulating the neural network model."""
    
    def __init__(self, model, params, n_agents=10000, T=1000):
        self.model = model.to(device)
        self.params = params
        self.n_agents = n_agents
        self.T = T
        self.model.eval()
        
    def initialize_population(self):
        """Initialize a population of agents."""
        return {
            'liquid': torch.rand(self.n_agents, 1, device=device) * self.params.liquid_max,
            'illiquid': torch.rand(self.n_agents, 1, device=device) * self.params.illiquid_max,
            'mortgage_debt': torch.zeros(self.n_agents, 1, device=device),
            'base_income': torch.exp(torch.randn(self.n_agents, 1, device=device) * 0.5 + 0.5),
            'education_level': torch.randint(0, 3, (self.n_agents,), device=device),
            'age': torch.randint(18, 66, (self.n_agents, 1), device=device).float(),
            'health_state': torch.multinomial(
                torch.tensor([0.75, 0.15, 0.08, 0.02], device=device),
                self.n_agents, replacement=True
            ),
            'housing_type': torch.zeros(self.n_agents, dtype=torch.long, device=device),
            'location': torch.multinomial(
                torch.ones(self.params.n_locations, device=device) / self.params.n_locations,
                self.n_agents, replacement=True
            )
        }
    
    def transition_health(self, health_state):
        """Markov transition for health state."""
        health_matrix = torch.tensor(self.params.health_transition, device=device)
        new_health = torch.zeros_like(health_state)
        
        for i in range(self.n_agents):
            h = health_state[i].item()
            probs = health_matrix[h]
            new_health[i] = torch.multinomial(probs, 1)
        
        return new_health
    
    def transition_age(self, age):
        """Age transition (with death and birth)."""
        new_age = age + 1
        
        # Death at age 65, reborn at age 20
        died = (new_age.squeeze() >= 65)
        new_age = torch.where(died.unsqueeze(1), 
                             torch.tensor(20.0, device=device), 
                             new_age)
        
        # Reset education for newborns (start with primary)
        reset_edu = died.long()
        
        return new_age, reset_edu
    
    def simulate_step(self, state):
        """Simulate one period forward."""
        with torch.no_grad():
            # Get policies from neural network
            policies = self.model(
                state['liquid'], state['illiquid'], state['mortgage_debt'],
                state['base_income'], state['education_level'], state['age'],
                state['health_state'], state['housing_type'], state['location']
            )
            
            # Compute budget components
            income = self.model.compute_income(
                state['base_income'], state['education_level'], 
                state['health_state'], state['location']
            )
            
            # Update assets
            new_liquid = policies['liquid_savings']
            new_illiquid = policies['illiquid_savings']
            
            # Housing choice (discrete)
            housing_choice = torch.argmax(policies['housing_choice'], dim=1)
            
            # Update mortgage debt based on housing choice
            # If choosing mortgage (1), add new debt; otherwise pay down
            new_mortgage = state['mortgage_debt'] * 0.95  # Pay down 5% per period
            new_mortgage = torch.where(
                (housing_choice == 1).unsqueeze(1),
                state['mortgage_debt'] + income * 0.1,  # Add 10% of income as new mortgage
                new_mortgage
            )
            new_mortgage = torch.clamp(new_mortgage, 0, 200)
            
            # Transition demographics
            new_age, reset_edu = self.transition_age(state['age'])
            new_education = torch.where(reset_edu > 0, 
                                       torch.zeros_like(state['education_level']),
                                       torch.argmax(policies['education_choice'], dim=1))
            new_health = self.transition_health(state['health_state'])
            
            # Migration choice
            migration_probs = policies['migration_choice']
            new_location = torch.multinomial(migration_probs, 1).squeeze()
            
            return {
                'liquid': torch.clamp(new_liquid, 0, self.params.liquid_max),
                'illiquid': torch.clamp(new_illiquid, 0, self.params.illiquid_max),
                'mortgage_debt': new_mortgage,
                'base_income': state['base_income'],  # Fixed for now
                'education_level': new_education,
                'age': new_age,
                'health_state': new_health,
                'housing_type': housing_choice,
                'location': new_location
            }, policies
    
    def compute_steady_state(self, burn_in=200):
        """Compute steady state by simulation."""
        print(f"Computing steady state (n={self.n_agents}, T={self.T})...")
        
        state = self.initialize_population()
        
        # Storage for aggregate variables
        aggregates = {
            'C': [], 'A_l': [], 'A_i': [], 'Debt': [],
            'Y': [], 'housing': [], 'location_dist': []
        }
        
        for t in range(self.T):
            state, policies = self.simulate_step(state)
            
            # Skip burn-in
            if t >= burn_in:
                with torch.no_grad():
                    income = self.model.compute_income(
                        state['base_income'], state['education_level'],
                        state['health_state'], state['location']
                    )
                    
                    aggregates['C'].append(policies['consumption'].mean().item())
                    aggregates['A_l'].append(state['liquid'].mean().item())
                    aggregates['A_i'].append(state['illiquid'].mean().item())
                    aggregates['Debt'].append(state['mortgage_debt'].mean().item())
                    aggregates['Y'].append(income.mean().item())
                    aggregates['housing'].append(
                        (state['housing_type'] > 0).float().mean().item()
                    )
                    
                    # Location distribution
                    loc_dist = torch.zeros(self.params.n_locations, device=device)
                    for i in range(self.params.n_locations):
                        loc_dist[i] = (state['location'] == i).float().mean()
                    aggregates['location_dist'].append(loc_dist.cpu().numpy())
            
            if t % 100 == 0:
                print(f"  Progress: {t}/{self.T}")
        
        # Compute steady state moments
        ss = {
            'mean_consumption': np.mean(aggregates['C']),
            'mean_liquid': np.mean(aggregates['A_l']),
            'mean_illiquid': np.mean(aggregates['A_i']),
            'mean_mortgage': np.mean(aggregates['Debt']),
            'mean_income': np.mean(aggregates['Y']),
            'homeownership_rate': np.mean(aggregates['housing']),
            'location_distribution': np.mean(aggregates['location_dist'], axis=0).tolist(),
            'mpc_approx': np.mean(aggregates['C']) / np.mean(aggregates['Y'])
        }
        
        return ss, state, aggregates
    
    def plot_diagnostics(self, aggregates):
        """Plot simulation diagnostics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Time series of aggregates
        ax = axes[0, 0]
        ax.plot(aggregates['C'], label='Consumption', alpha=0.7)
        ax.axhline(np.mean(aggregates['C']), color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Consumption')
        ax.set_title('Consumption Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(aggregates['A_l'], label='Liquid', alpha=0.7)
        ax.plot(aggregates['A_i'], label='Illiquid', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Assets')
        ax.set_title('Asset Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.plot(aggregates['Y'], label='Income', alpha=0.7, color='green')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Income')
        ax.set_title('Income Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Distribution histograms
        ax = axes[1, 0]
        ax.hist(aggregates['C'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Consumption')
        ax.set_ylabel('Frequency')
        ax.set_title('Consumption Distribution')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.hist(aggregates['housing'], bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax.set_xlabel('Homeownership Rate')
        ax.set_ylabel('Frequency')
        ax.set_title('Housing Distribution Over Time')
        ax.grid(True, alpha=0.3)
        
        # Summary
        ax = axes[1, 2]
        ax.axis('off')
        summary = f"""
        STEADY STATE MOMENTS
        ====================
        
        Mean Consumption: {np.mean(aggregates['C']):.2f}
        Mean Liquid Assets: {np.mean(aggregates['A_l']):.1f}
        Mean Illiquid Assets: {np.mean(aggregates['A_i']):.1f}
        Mean Mortgage: {np.mean(aggregates['Debt']):.1f}
        Mean Income: {np.mean(aggregates['Y']):.2f}
        MPC (approx): {np.mean(aggregates['C']) / np.mean(aggregates['Y']):.3f}
        Homeownership: {np.mean(aggregates['housing']):.1%}
        
        Top Locations:
        - KL Urban: {np.mean([d[4] for d in aggregates['location_dist']]):.1%}
        - Selangor Urban: {np.mean([d[2] for d in aggregates['location_dist']]):.1%}
        """
        ax.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(SIMULATION_DIR / 'steady_state_diagnostics.png', dpi=300)
        print(f"Saved: {SIMULATION_DIR / 'steady_state_diagnostics.png'}")
        plt.close()

def main():
    print("="*60)
    print("COMPUTING STEADY STATE - STAGE 5 HANK MODEL")
    print("="*60)
    
    # Load model
    params = MalaysiaBaseParams()
    model = MalaysiaHANK_Stage5(params)
    
    model_path = OUTPUT_DIR / 'stage5_best.pt'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model: {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Using untrained model (for testing)")
    
    # Compute steady state
    computer = SteadyStateComputer(model, params, n_agents=5000, T=500)
    ss, final_state, aggregates = computer.compute_steady_state(burn_in=100)
    
    # Save results
    with open(SIMULATION_DIR / 'steady_state.json', 'w') as f:
        json.dump(ss, f, indent=2)
    
    print("\n" + "="*60)
    print("STEADY STATE RESULTS")
    print("="*60)
    for key, value in ss.items():
        if isinstance(value, list):
            print(f"{key}: [array of {len(value)} elements]")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Plot
    computer.plot_diagnostics(aggregates)
    
    print("\n" + "="*60)
    print(f"Steady state saved to: {SIMULATION_DIR}")
    print("Ready for impulse response computation!")
    print("="*60)

if __name__ == "__main__":
    main()
