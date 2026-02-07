"""
Malaysia HANK - Stage 1: Base Model
====================================
Simple 2-asset HANK model with Malaysian calibration.

This is the FOUNDATION. Get this working first before adding complexity.
State space: ~700 states (manageable)
Training time: 10-30 minutes
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# Setup
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/stage1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ====================================================================================
# STAGE 1: BASE PARAMETERS (Simplified)
# ====================================================================================

class MalaysiaBaseParams:
    """Simplified parameters for Stage 1 base model."""
    
    def __init__(self):
        # Preferences (will be calibrated)
        self.beta = 0.92           # Discount factor
        self.sigma = 1.5           # Risk aversion (CRRA)
        
        # Interest rates (from data)
        self.r_liquid = 0.03       # Bank deposit rate
        self.r_illiquid = 0.055    # EPF return (5.5%)
        
        # Income process (Rouwenhorst)
        self.rho_e = 0.91          # Persistence
        self.sigma_e = 0.45        # Volatility
        self.n_e = 7               # Income grid points
        
        # Asset grids
        self.n_liquid = 50
        self.n_illiquid = 30
        self.liquid_max = 50.0
        self.illiquid_max = 200.0
        
        # Borrowing constraint
        self.a_min = 0.0
        
        # EPF system
        self.epf_employee_rate = 0.11
        self.epf_employer_rate = 0.12
        
        # Load calibration targets
        with open(DATA_DIR / 'calibration_targets.json', 'r') as f:
            self.targets = json.load(f)


# ====================================================================================
# STAGE 1: NEURAL NETWORK (Simple)
# ====================================================================================

class MalaysiaHANK_Stage1(nn.Module):
    """
    Stage 1: Base HANK model.
    
    State: (liquid_assets, illiquid_assets, income_shock)
    Policy: (consumption, liquid_savings, illiquid_savings)
    """
    
    def __init__(self, params: MalaysiaBaseParams):
        super().__init__()
        self.params = params
        
        # State encoder: (liquid, illiquid, income) -> features
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        
        # Policy head: features -> (c, a_l, a_i)
        self.policy_head = nn.Sequential(
            nn.Linear(64, 3),
            nn.Softplus()  # Ensure positive consumption and savings
        )
        
    def forward(self, liquid: torch.Tensor, illiquid: torch.Tensor, 
                income: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            liquid: [batch, 1] liquid assets
            illiquid: [batch, 1] illiquid assets (EPF)
            income: [batch, 1] labor income
            
        Returns:
            Dictionary with consumption and asset policies
        """
        # Normalize inputs (helps training)
        liquid_norm = liquid / self.params.liquid_max
        illiquid_norm = illiquid / self.params.illiquid_max
        income_norm = income / 2.0  # Normalize by approx mean income
        
        # Encode state
        state = torch.cat([liquid_norm, illiquid_norm, income_norm], dim=-1)
        features = self.encoder(state)
        
        # Get policies (normalized)
        policies_norm = self.policy_head(features)
        
        # Denormalize
        consumption = policies_norm[:, 0:1] * 2.0  # Scale to income units
        liquid_savings = policies_norm[:, 1:2] * self.params.liquid_max
        illiquid_savings = policies_norm[:, 2:3] * self.params.illiquid_max
        
        return {
            'consumption': consumption,
            'liquid_savings': liquid_savings,
            'illiquid_savings': illiquid_savings
        }
    
    def compute_budget(self, policies: Dict, state: Dict) -> torch.Tensor:
        """Compute budget constraint residual."""
        liquid = state['liquid']
        illiquid = state['illiquid']
        income = state['income']
        
        c = policies['consumption']
        a_l = policies['liquid_savings']
        a_i = policies['illiquid_savings']
        
        # Cash on hand
        r_l = self.params.r_liquid
        r_i = self.params.r_illiquid
        coh = (1 + r_l) * liquid + (1 + r_i) * illiquid + income
        
        # Budget residual (should be 0)
        residual = c + a_l + a_i - coh
        return residual


# ====================================================================================
# STAGE 1: TRAINER
# ====================================================================================

class Stage1Trainer:
    """Trainer for Stage 1 base model."""
    
    def __init__(self, model: MalaysiaHANK_Stage1, params: MalaysiaBaseParams, 
                 lr: float = 1e-3):
        self.model = model.to(device)
        self.params = params
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.5
        )
        self.history = {'loss': [], 'budget_loss': [], 'euler_loss': []}
        
    def sample_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random states for training."""
        # Uniform sampling over state space
        liquid = torch.rand(batch_size, 1, device=device) * self.params.liquid_max
        illiquid = torch.rand(batch_size, 1, device=device) * self.params.illiquid_max
        
        # Log-normal-ish income distribution
        income = torch.exp(torch.randn(batch_size, 1, device=device) * 0.5 + 0.5)
        income = torch.clamp(income, 0.5, 3.0)  # Reasonable income range
        
        return {
            'liquid': liquid,
            'illiquid': illiquid,
            'income': income
        }
    
    def compute_loss(self, batch_size: int = 512) -> Tuple[torch.Tensor, Dict]:
        """Compute physics-informed loss."""
        # Sample states
        state = self.sample_state(batch_size)
        
        # Forward pass
        policies = self.model(state['liquid'], state['illiquid'], state['income'])
        
        # 1. Budget constraint (hard constraint - high weight)
        budget_residual = self.model.compute_budget(policies, state)
        budget_loss = torch.mean(budget_residual ** 2)
        
        # 2. Borrowing constraints
        liquid_violation = torch.relu(-policies['liquid_savings'])
        illiquid_violation = torch.relu(-policies['illiquid_savings'])
        constraint_loss = torch.mean(liquid_violation ** 2 + illiquid_violation ** 2)
        
        # 3. Euler equation (simplified - assume next period similar)
        # This is a rough approximation for fast training
        # Full model would compute E[c_t+1^(-sigma)]
        beta = self.params.beta
        sigma = self.params.sigma
        r_l = self.params.r_liquid
        
        c = policies['consumption']
        mu_t = c ** (-sigma)
        
        # Very simplified: assume next period marginal utility similar
        # (Proper implementation would require simulation)
        euler_residual = mu_t - beta * (1 + r_l) * mu_t
        euler_loss = torch.mean(euler_residual ** 2)
        
        # Total loss
        loss = 10.0 * budget_loss + 100.0 * constraint_loss + 1.0 * euler_loss
        
        metrics = {
            'total': loss.item(),
            'budget': budget_loss.item(),
            'constraint': constraint_loss.item(),
            'euler': euler_loss.item()
        }
        
        return loss, metrics
    
    def train(self, n_epochs: int = 1000, print_every: int = 100):
        """Train the model."""
        print("="*60)
        print("STAGE 1: TRAINING BASE HANK MODEL")
        print("="*60)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training epochs: {n_epochs}")
        print(f"Device: {device}")
        print()
        
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            loss, metrics = self.compute_loss()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step(metrics['total'])
            
            # Record history
            self.history['loss'].append(metrics['total'])
            self.history['budget_loss'].append(metrics['budget'])
            self.history['euler_loss'].append(metrics['euler'])
            
            # Save best model
            if metrics['total'] < best_loss:
                best_loss = metrics['total']
                torch.save(self.model.state_dict(), OUTPUT_DIR / 'stage1_best_model.pt')
            
            # Print progress
            if epoch % print_every == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch:4d}/{n_epochs}")
                print(f"  Loss: {metrics['total']:.6f}  "
                      f"(Budget: {metrics['budget']:.6f}, "
                      f"Constraint: {metrics['constraint']:.6f})")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best loss: {best_loss:.6f}")
        print(f"Model saved to: {OUTPUT_DIR / 'stage1_best_model.pt'}")
        
        return self.history
    
    def plot_training(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss over time
        ax = axes[0]
        ax.plot(self.history['loss'], label='Total Loss', linewidth=2)
        ax.plot(self.history['budget_loss'], label='Budget Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Final loss breakdown
        ax = axes[1]
        losses = {
            'Total': self.history['loss'][-1],
            'Budget': self.history['budget_loss'][-1],
            'Euler': self.history['euler_loss'][-1]
        }
        ax.bar(losses.keys(), losses.values(), color=['blue', 'orange', 'green'])
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss Components')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'stage1_training.png', dpi=300)
        print(f"Training plot saved to: {OUTPUT_DIR / 'stage1_training.png'}")
        plt.close()


# ====================================================================================
# STAGE 1: VALIDATION
# ====================================================================================

class Stage1Validator:
    """Validate trained Stage 1 model."""
    
    def __init__(self, model: MalaysiaHANK_Stage1, params: MalaysiaBaseParams):
        self.model = model
        self.params = params
        self.model.eval()
    
    def compute_steady_state(self, n_agents: int = 10000) -> Dict[str, float]:
        """Compute steady state statistics."""
        print("\nComputing steady state...")
        
        with torch.no_grad():
            # Sample agents
            state = {
                'liquid': torch.rand(n_agents, 1, device=device) * self.params.liquid_max,
                'illiquid': torch.rand(n_agents, 1, device=device) * self.params.illiquid_max,
                'income': torch.exp(torch.randn(n_agents, 1, device=device) * 0.5 + 0.5)
            }
            
            # Get policies
            policies = self.model(state['liquid'], state['illiquid'], state['income'])
            
            # Compute statistics
            stats = {
                'mean_liquid': state['liquid'].mean().item(),
                'mean_illiquid': state['illiquid'].mean().item(),
                'mean_consumption': policies['consumption'].mean().item(),
                'mean_income': state['income'].mean().item(),
                'mpc_approx': (policies['consumption'] / state['income']).mean().item()
            }
            
            return stats
    
    def print_validation(self):
        """Print validation results."""
        stats = self.compute_steady_state()
        
        print("\n" + "="*60)
        print("STAGE 1 VALIDATION RESULTS")
        print("="*60)
        
        print("\nSteady State Statistics:")
        print(f"  Mean liquid assets:      RM {stats['mean_liquid']:,.0f}")
        print(f"  Mean illiquid (EPF):     RM {stats['mean_illiquid']:,.0f}")
        print(f"  Total wealth:            RM {stats['mean_liquid'] + stats['mean_illiquid']:,.0f}")
        print(f"  Mean consumption:        RM {stats['mean_consumption']:,.2f}")
        print(f"  Consumption/Income:      {stats['mean_consumption']/stats['mean_income']:.2%}")
        print(f"  Approx MPC:              {stats['mpc_approx']:.3f}")
        
        print("\nComparison to Data Targets:")
        print(f"  Target income Gini:      {self.params.targets['income_gini']:.3f}")
        print(f"  Target HH debt/GDP:      {self.params.targets['household_debt_gdp']:.3f}")
        
        print("\n✓ Stage 1 base model validation complete!")
        print(f"  Next step: Run 'train_stage2_education.py' to add education")
        
        # Save stats
        with open(OUTPUT_DIR / 'stage1_validation.json', 'w') as f:
            json.dump(stats, f, indent=2)


# ====================================================================================
# MAIN
# ====================================================================================

def main():
    """Run Stage 1 training and validation."""
    
    print("="*60)
    print("MALAYSIA HANK - STAGE 1: BASE MODEL")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create parameters
    params = MalaysiaBaseParams()
    
    # Create model
    model = MalaysiaHANK_Stage1(params)
    
    # Train
    trainer = Stage1Trainer(model, params, lr=1e-3)
    history = trainer.train(n_epochs=1000, print_every=100)
    
    # Plot training
    trainer.plot_training()
    
    # Load best model for validation
    model.load_state_dict(torch.load(OUTPUT_DIR / 'stage1_best_model.pt'))
    
    # Validate
    validator = Stage1Validator(model, params)
    validator.print_validation()
    
    print("\n" + "="*60)
    print("STAGE 1 COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Next steps:")
    print("  1. Review results above")
    print("  2. Check training plot: outputs/stage1/stage1_training.png")
    print("  3. If satisfied, run: python train_stage2_education.py")
    print()


if __name__ == "__main__":
    main()
