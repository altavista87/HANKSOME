"""
Malaysia HANK - Stage 1 Enhanced (with diagnostics)
===================================================
Base model with extensive logging and validation.
Run this tonight - will save detailed progress.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Setup
DATA_DIR = Path("/Users/sir/malaysia_hank/data")
OUTPUT_DIR = Path("/Users/sir/malaysia_hank/outputs/stage1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = OUTPUT_DIR / 'training_log.txt'

def log(msg):
    """Print and log message."""
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")

# ====================================================================================
# PARAMETERS
# ====================================================================================

class MalaysiaBaseParams:
    def __init__(self):
        self.beta = 0.92
        self.sigma = 1.5
        self.r_liquid = 0.03
        self.r_illiquid = 0.055
        self.rho_e = 0.91
        self.sigma_e = 0.45
        self.n_e = 7
        self.n_liquid = 50
        self.n_illiquid = 30
        self.liquid_max = 50.0
        self.illiquid_max = 200.0
        self.a_min = 0.0
        self.epf_employee_rate = 0.11
        self.epf_employer_rate = 0.12
        
        with open(DATA_DIR / 'calibration_targets.json', 'r') as f:
            self.targets = json.load(f)

# ====================================================================================
# MODEL
# ====================================================================================

class MalaysiaHANK_Stage1(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        self.encoder = nn.Sequential(
            nn.Linear(3, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
        )
        self.policy_head = nn.Sequential(nn.Linear(64, 3), nn.Softplus())
        
    def forward(self, liquid, illiquid, income):
        liquid_norm = liquid / self.params.liquid_max
        illiquid_norm = illiquid / self.params.illiquid_max
        income_norm = income / 2.0
        
        state = torch.cat([liquid_norm, illiquid_norm, income_norm], dim=-1)
        features = self.encoder(state)
        policies_norm = self.policy_head(features)
        
        consumption = policies_norm[:, 0:1] * 2.0
        liquid_savings = policies_norm[:, 1:2] * self.params.liquid_max
        illiquid_savings = policies_norm[:, 2:3] * self.params.illiquid_max
        
        return {
            'consumption': consumption,
            'liquid_savings': liquid_savings,
            'illiquid_savings': illiquid_savings
        }
    
    def compute_budget(self, policies, state):
        liquid, illiquid, income = state['liquid'], state['illiquid'], state['income']
        c = policies['consumption']
        a_l = policies['liquid_savings']
        a_i = policies['illiquid_savings']
        
        r_l, r_i = self.params.r_liquid, self.params.r_illiquid
        coh = (1 + r_l) * liquid + (1 + r_i) * illiquid + income
        residual = c + a_l + a_i - coh
        return residual

# ====================================================================================
# TRAINER WITH DIAGNOSTICS
# ====================================================================================

class Stage1Trainer:
    def __init__(self, model, params, lr=1e-3):
        self.model = model.to(device)
        self.params = params
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.5
        )
        self.history = {
            'epoch': [], 'loss': [], 'budget_loss': [], 
            'constraint_loss': [], 'euler_loss': [],
            'mean_c': [], 'mean_a_l': [], 'mean_a_i': [],
            'time': []
        }
        self.start_time = time.time()
        
    def sample_state(self, batch_size):
        liquid = torch.rand(batch_size, 1, device=device) * self.params.liquid_max
        illiquid = torch.rand(batch_size, 1, device=device) * self.params.illiquid_max
        income = torch.exp(torch.randn(batch_size, 1, device=device) * 0.5 + 0.5)
        income = torch.clamp(income, 0.5, 3.0)
        return {'liquid': liquid, 'illiquid': illiquid, 'income': income}
    
    def compute_loss(self, batch_size=512):
        state = self.sample_state(batch_size)
        policies = self.model(state['liquid'], state['illiquid'], state['income'])
        
        # Budget
        budget_residual = self.model.compute_budget(policies, state)
        budget_loss = torch.mean(budget_residual ** 2)
        
        # Constraints
        liquid_viol = torch.relu(-policies['liquid_savings'])
        illiquid_viol = torch.relu(-policies['illiquid_savings'])
        constraint_loss = torch.mean(liquid_viol ** 2 + illiquid_viol ** 2)
        
        # Euler (simplified)
        c = policies['consumption']
        mu_t = c ** (-self.params.sigma)
        beta, r_l = self.params.beta, self.params.r_liquid
        euler_residual = mu_t - beta * (1 + r_l) * mu_t
        euler_loss = torch.mean(euler_residual ** 2)
        
        loss = 10.0 * budget_loss + 100.0 * constraint_loss + 1.0 * euler_loss
        
        # Compute means for diagnostics
        with torch.no_grad():
            mean_c = policies['consumption'].mean().item()
            mean_a_l = policies['liquid_savings'].mean().item()
            mean_a_i = policies['illiquid_savings'].mean().item()
        
        return loss, {
            'total': loss.item(), 'budget': budget_loss.item(),
            'constraint': constraint_loss.item(), 'euler': euler_loss.item(),
            'mean_c': mean_c, 'mean_a_l': mean_a_l, 'mean_a_i': mean_a_i
        }
    
    def train(self, n_epochs=1000, print_every=100):
        log("="*60)
        log("STAGE 1: TRAINING BASE HANK MODEL (Enhanced)")
        log("="*60)
        log(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        log(f"Epochs: {n_epochs}, Device: {device}")
        log("")
        
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step(metrics['total'])
            
            elapsed = time.time() - self.start_time
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['loss'].append(metrics['total'])
            self.history['budget_loss'].append(metrics['budget'])
            self.history['constraint_loss'].append(metrics['constraint'])
            self.history['euler_loss'].append(metrics['euler'])
            self.history['mean_c'].append(metrics['mean_c'])
            self.history['mean_a_l'].append(metrics['mean_a_l'])
            self.history['mean_a_i'].append(metrics['mean_a_i'])
            self.history['time'].append(elapsed)
            
            if metrics['total'] < best_loss:
                best_loss = metrics['total']
                torch.save(self.model.state_dict(), OUTPUT_DIR / 'stage1_best.pt')
            
            if epoch % print_every == 0 or epoch == n_epochs - 1:
                log(f"Epoch {epoch:4d} | Loss: {metrics['total']:.6f} | "
                    f"C: {metrics['mean_c']:.2f}, L: {metrics['mean_a_l']:.1f}, I: {metrics['mean_a_i']:.1f} | "
                    f"Time: {elapsed:.1f}s")
        
        log("")
        log("="*60)
        log("TRAINING COMPLETE")
        log(f"Best loss: {best_loss:.6f}")
        log(f"Total time: {elapsed/60:.1f} minutes")
        log("="*60)
        
        # Save history
        pd.DataFrame(self.history).to_csv(OUTPUT_DIR / 'training_history.csv', index=False)
        return self.history
    
    def plot_diagnostics(self):
        """Create comprehensive diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss curves
        ax = axes[0, 0]
        ax.semilogy(self.history['epoch'], self.history['loss'], 'b-', label='Total', linewidth=2)
        ax.semilogy(self.history['epoch'], self.history['budget_loss'], 'r--', label='Budget', alpha=0.7)
        ax.semilogy(self.history['epoch'], self.history['constraint_loss'], 'g:', label='Constraint', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Policy means over time
        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['mean_c'], 'b-', label='Consumption', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Consumption')
        ax.set_title('Consumption Convergence')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.plot(self.history['epoch'], self.history['mean_a_l'], 'r-', label='Liquid', linewidth=2)
        ax.plot(self.history['epoch'], self.history['mean_a_i'], 'g-', label='Illiquid', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Assets')
        ax.set_title('Asset Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss components final
        ax = axes[1, 0]
        final_losses = [self.history['loss'][-1], self.history['budget_loss'][-1], 
                       self.history['constraint_loss'][-1], self.history['euler_loss'][-1]]
        ax.bar(['Total', 'Budget', 'Constraint', 'Euler'], final_losses, 
               color=['blue', 'red', 'green', 'orange'])
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss Components')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Training time
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], np.array(self.history['time'])/60, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (minutes)')
        ax.set_title('Cumulative Training Time')
        ax.grid(True, alpha=0.3)
        
        # Summary statistics table
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
        STAGE 1 TRAINING SUMMARY
        ========================
        
        Training Time: {self.history['time'][-1]/60:.1f} minutes
        Total Epochs: {len(self.history['epoch'])}
        
        Final Loss: {self.history['loss'][-1]:.6f}
        Budget Loss: {self.history['budget_loss'][-1]:.6f}
        Constraint Loss: {self.history['constraint_loss'][-1]:.6f}
        
        Mean Consumption: {self.history['mean_c'][-1]:.2f}
        Mean Liquid Assets: {self.history['mean_a_l'][-1]:.1f}
        Mean Illiquid Assets: {self.history['mean_a_i'][-1]:.1f}
        
        Target Income Gini: {self.params.targets['income_gini']:.3f}
        Target HH Debt/GDP: {self.params.targets['household_debt_gdp']:.3f}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'stage1_diagnostics.png', dpi=300, bbox_inches='tight')
        log(f"Saved diagnostics: {OUTPUT_DIR / 'stage1_diagnostics.png'}")
        plt.close()

# ====================================================================================
# VALIDATION
# ====================================================================================

class Stage1Validator:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.model.eval()
    
    def validate(self, n_agents=10000):
        log("")
        log("="*60)
        log("STAGE 1 VALIDATION")
        log("="*60)
        
        with torch.no_grad():
            state = {
                'liquid': torch.rand(n_agents, 1, device=device) * self.params.liquid_max,
                'illiquid': torch.rand(n_agents, 1, device=device) * self.params.illiquid_max,
                'income': torch.exp(torch.randn(n_agents, 1, device=device) * 0.5 + 0.5)
            }
            policies = self.model(state['liquid'], state['illiquid'], state['income'])
            
            # Compute statistics
            stats = {
                'n_agents': n_agents,
                'mean_liquid': state['liquid'].mean().item(),
                'mean_illiquid': state['illiquid'].mean().item(),
                'mean_consumption': policies['consumption'].mean().item(),
                'mean_income': state['income'].mean().item(),
                'mpc_ratio': (policies['consumption'] / state['income']).mean().item(),
                'consumption_std': policies['consumption'].std().item(),
                'budget_residual_mean': self.model.compute_budget(policies, state).mean().item()
            }
            
            log(f"Steady State (n={n_agents}):")
            log(f"  Mean liquid assets:      RM {stats['mean_liquid']:,.0f}")
            log(f"  Mean illiquid (EPF):     RM {stats['mean_illiquid']:,.0f}")
            log(f"  Total wealth:            RM {stats['mean_liquid'] + stats['mean_illiquid']:,.0f}")
            log(f"  Mean consumption:        RM {stats['mean_consumption']:.2f}")
            log(f"  Consumption/Income:      {stats['mean_consumption']/stats['mean_income']:.1%}")
            log(f"  Approx MPC:              {stats['mpc_ratio']:.3f}")
            log(f"  Budget residual:         {stats['budget_residual_mean']:.4f}")
            
            log("")
            log("Data Targets:")
            log(f"  Income Gini:             {self.params.targets['income_gini']:.3f}")
            log(f"  Household Debt/GDP:      {self.params.targets['household_debt_gdp']:.3f}")
            
            # Validation checks
            log("")
            log("Validation Checks:")
            checks = [
                ("Budget residual < 0.1", abs(stats['budget_residual_mean']) < 0.1),
                ("Consumption positive", stats['mean_consumption'] > 0),
                ("Assets non-negative", stats['mean_liquid'] >= 0 and stats['mean_illiquid'] >= 0),
                ("MPC reasonable", 0.05 < stats['mpc_ratio'] < 0.8),
                ("Consumption < Income", stats['mean_consumption'] < stats['mean_income'] * 1.5)
            ]
            
            all_pass = True
            for check_name, passed in checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                log(f"  {check_name:30s} {status}")
                all_pass = all_pass and passed
            
            if all_pass:
                log("")
                log("✓ ALL VALIDATION CHECKS PASSED")
                log("  Ready for Stage 2: Add education")
            else:
                log("")
                log("⚠ SOME CHECKS FAILED - Review before proceeding")
            
            # Save
            with open(OUTPUT_DIR / 'validation_results.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            return stats

# ====================================================================================
# MAIN
# ====================================================================================

def main():
    log("="*60)
    log("MALAYSIA HANK - STAGE 1 ENHANCED")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    
    params = MalaysiaBaseParams()
    model = MalaysiaHANK_Stage1(params)
    
    trainer = Stage1Trainer(model, params, lr=1e-3)
    history = trainer.train(n_epochs=1000, print_every=100)
    trainer.plot_diagnostics()
    
    model.load_state_dict(torch.load(OUTPUT_DIR / 'stage1_best.pt'))
    validator = Stage1Validator(model, params)
    validator.validate()
    
    log("")
    log("="*60)
    log("STAGE 1 COMPLETE")
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("Outputs:")
    log(f"  - Model: {OUTPUT_DIR / 'stage1_best.pt'}")
    log(f"  - History: {OUTPUT_DIR / 'training_history.csv'}")
    log(f"  - Diagnostics: {OUTPUT_DIR / 'stage1_diagnostics.png'}")
    log(f"  - Validation: {OUTPUT_DIR / 'validation_results.json'}")
    log(f"  - Log: {LOG_FILE}")

if __name__ == "__main__":
    main()
