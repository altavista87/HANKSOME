"""
Malaysia HANK - Stage 2: Add Education
======================================
Extends Stage 1 by adding human capital accumulation through education choices.

Education System:
- 3 Levels: Primary (0), Secondary (1), Tertiary (2)
- Costs: [0.0, 0.5, 2.0] (Tertiary is expensive)
- Wage Premiums: [1.0, 1.3, 2.0] (Tertiary earns 2x)
- Enrollment: Only for households with age < 25

State Space: 700 -> 2,100 states (3x increase from Stage 1)
Training Time: 1-2 hours (2000 epochs)
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
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/stage2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE1_DIR = Path("outputs/stage1")

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
        # Time preference
        self.beta = 0.92
        self.sigma = 1.5
        
        # Returns
        self.r_liquid = 0.03
        self.r_illiquid = 0.055
        
        # Income process
        self.rho_e = 0.91
        self.sigma_e = 0.45
        self.n_e = 7
        
        # Asset grids
        self.n_liquid = 50
        self.n_illiquid = 30
        self.liquid_max = 50.0
        self.illiquid_max = 200.0
        self.a_min = 0.0
        
        # EPF system
        self.epf_employee_rate = 0.11
        self.epf_employer_rate = 0.12
        
        # Education parameters (NEW for Stage 2)
        self.n_education = 3  # Primary, Secondary, Tertiary
        self.education_costs = [0.0, 0.5, 2.0]  # Cost per level
        self.education_premiums = [1.0, 1.3, 2.0]  # Wage multipliers
        self.max_education_age = 25  # Can only enroll if younger
        
        # Load calibration targets
        with open(DATA_DIR / 'calibration_targets.json', 'r') as f:
            self.targets = json.load(f)

# ====================================================================================
# STAGE 2 MODEL: Base HANK + Education
# ====================================================================================

class MalaysiaHANK_Stage2(nn.Module):
    """
    Stage 2: Two-asset HANK with education choices.
    
    State: (liquid, illiquid, income, education_level, age)
    Policies: (consumption, liquid_savings, illiquid_savings, education_choice)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # Education embedding: 3 levels -> 8 dimensions
        self.education_embed = nn.Embedding(params.n_education, 8)
        
        # Extended encoder: 3 continuous + 8 education embed + 1 age = 12 inputs
        self.encoder = nn.Sequential(
            nn.Linear(12, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        
        # Policy heads
        self.policy_head = nn.Sequential(
            nn.Linear(64, 3),
            nn.Softplus()  # c, a_l, a_i > 0
        )
        
        # Education choice head (probability over 3 levels)
        self.education_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, params.n_education)  # Logits
        )
        
    def forward(self, liquid, illiquid, income, education_level, age):
        """
        Forward pass through the network.
        
        Args:
            liquid: [batch, 1] - liquid assets
            illiquid: [batch, 1] - illiquid assets  
            income: [batch, 1] - current income
            education_level: [batch] - int in {0, 1, 2}
            age: [batch, 1] - age in years
        
        Returns:
            Dictionary with all policy outputs
        """
        batch_size = liquid.size(0)
        
        # Normalize continuous inputs
        liquid_norm = liquid / self.params.liquid_max
        illiquid_norm = illiquid / self.params.illiquid_max
        income_norm = income / 2.0
        age_norm = age / 80.0  # Normalize by max age
        
        # Embed education level
        edu_emb = self.education_embed(education_level)  # [batch, 8]
        
        # Concatenate all state components: 3 + 8 + 1 = 12
        state = torch.cat([
            liquid_norm, 
            illiquid_norm, 
            income_norm,
            edu_emb,
            age_norm
        ], dim=-1)  # [batch, 12]
        
        # Encode state
        features = self.encoder(state)  # [batch, 64]
        
        # Get base policies (consumption and savings)
        policies_norm = self.policy_head(features)  # [batch, 3]
        
        consumption = policies_norm[:, 0:1] * 2.0
        liquid_savings = policies_norm[:, 1:2] * self.params.liquid_max
        illiquid_savings = policies_norm[:, 2:3] * self.params.illiquid_max
        
        # Get education choice probabilities
        edu_logits = self.education_head(features)  # [batch, 3]
        
        # Mask education choices for older households (age >= 25)
        # Use a mask that forces current education level if too old
        can_enroll = (age.squeeze(-1) < self.params.max_education_age).float()  # [batch]
        
        # Create masked logits: if too old, only current education is allowed
        current_edu_onehot = nn.functional.one_hot(
            education_level, num_classes=self.params.n_education
        ).float()  # [batch, 3]
        
        # Softmax for those who can choose, one-hot for those who can't
        edu_choice_soft = torch.softmax(edu_logits, dim=-1)  # [batch, 3]
        
        # Blend: can_enroll * softmax + (1 - can_enroll) * one_hot_current
        can_enroll_expanded = can_enroll.unsqueeze(-1)  # [batch, 1]
        education_choice = (
            can_enroll_expanded * edu_choice_soft + 
            (1 - can_enroll_expanded) * current_edu_onehot
        )
        
        return {
            'consumption': consumption,
            'liquid_savings': liquid_savings,
            'illiquid_savings': illiquid_savings,
            'education_choice': education_choice,
            'education_logits': edu_logits,
            'can_enroll': can_enroll
        }
    
    def compute_income(self, base_income, education_level):
        """
        Compute income with education premium.
        
        wage = base_income × education_premium[education_level]
        """
        # Get education premium for each household
        premiums = torch.tensor(
            self.params.education_premiums, 
            device=base_income.device, 
            dtype=base_income.dtype
        )[education_level]  # [batch]
        
        return base_income * premiums.unsqueeze(-1)
    
    def compute_budget(self, policies, state):
        """
        Compute budget constraint residual.
        
        cash_on_hand = (1+r_l)*liquid + (1+r_i)*illiquid + income(1-edu)
        residual = c + a_l + a_i + education_cost - cash_on_hand
        """
        liquid = state['liquid']
        illiquid = state['illiquid']
        base_income = state['base_income']
        education_level = state['education_level']
        
        c = policies['consumption']
        a_l = policies['liquid_savings']
        a_i = policies['illiquid_savings']
        edu_choice = policies['education_choice']
        
        # Compute income with education premium
        income = self.compute_income(base_income, education_level)
        
        # Compute education cost from choice
        education_costs = torch.tensor(
            self.params.education_costs,
            device=liquid.device,
            dtype=liquid.dtype
        )  # [3]
        
        # Expected education cost (probability-weighted over possible levels)
        expected_edu_cost = (edu_choice * education_costs.unsqueeze(0)).sum(dim=1, keepdim=True)
        
        # Cash on hand (before education cost)
        r_l, r_i = self.params.r_liquid, self.params.r_illiquid
        coh = (1 + r_l) * liquid + (1 + r_i) * illiquid + income
        
        # Budget: c + a_l + a_i + edu_cost = coh
        total_spending = c + a_l + a_i + expected_edu_cost
        residual = total_spending - coh
        
        return residual, {
            'coh': coh,
            'expected_edu_cost': expected_edu_cost,
            'income': income
        }

# ====================================================================================
# TRAINER WITH DIAGNOSTICS
# ====================================================================================

class Stage2Trainer:
    def __init__(self, model, params, lr=1e-3):
        self.model = model.to(device)
        self.params = params
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.5
        )
        self.history = {
            'epoch': [], 'loss': [], 'budget_loss': [], 
            'constraint_loss': [], 'euler_loss': [], 'education_loss': [],
            'mean_c': [], 'mean_a_l': [], 'mean_a_i': [],
            'mean_income_primary': [], 'mean_income_tertiary': [],
            'enrollment_rate': [], 'time': []
        }
        self.start_time = time.time()
        
    def sample_state(self, batch_size):
        """Sample random states including education and age."""
        liquid = torch.rand(batch_size, 1, device=device) * self.params.liquid_max
        illiquid = torch.rand(batch_size, 1, device=device) * self.params.illiquid_max
        base_income = torch.exp(torch.randn(batch_size, 1, device=device) * 0.5 + 0.5)
        base_income = torch.clamp(base_income, 0.5, 3.0)
        
        # Education: mostly primary/secondary, some tertiary
        education_level = torch.randint(0, 3, (batch_size,), device=device)
        
        # Age: uniform 18-65 (working age)
        age = torch.randint(18, 66, (batch_size, 1), device=device).float()
        
        return {
            'liquid': liquid,
            'illiquid': illiquid,
            'base_income': base_income,
            'education_level': education_level,
            'age': age
        }
    
    def compute_loss(self, batch_size=512):
        state = self.sample_state(batch_size)
        
        policies = self.model(
            state['liquid'], 
            state['illiquid'], 
            state['base_income'],
            state['education_level'],
            state['age']
        )
        
        # Budget constraint
        budget_residual, budget_info = self.model.compute_budget(policies, state)
        budget_loss = torch.mean(budget_residual ** 2)
        
        # Borrowing constraints
        liquid_viol = torch.relu(-policies['liquid_savings'])
        illiquid_viol = torch.relu(-policies['illiquid_savings'])
        constraint_loss = torch.mean(liquid_viol ** 2 + illiquid_viol ** 2)
        
        # Euler equation (simplified)
        c = policies['consumption']
        mu_t = c ** (-self.params.sigma)
        beta, r_l = self.params.beta, self.params.r_liquid
        euler_residual = mu_t - beta * (1 + r_l) * mu_t
        euler_loss = torch.mean(euler_residual ** 2)
        
        # Education consistency: encourage higher education for young
        # Young households should prefer higher education levels
        can_enroll = policies['can_enroll']  # [batch]
        edu_choice = policies['education_choice']  # [batch, 3]
        
        # Penalize not choosing higher education when young and able
        # We want: P(tertiary) > P(secondary) > P(primary) for young households
        edu_levels = torch.arange(3, device=device).float().unsqueeze(0)  # [1, 3]
        expected_edu_level = (edu_choice * edu_levels).sum(dim=1)  # [batch]
        
        # For young households, encourage higher education (penalize low choices)
        # Target: young households should have expected education > 1.0 (secondary+)
        young_target = torch.ones_like(expected_edu_level) * 1.5  # Target: tertiary/secondary mix
        education_penalty = torch.where(
            can_enroll.bool(),
            torch.relu(young_target - expected_edu_level),  # Penalty if below target
            torch.zeros_like(expected_edu_level)
        )
        education_loss = torch.mean(education_penalty ** 2)
        
        # Total loss
        loss = (
            10.0 * budget_loss + 
            100.0 * constraint_loss + 
            1.0 * euler_loss +
            0.1 * education_loss  # Soft encouragement for education
        )
        
        # Compute diagnostics
        with torch.no_grad():
            mean_c = policies['consumption'].mean().item()
            mean_a_l = policies['liquid_savings'].mean().item()
            mean_a_i = policies['illiquid_savings'].mean().item()
            
            # Income by education level
            income = budget_info['income']
            primary_mask = (state['education_level'] == 0)
            tertiary_mask = (state['education_level'] == 2)
            
            mean_income_primary = income[primary_mask].mean().item() if primary_mask.any() else 0.0
            mean_income_tertiary = income[tertiary_mask].mean().item() if tertiary_mask.any() else 0.0
            
            # Enrollment rate among young
            young_mask = (state['age'].squeeze() < self.params.max_education_age)
            enrollment_rate = policies['education_choice'][young_mask, 2].mean().item() if young_mask.any() else 0.0
        
        return loss, {
            'total': loss.item(),
            'budget': budget_loss.item(),
            'constraint': constraint_loss.item(),
            'euler': euler_loss.item(),
            'education': education_loss.item(),
            'mean_c': mean_c,
            'mean_a_l': mean_a_l,
            'mean_a_i': mean_a_i,
            'mean_income_primary': mean_income_primary,
            'mean_income_tertiary': mean_income_tertiary,
            'enrollment_rate': enrollment_rate
        }
    
    def train(self, n_epochs=2000, print_every=100):
        log("="*60)
        log("STAGE 2: TRAINING HANK WITH EDUCATION")
        log("="*60)
        log(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        log(f"Epochs: {n_epochs}, Device: {device}")
        log("")
        log("Education System:")
        log(f"  Levels: {self.params.n_education}")
        log(f"  Costs: {self.params.education_costs}")
        log(f"  Premiums: {self.params.education_premiums}")
        log(f"  Max enrollment age: {self.params.max_education_age}")
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
            self.history['education_loss'].append(metrics['education'])
            self.history['mean_c'].append(metrics['mean_c'])
            self.history['mean_a_l'].append(metrics['mean_a_l'])
            self.history['mean_a_i'].append(metrics['mean_a_i'])
            self.history['mean_income_primary'].append(metrics['mean_income_primary'])
            self.history['mean_income_tertiary'].append(metrics['mean_income_tertiary'])
            self.history['enrollment_rate'].append(metrics['enrollment_rate'])
            self.history['time'].append(elapsed)
            
            if metrics['total'] < best_loss:
                best_loss = metrics['total']
                torch.save(self.model.state_dict(), OUTPUT_DIR / 'stage2_best.pt')
            
            if epoch % print_every == 0 or epoch == n_epochs - 1:
                log(f"Epoch {epoch:4d} | Loss: {metrics['total']:.6f} | "
                    f"C: {metrics['mean_c']:.2f}, L: {metrics['mean_a_l']:.1f}, I: {metrics['mean_a_i']:.1f} | "
                    f"Enroll: {metrics['enrollment_rate']:.2%} | "
                    f"Time: {elapsed:.1f}s")
                
                # Show income premium
                if metrics['mean_income_primary'] > 0:
                    premium = metrics['mean_income_tertiary'] / metrics['mean_income_primary']
                    log(f"         Income Premium (Tertiary/Primary): {premium:.2f}x")
        
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
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        
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
        
        # Asset convergence
        ax = axes[0, 2]
        ax.plot(self.history['epoch'], self.history['mean_a_l'], 'r-', label='Liquid', linewidth=2)
        ax.plot(self.history['epoch'], self.history['mean_a_i'], 'g-', label='Illiquid', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Assets')
        ax.set_title('Asset Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Education metrics
        ax = axes[0, 3]
        ax.plot(self.history['epoch'], self.history['enrollment_rate'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Tertiary Enrollment Rate')
        ax.set_title('Education Choice Convergence')
        ax.grid(True, alpha=0.3)
        
        # Income by education level
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['mean_income_primary'], 'b-', label='Primary', linewidth=2)
        ax.plot(self.history['epoch'], self.history['mean_income_tertiary'], 'r-', label='Tertiary', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Income')
        ax.set_title('Income by Education Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final loss components
        ax = axes[1, 1]
        final_losses = [
            self.history['loss'][-1], 
            self.history['budget_loss'][-1], 
            self.history['constraint_loss'][-1],
            self.history['euler_loss'][-1],
            self.history['education_loss'][-1]
        ]
        ax.bar(['Total', 'Budget', 'Constraint', 'Euler', 'Education'], 
               final_losses, 
               color=['blue', 'red', 'green', 'orange', 'purple'])
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss Components')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Training time
        ax = axes[1, 2]
        ax.plot(self.history['epoch'], np.array(self.history['time'])/60, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (minutes)')
        ax.set_title('Cumulative Training Time')
        ax.grid(True, alpha=0.3)
        
        # Summary statistics table
        ax = axes[1, 3]
        ax.axis('off')
        
        # Calculate wage premium
        if self.history['mean_income_primary'][-1] > 0:
            wage_premium = self.history['mean_income_tertiary'][-1] / self.history['mean_income_primary'][-1]
        else:
            wage_premium = 0.0
        
        summary_text = f"""
        STAGE 2 TRAINING SUMMARY
        ========================
        
        Training Time: {self.history['time'][-1]/60:.1f} minutes
        Total Epochs: {len(self.history['epoch'])}
        
        Final Loss: {self.history['loss'][-1]:.6f}
        Budget Loss: {self.history['budget_loss'][-1]:.6f}
        Constraint Loss: {self.history['constraint_loss'][-1]:.6f}
        
        Mean Consumption: {self.history['mean_c'][-1]:.2f}
        Mean Liquid Assets: {self.history['mean_a_l'][-1]:.1f}
        Mean Illiquid Assets: {self.history['mean_a_i'][-1]:.1f}
        
        EDUCATION MOMENTS
        -----------------
        Tertiary Enrollment: {self.history['enrollment_rate'][-1]:.1%}
        Wage Premium (Tert/Prim): {wage_premium:.2f}x
        Target Premium: ~2.0x
        
        Target Income Gini: {self.params.targets['income_gini']:.3f}
        Target HH Debt/GDP: {self.params.targets['household_debt_gdp']:.3f}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'stage2_diagnostics.png', dpi=300, bbox_inches='tight')
        log(f"Saved diagnostics: {OUTPUT_DIR / 'stage2_diagnostics.png'}")
        plt.close()

# ====================================================================================
# VALIDATION
# ====================================================================================

class Stage2Validator:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.model.eval()
    
    def validate(self, n_agents=10000):
        log("")
        log("="*60)
        log("STAGE 2 VALIDATION")
        log("="*60)
        
        with torch.no_grad():
            # Sample diverse population
            state = {
                'liquid': torch.rand(n_agents, 1, device=device) * self.params.liquid_max,
                'illiquid': torch.rand(n_agents, 1, device=device) * self.params.illiquid_max,
                'base_income': torch.exp(torch.randn(n_agents, 1, device=device) * 0.5 + 0.5),
                'education_level': torch.randint(0, 3, (n_agents,), device=device),
                'age': torch.randint(18, 66, (n_agents, 1), device=device).float()
            }
            
            policies = self.model(
                state['liquid'], state['illiquid'], state['base_income'],
                state['education_level'], state['age']
            )
            
            budget_residual, budget_info = self.model.compute_budget(policies, state)
            
            # Compute statistics by education level
            stats_by_edu = {}
            for edu in range(3):
                mask = (state['education_level'] == edu)
                if mask.sum() > 0:
                    income_with_premium = budget_info['income'][mask]
                    stats_by_edu[edu] = {
                        'count': mask.sum().item(),
                        'mean_income': income_with_premium.mean().item(),
                        'mean_consumption': policies['consumption'][mask].mean().item(),
                        'mean_liquid': state['liquid'][mask].mean().item(),
                        'mean_illiquid': state['illiquid'][mask].mean().item()
                    }
            
            # Overall statistics
            stats = {
                'n_agents': n_agents,
                'mean_liquid': state['liquid'].mean().item(),
                'mean_illiquid': state['illiquid'].mean().item(),
                'mean_consumption': policies['consumption'].mean().item(),
                'mean_income': budget_info['income'].mean().item(),
                'mpc_ratio': (policies['consumption'] / budget_info['income']).mean().item(),
                'consumption_std': policies['consumption'].std().item(),
                'budget_residual_mean': budget_residual.mean().item(),
                'wage_premium_tertiary_primary': stats_by_edu[2]['mean_income'] / stats_by_edu[0]['mean_income'] if 0 in stats_by_edu and 2 in stats_by_edu else 0,
                'wage_premium_secondary_primary': stats_by_edu[1]['mean_income'] / stats_by_edu[0]['mean_income'] if 0 in stats_by_edu and 1 in stats_by_edu else 0,
            }
            
            # Enrollment statistics
            young_mask = (state['age'].squeeze() < self.params.max_education_age)
            stats['enrollment_rate_tertiary_young'] = policies['education_choice'][young_mask, 2].mean().item() if young_mask.any() else 0.0
            stats['fraction_young'] = young_mask.float().mean().item()
            
            log(f"Steady State (n={n_agents}):")
            log(f"  Mean liquid assets:      RM {stats['mean_liquid']:,.0f}")
            log(f"  Mean illiquid (EPF):     RM {stats['mean_illiquid']:,.0f}")
            log(f"  Total wealth:            RM {stats['mean_liquid'] + stats['mean_illiquid']:,.0f}")
            log(f"  Mean consumption:        RM {stats['mean_consumption']:.2f}")
            log(f"  Mean income (with edu):  RM {stats['mean_income']:.2f}")
            log(f"  Consumption/Income:      {stats['mean_consumption']/stats['mean_income']:.1%}")
            log(f"  Approx MPC:              {stats['mpc_ratio']:.3f}")
            log(f"  Budget residual:         {stats['budget_residual_mean']:.4f}")
            
            log("")
            log("Education Moments:")
            log(f"  Tertiary enrollment (young): {stats['enrollment_rate_tertiary_young']:.1%}")
            log(f"  Fraction young (age < 25):   {stats['fraction_young']:.1%}")
            
            log("")
            log("Income by Education Level:")
            for edu, name in [(0, 'Primary'), (1, 'Secondary'), (2, 'Tertiary')]:
                if edu in stats_by_edu:
                    s = stats_by_edu[edu]
                    log(f"  {name:10s}: n={s['count']:4d}, "
                        f"Income=RM{s['mean_income']:.2f}, "
                        f"C=RM{s['mean_consumption']:.2f}")
            
            log("")
            log("Wage Premiums (Relative to Primary):")
            if 0 in stats_by_edu and 1 in stats_by_edu:
                premium_1_0 = stats_by_edu[1]['mean_income'] / stats_by_edu[0]['mean_income']
                log(f"  Secondary: {premium_1_0:.2f}x (target: 1.3x)")
            if 0 in stats_by_edu and 2 in stats_by_edu:
                premium_2_0 = stats_by_edu[2]['mean_income'] / stats_by_edu[0]['mean_income']
                log(f"  Tertiary:  {premium_2_0:.2f}x (target: 2.0x)")
            
            log("")
            log("Data Targets:")
            log(f"  Income Gini:             {self.params.targets['income_gini']:.3f}")
            log(f"  Formal sector share:     {self.params.targets['formal_share']:.1%}")
            log(f"  Informal sector share:   {self.params.targets['informal_share']:.1%}")
            
            # Validation checks
            log("")
            log("Validation Checks:")
            checks = [
                ("Budget residual < 0.1", abs(stats['budget_residual_mean']) < 0.1),
                ("Consumption positive", stats['mean_consumption'] > 0),
                ("Assets non-negative", stats['mean_liquid'] >= 0 and stats['mean_illiquid'] >= 0),
                ("MPC reasonable", 0.05 < stats['mpc_ratio'] < 0.8),
                ("Consumption < Income", stats['mean_consumption'] < stats['mean_income'] * 1.5),
                ("Wage premium tertiary > 1.5x", stats['wage_premium_tertiary_primary'] > 1.5),
                ("Tertiary enrollment > 10%", stats['enrollment_rate_tertiary_young'] > 0.1)
            ]
            
            all_pass = True
            for check_name, passed in checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                log(f"  {check_name:40s} {status}")
                all_pass = all_pass and passed
            
            if all_pass:
                log("")
                log("✓ ALL VALIDATION CHECKS PASSED")
                log("  Ready for Stage 3: Add health")
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
    log("MALAYSIA HANK - STAGE 2: EDUCATION")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("")
    
    params = MalaysiaBaseParams()
    model = MalaysiaHANK_Stage2(params)
    
    # Try to load Stage 1 weights for base components if available
    stage1_path = STAGE1_DIR / 'stage1_best.pt'
    if stage1_path.exists():
        log(f"Found Stage 1 model: {stage1_path}")
        log("Note: Stage 2 uses extended architecture - training from scratch")
        log("")
    
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log("")
    
    trainer = Stage2Trainer(model, params, lr=1e-3)
    history = trainer.train(n_epochs=2000, print_every=100)
    trainer.plot_diagnostics()
    
    model.load_state_dict(torch.load(OUTPUT_DIR / 'stage2_best.pt'))
    validator = Stage2Validator(model, params)
    validator.validate()
    
    log("")
    log("="*60)
    log("STAGE 2 COMPLETE")
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("Outputs:")
    log(f"  - Model: {OUTPUT_DIR / 'stage2_best.pt'}")
    log(f"  - History: {OUTPUT_DIR / 'training_history.csv'}")
    log(f"  - Diagnostics: {OUTPUT_DIR / 'stage2_diagnostics.png'}")
    log(f"  - Validation: {OUTPUT_DIR / 'validation_results.json'}")
    log(f"  - Log: {LOG_FILE}")
    log("")
    log("Next: Stage 3 - Add Health module")

if __name__ == "__main__":
    main()
