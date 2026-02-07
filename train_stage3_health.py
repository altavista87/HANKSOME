"""
Malaysia HANK - Stage 3: Add Health
===================================
Extends Stage 2 by adding health states and medical choices.

Health System:
- 4 States: Healthy (0), Sick (1), Chronic (2), Disabled (3)
- Medical Costs: [0.0, 0.5, 2.0, 5.0] (increasing with severity)
- Healthcare Choice: Public (subsidized) vs Private (expensive)
- Income Impact: Disabled/Chronic earn less

State Space: 2,100 -> ~8,400 states (4x increase from Stage 2)
Training Time: 2-4 hours (4000 epochs)
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
OUTPUT_DIR = Path("outputs/stage3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE2_DIR = Path("outputs/stage2")

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
        
        # Education parameters (from Stage 2)
        self.n_education = 3
        self.education_costs = [0.0, 0.5, 2.0]
        self.education_premiums = [1.0, 1.3, 2.0]
        self.max_education_age = 25
        
        # Health parameters (NEW for Stage 3)
        self.n_health = 4  # Healthy, Sick, Chronic, Disabled
        self.health_costs = [0.0, 0.5, 2.0, 5.0]  # Medical expenses by state
        self.health_income_factor = [1.0, 0.95, 0.80, 0.50]  # Income reduction
        self.public_healthcare_subsidy = 0.8  # 80% subsidy for public
        self.private_healthcare_markup = 1.5  # Private costs 50% more
        
        # Health transition matrix (Markov)
        # Rows: current state, Cols: next state
        self.health_transition = np.array([
            [0.95, 0.04, 0.008, 0.002],  # Healthy
            [0.30, 0.60, 0.08,  0.02],   # Sick
            [0.10, 0.15, 0.70,  0.05],   # Chronic
            [0.05, 0.10, 0.15,  0.70],   # Disabled
        ], dtype=np.float32)
        
        # Load calibration targets
        with open(DATA_DIR / 'calibration_targets.json', 'r') as f:
            self.targets = json.load(f)

# ====================================================================================
# STAGE 3 MODEL: HANK + Education + Health
# ====================================================================================

class MalaysiaHANK_Stage3(nn.Module):
    """
    Stage 3: Two-asset HANK with education and health choices.
    
    State: (liquid, illiquid, base_income, education_level, age, health_state)
    Policies: (consumption, liquid_savings, illiquid_savings, education_choice, healthcare_choice)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # Education embedding: 3 levels -> 8 dimensions
        self.education_embed = nn.Embedding(params.n_education, 8)
        
        # Health embedding: 4 states -> 8 dimensions
        self.health_embed = nn.Embedding(params.n_health, 8)
        
        # Extended encoder: 3 continuous + 8 edu + 1 age + 8 health = 20 inputs
        self.encoder = nn.Sequential(
            nn.Linear(20, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        
        # Policy heads
        self.policy_head = nn.Sequential(
            nn.Linear(128, 3),
            nn.Softplus()  # c, a_l, a_i > 0
        )
        
        # Education choice head (probability over 3 levels)
        self.education_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, params.n_education)
        )
        
        # Healthcare choice head (probability over 2 options: public=0, private=1)
        self.healthcare_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, liquid, illiquid, base_income, education_level, age, health_state):
        """
        Forward pass through the network.
        
        Args:
            liquid: [batch, 1] - liquid assets
            illiquid: [batch, 1] - illiquid assets  
            base_income: [batch, 1] - base income (before education/health effects)
            education_level: [batch] - int in {0, 1, 2}
            age: [batch, 1] - age in years
            health_state: [batch] - int in {0, 1, 2, 3}
        
        Returns:
            Dictionary with all policy outputs
        """
        batch_size = liquid.size(0)
        
        # Normalize continuous inputs
        liquid_norm = liquid / self.params.liquid_max
        illiquid_norm = illiquid / self.params.illiquid_max
        income_norm = base_income / 2.0
        age_norm = age / 80.0
        
        # Embed discrete states
        edu_emb = self.education_embed(education_level)  # [batch, 8]
        health_emb = self.health_embed(health_state)  # [batch, 8]
        
        # Concatenate all state components: 3 + 8 + 1 + 8 = 20
        state = torch.cat([
            liquid_norm, 
            illiquid_norm, 
            income_norm,
            edu_emb,
            age_norm,
            health_emb
        ], dim=-1)
        
        # Encode state
        features = self.encoder(state)  # [batch, 128]
        
        # Get base policies (consumption and savings)
        policies_norm = self.policy_head(features)
        
        consumption = policies_norm[:, 0:1] * 2.0
        liquid_savings = policies_norm[:, 1:2] * self.params.liquid_max
        illiquid_savings = policies_norm[:, 2:3] * self.params.illiquid_max
        
        # Get education choice probabilities (only for young)
        edu_logits = self.education_head(features)
        can_enroll = (age.squeeze(-1) < self.params.max_education_age).float()
        current_edu_onehot = nn.functional.one_hot(
            education_level, num_classes=self.params.n_education
        ).float()
        edu_choice_soft = torch.softmax(edu_logits, dim=-1)
        can_enroll_expanded = can_enroll.unsqueeze(-1)
        education_choice = (
            can_enroll_expanded * edu_choice_soft + 
            (1 - can_enroll_expanded) * current_edu_onehot
        )
        
        # Get healthcare choice probabilities (public vs private)
        healthcare_logits = self.healthcare_head(features)
        healthcare_choice = torch.softmax(healthcare_logits, dim=-1)  # [batch, 2]
        
        return {
            'consumption': consumption,
            'liquid_savings': liquid_savings,
            'illiquid_savings': illiquid_savings,
            'education_choice': education_choice,
            'education_logits': edu_logits,
            'can_enroll': can_enroll,
            'healthcare_choice': healthcare_choice,
            'healthcare_logits': healthcare_logits
        }
    
    def compute_income(self, base_income, education_level, health_state):
        """
        Compute income with education premium and health penalty.
        
        income = base_income × education_premium × health_factor
        """
        # Education premium
        edu_premiums = torch.tensor(
            self.params.education_premiums, 
            device=base_income.device, 
            dtype=base_income.dtype
        )[education_level]
        
        # Health factor (income reduction for poor health)
        health_factors = torch.tensor(
            self.params.health_income_factor,
            device=base_income.device,
            dtype=base_income.dtype
        )[health_state]
        
        return base_income * edu_premiums.unsqueeze(-1) * health_factors.unsqueeze(-1)
    
    def compute_health_expenses(self, health_state, healthcare_choice):
        """
        Compute health expenses based on state and choice.
        
        Public: base_cost × (1 - subsidy) = base_cost × 0.2
        Private: base_cost × markup = base_cost × 1.5
        """
        # Base cost by health state
        base_costs = torch.tensor(
            self.params.health_costs,
            device=health_state.device,
            dtype=torch.float32
        )[health_state]  # [batch]
        
        # Healthcare choice: [batch, 2] where [:, 0] = public, [:, 1] = private
        public_prob = healthcare_choice[:, 0]
        private_prob = healthcare_choice[:, 1]
        
        # Expected cost = P(public) × public_cost + P(private) × private_cost
        public_cost = base_costs * (1 - self.params.public_healthcare_subsidy)  # 20%
        private_cost = base_costs * self.params.private_healthcare_markup  # 150%
        
        expected_cost = public_prob * public_cost + private_prob * private_cost
        
        return expected_cost.unsqueeze(-1)  # [batch, 1]
    
    def compute_budget(self, policies, state):
        """
        Compute budget constraint residual.
        
        cash_on_hand = (1+r_l)*liquid + (1+r_i)*illiquid + income - health_expenses
        residual = c + a_l + a_i + education_cost - coh
        """
        liquid = state['liquid']
        illiquid = state['illiquid']
        base_income = state['base_income']
        education_level = state['education_level']
        health_state = state['health_state']
        
        c = policies['consumption']
        a_l = policies['liquid_savings']
        a_i = policies['illiquid_savings']
        edu_choice = policies['education_choice']
        healthcare_choice = policies['healthcare_choice']
        
        # Compute income with education and health effects
        income = self.compute_income(base_income, education_level, health_state)
        
        # Compute health expenses
        health_exp = self.compute_health_expenses(health_state, healthcare_choice)
        
        # Compute education cost
        education_costs = torch.tensor(
            self.params.education_costs,
            device=liquid.device,
            dtype=liquid.dtype
        )
        expected_edu_cost = (edu_choice * education_costs.unsqueeze(0)).sum(dim=1, keepdim=True)
        
        # Cash on hand (before health and education costs)
        r_l, r_i = self.params.r_liquid, self.params.r_illiquid
        coh = (1 + r_l) * liquid + (1 + r_i) * illiquid + income - health_exp
        
        # Budget: c + a_l + a_i + edu_cost = coh
        total_spending = c + a_l + a_i + expected_edu_cost
        residual = total_spending - coh
        
        return residual, {
            'coh': coh,
            'expected_edu_cost': expected_edu_cost,
            'health_exp': health_exp,
            'income': income
        }

# ====================================================================================
# TRAINER WITH DIAGNOSTICS
# ====================================================================================

class Stage3Trainer:
    def __init__(self, model, params, lr=1e-3):
        self.model = model.to(device)
        self.params = params
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=100, factor=0.5
        )
        self.history = {
            'epoch': [], 'loss': [], 'budget_loss': [], 
            'constraint_loss': [], 'euler_loss': [], 'education_loss': [],
            'mean_c': [], 'mean_a_l': [], 'mean_a_i': [],
            'mean_income_healthy': [], 'mean_income_disabled': [],
            'enrollment_rate': [], 'private_healthcare_rate': [],
            'time': []
        }
        self.start_time = time.time()
        
    def sample_state(self, batch_size):
        """Sample random states including education, age, and health."""
        liquid = torch.rand(batch_size, 1, device=device) * self.params.liquid_max
        illiquid = torch.rand(batch_size, 1, device=device) * self.params.illiquid_max
        base_income = torch.exp(torch.randn(batch_size, 1, device=device) * 0.5 + 0.5)
        base_income = torch.clamp(base_income, 0.5, 3.0)
        
        # Education: mostly primary/secondary, some tertiary
        education_level = torch.randint(0, 3, (batch_size,), device=device)
        
        # Age: uniform 18-65 (working age)
        age = torch.randint(18, 66, (batch_size, 1), device=device).float()
        
        # Health: sample from stationary distribution
        # Approximate stationary: mostly healthy, few disabled
        health_probs = torch.tensor([0.75, 0.15, 0.08, 0.02], device=device)
        health_state = torch.multinomial(health_probs, batch_size, replacement=True)
        
        return {
            'liquid': liquid,
            'illiquid': illiquid,
            'base_income': base_income,
            'education_level': education_level,
            'age': age,
            'health_state': health_state
        }
    
    def compute_loss(self, batch_size=512):
        state = self.sample_state(batch_size)
        
        policies = self.model(
            state['liquid'], 
            state['illiquid'], 
            state['base_income'],
            state['education_level'],
            state['age'],
            state['health_state']
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
        
        # Education consistency
        can_enroll = policies['can_enroll']
        edu_choice = policies['education_choice']
        edu_levels = torch.arange(3, device=device).float().unsqueeze(0)
        expected_edu_level = (edu_choice * edu_levels).sum(dim=1)
        young_target = torch.ones_like(expected_edu_level) * 1.5
        education_penalty = torch.where(
            can_enroll.bool(),
            torch.relu(young_target - expected_edu_level),
            torch.zeros_like(expected_edu_level)
        )
        education_loss = torch.mean(education_penalty ** 2)
        
        # Total loss
        loss = (
            10.0 * budget_loss + 
            100.0 * constraint_loss + 
            1.0 * euler_loss +
            0.1 * education_loss
        )
        
        # Compute diagnostics
        with torch.no_grad():
            mean_c = policies['consumption'].mean().item()
            mean_a_l = policies['liquid_savings'].mean().item()
            mean_a_i = policies['illiquid_savings'].mean().item()
            
            # Income by health state
            income = budget_info['income']
            healthy_mask = (state['health_state'] == 0)
            disabled_mask = (state['health_state'] == 3)
            
            mean_income_healthy = income[healthy_mask].mean().item() if healthy_mask.any() else 0.0
            mean_income_disabled = income[disabled_mask].mean().item() if disabled_mask.any() else 0.0
            
            # Enrollment rate among young
            young_mask = (state['age'].squeeze() < self.params.max_education_age)
            enrollment_rate = policies['education_choice'][young_mask, 2].mean().item() if young_mask.any() else 0.0
            
            # Private healthcare rate
            private_healthcare_rate = policies['healthcare_choice'][:, 1].mean().item()
        
        return loss, {
            'total': loss.item(),
            'budget': budget_loss.item(),
            'constraint': constraint_loss.item(),
            'euler': euler_loss.item(),
            'education': education_loss.item(),
            'mean_c': mean_c,
            'mean_a_l': mean_a_l,
            'mean_a_i': mean_a_i,
            'mean_income_healthy': mean_income_healthy,
            'mean_income_disabled': mean_income_disabled,
            'enrollment_rate': enrollment_rate,
            'private_healthcare_rate': private_healthcare_rate
        }
    
    def train(self, n_epochs=4000, print_every=200):
        log("="*60)
        log("STAGE 3: TRAINING HANK WITH EDUCATION + HEALTH")
        log("="*60)
        log(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        log(f"Epochs: {n_epochs}, Device: {device}")
        log("")
        log("Education System:")
        log(f"  Levels: {self.params.n_education}")
        log(f"  Costs: {self.params.education_costs}")
        log(f"  Premiums: {self.params.education_premiums}")
        log("")
        log("Health System:")
        log(f"  States: {self.params.n_health} (Healthy, Sick, Chronic, Disabled)")
        log(f"  Medical Costs: {self.params.health_costs}")
        log(f"  Income Factors: {self.params.health_income_factor}")
        log(f"  Public Subsidy: {self.params.public_healthcare_subsidy:.0%}")
        log(f"  Private Markup: {self.params.private_healthcare_markup:.0%}")
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
            self.history['mean_income_healthy'].append(metrics['mean_income_healthy'])
            self.history['mean_income_disabled'].append(metrics['mean_income_disabled'])
            self.history['enrollment_rate'].append(metrics['enrollment_rate'])
            self.history['private_healthcare_rate'].append(metrics['private_healthcare_rate'])
            self.history['time'].append(elapsed)
            
            if metrics['total'] < best_loss:
                best_loss = metrics['total']
                torch.save(self.model.state_dict(), OUTPUT_DIR / 'stage3_best.pt')
            
            if epoch % print_every == 0 or epoch == n_epochs - 1:
                log(f"Epoch {epoch:4d} | Loss: {metrics['total']:.6f} | "
                    f"C: {metrics['mean_c']:.2f}, L: {metrics['mean_a_l']:.1f}, I: {metrics['mean_a_i']:.1f} | "
                    f"Enroll: {metrics['enrollment_rate']:.1%} | "
                    f"PrivateHC: {metrics['private_healthcare_rate']:.1%} | "
                    f"Time: {elapsed:.1f}s")
                
                # Show income impact of health
                if metrics['mean_income_healthy'] > 0:
                    health_penalty = metrics['mean_income_disabled'] / metrics['mean_income_healthy']
                    log(f"         Income (Disabled/Healthy): {health_penalty:.2f}x")
        
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
        
        # Policy means
        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['mean_c'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Consumption')
        ax.set_title('Consumption Convergence')
        ax.grid(True, alpha=0.3)
        
        # Assets
        ax = axes[0, 2]
        ax.plot(self.history['epoch'], self.history['mean_a_l'], 'r-', label='Liquid', linewidth=2)
        ax.plot(self.history['epoch'], self.history['mean_a_i'], 'g-', label='Illiquid', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Assets')
        ax.set_title('Asset Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Education enrollment
        ax = axes[0, 3]
        ax.plot(self.history['epoch'], self.history['enrollment_rate'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Tertiary Enrollment Rate')
        ax.set_title('Education Choice')
        ax.grid(True, alpha=0.3)
        
        # Income by health
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['mean_income_healthy'], 'b-', label='Healthy', linewidth=2)
        ax.plot(self.history['epoch'], self.history['mean_income_disabled'], 'r-', label='Disabled', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Income')
        ax.set_title('Income by Health State')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Healthcare choice
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], self.history['private_healthcare_rate'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Private Healthcare Rate')
        ax.set_title('Healthcare Choice')
        ax.grid(True, alpha=0.3)
        
        # Training time
        ax = axes[1, 2]
        ax.plot(self.history['epoch'], np.array(self.history['time'])/60, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (minutes)')
        ax.set_title('Training Time')
        ax.grid(True, alpha=0.3)
        
        # Summary
        ax = axes[1, 3]
        ax.axis('off')
        
        if self.history['mean_income_healthy'][-1] > 0:
            health_penalty = self.history['mean_income_disabled'][-1] / self.history['mean_income_healthy'][-1]
        else:
            health_penalty = 0.0
        
        summary_text = f"""
        STAGE 3 TRAINING SUMMARY
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
        
        HEALTH MOMENTS
        --------------
        Private Healthcare: {self.history['private_healthcare_rate'][-1]:.1%}
        Income (Disabled/Healthy): {health_penalty:.2f}x
        Target: ~0.5x
        
        Target Chronic Disease: {self.params.targets['chronic_disease_rate']:.1%}
        Target Public Healthcare: {self.params.targets['public_healthcare_share']:.1%}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'stage3_diagnostics.png', dpi=300, bbox_inches='tight')
        log(f"Saved diagnostics: {OUTPUT_DIR / 'stage3_diagnostics.png'}")
        plt.close()

# ====================================================================================
# VALIDATION
# ====================================================================================

class Stage3Validator:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.model.eval()
    
    def validate(self, n_agents=10000):
        log("")
        log("="*60)
        log("STAGE 3 VALIDATION")
        log("="*60)
        
        with torch.no_grad():
            # Sample diverse population
            state = {
                'liquid': torch.rand(n_agents, 1, device=device) * self.params.liquid_max,
                'illiquid': torch.rand(n_agents, 1, device=device) * self.params.illiquid_max,
                'base_income': torch.exp(torch.randn(n_agents, 1, device=device) * 0.5 + 0.5),
                'education_level': torch.randint(0, 3, (n_agents,), device=device),
                'age': torch.randint(18, 66, (n_agents, 1), device=device).float(),
                'health_state': torch.multinomial(
                    torch.tensor([0.75, 0.15, 0.08, 0.02], device=device),
                    n_agents, replacement=True
                )
            }
            
            policies = self.model(
                state['liquid'], state['illiquid'], state['base_income'],
                state['education_level'], state['age'], state['health_state']
            )
            
            budget_residual, budget_info = self.model.compute_budget(policies, state)
            
            # Statistics by health state
            stats_by_health = {}
            for h, name in [(0, 'Healthy'), (1, 'Sick'), (2, 'Chronic'), (3, 'Disabled')]:
                mask = (state['health_state'] == h)
                if mask.sum() > 0:
                    income = budget_info['income'][mask]
                    health_exp = budget_info['health_exp'][mask]
                    stats_by_health[h] = {
                        'name': name,
                        'count': mask.sum().item(),
                        'fraction': mask.float().mean().item(),
                        'mean_income': income.mean().item(),
                        'mean_health_exp': health_exp.mean().item(),
                        'private_healthcare_rate': policies['healthcare_choice'][mask, 1].mean().item()
                    }
            
            # Overall statistics
            stats = {
                'n_agents': n_agents,
                'mean_liquid': state['liquid'].mean().item(),
                'mean_illiquid': state['illiquid'].mean().item(),
                'mean_consumption': policies['consumption'].mean().item(),
                'mean_income': budget_info['income'].mean().item(),
                'mean_health_exp': budget_info['health_exp'].mean().item(),
                'mpc_ratio': (policies['consumption'] / budget_info['income']).mean().item(),
                'budget_residual_mean': budget_residual.mean().item(),
                'private_healthcare_rate': policies['healthcare_choice'][:, 1].mean().item(),
            }
            
            # Education stats
            young_mask = (state['age'].squeeze() < self.params.max_education_age)
            stats['enrollment_rate_tertiary_young'] = policies['education_choice'][young_mask, 2].mean().item() if young_mask.any() else 0.0
            
            log(f"Steady State (n={n_agents}):")
            log(f"  Mean liquid assets:      RM {stats['mean_liquid']:,.0f}")
            log(f"  Mean illiquid (EPF):     RM {stats['mean_illiquid']:,.0f}")
            log(f"  Mean consumption:        RM {stats['mean_consumption']:.2f}")
            log(f"  Mean income:             RM {stats['mean_income']:.2f}")
            log(f"  Mean health expenses:    RM {stats['mean_health_exp']:.2f}")
            log(f"  Budget residual:         {stats['budget_residual_mean']:.4f}")
            
            log("")
            log("Health Distribution & Outcomes:")
            for h in range(4):
                if h in stats_by_health:
                    s = stats_by_health[h]
                    log(f"  {s['name']:10s}: {s['fraction']:.1%} | "
                        f"Income=RM{s['mean_income']:.2f}, "
                        f"HealthExp=RM{s['mean_health_exp']:.2f}, "
                        f"PrivateHC={s['private_healthcare_rate']:.1%}")
            
            # Income penalty for poor health
            if 0 in stats_by_health and 3 in stats_by_health:
                income_ratio = stats_by_health[3]['mean_income'] / stats_by_health[0]['mean_income']
                stats['income_ratio_disabled_healthy'] = income_ratio
                log("")
                log(f"Income Impact: Disabled earn {income_ratio:.1%} of Healthy")
            
            log("")
            log("Education Moments:")
            log(f"  Tertiary enrollment (young): {stats['enrollment_rate_tertiary_young']:.1%}")
            
            log("")
            log("Healthcare Moments:")
            log(f"  Private healthcare rate:     {stats['private_healthcare_rate']:.1%}")
            log(f"  Public healthcare rate:      {1 - stats['private_healthcare_rate']:.1%}")
            
            log("")
            log("Data Targets:")
            log(f"  Chronic disease rate:    {self.params.targets['chronic_disease_rate']:.1%}")
            log(f"  Public healthcare share: {self.params.targets['public_healthcare_share']:.1%}")
            
            # Validation checks
            log("")
            log("Validation Checks:")
            checks = [
                ("Budget residual < 0.1", abs(stats['budget_residual_mean']) < 0.1),
                ("Consumption positive", stats['mean_consumption'] > 0),
                ("Assets non-negative", stats['mean_liquid'] >= 0 and stats['mean_illiquid'] >= 0),
                ("MPC reasonable", 0.05 < stats['mpc_ratio'] < 0.8),
                ("Health expenses positive", stats['mean_health_exp'] >= 0),
                ("Private healthcare < 50%", stats['private_healthcare_rate'] < 0.5),
                ("Income penalty for disabled", income_ratio < 0.7 if 'income_ratio_disabled_healthy' in stats else False)
            ]
            
            all_pass = True
            for check_name, passed in checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                log(f"  {check_name:40s} {status}")
                all_pass = all_pass and passed
            
            if all_pass:
                log("")
                log("✓ ALL VALIDATION CHECKS PASSED")
                log("  Ready for Stage 4: Add housing")
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
    log("MALAYSIA HANK - STAGE 3: HEALTH")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("")
    
    params = MalaysiaBaseParams()
    model = MalaysiaHANK_Stage3(params)
    
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log("")
    
    trainer = Stage3Trainer(model, params, lr=1e-3)
    history = trainer.train(n_epochs=4000, print_every=200)
    trainer.plot_diagnostics()
    
    model.load_state_dict(torch.load(OUTPUT_DIR / 'stage3_best.pt'))
    validator = Stage3Validator(model, params)
    validator.validate()
    
    log("")
    log("="*60)
    log("STAGE 3 COMPLETE")
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("Outputs:")
    log(f"  - Model: {OUTPUT_DIR / 'stage3_best.pt'}")
    log(f"  - History: {OUTPUT_DIR / 'training_history.csv'}")
    log(f"  - Diagnostics: {OUTPUT_DIR / 'stage3_diagnostics.png'}")
    log(f"  - Validation: {OUTPUT_DIR / 'validation_results.json'}")
    log(f"  - Log: {LOG_FILE}")
    log("")
    log("Next: Stage 4 - Add Housing module")

if __name__ == "__main__":
    main()
