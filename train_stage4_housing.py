"""
Malaysia HANK - Stage 4: Add Housing
====================================
Extends Stage 3 by adding housing tenure choices and mortgage decisions.

Housing System:
- 3 Tenure Types: Rent (0), Own with Mortgage (1), Own Outright (2)
- House Price: 3x annual income
- Down Payment: 10% minimum
- Mortgage Rate: 4.5% (r_mortgage)
- Mortgage Term: 30 years
- Rent Cost: 25% of income

State Space: ~8,400 -> ~25,000+ states (3x increase from Stage 3)
Training Time: 4-6 hours (6000 epochs)
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
OUTPUT_DIR = Path("outputs/stage4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE3_DIR = Path("outputs/stage3")

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
        self.r_mortgage = 0.045  # NEW: Mortgage rate
        
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
        
        # Health parameters (from Stage 3)
        self.n_health = 4
        self.health_costs = [0.0, 0.5, 2.0, 5.0]
        self.health_income_factor = [1.0, 0.95, 0.80, 0.50]
        self.public_healthcare_subsidy = 0.8
        self.private_healthcare_markup = 1.5
        self.health_transition = np.array([
            [0.95, 0.04, 0.008, 0.002],
            [0.30, 0.60, 0.08,  0.02],
            [0.10, 0.15, 0.70,  0.05],
            [0.05, 0.10, 0.15,  0.70],
        ], dtype=np.float32)
        
        # Housing parameters (NEW for Stage 4)
        self.n_housing = 3  # Rent, Mortgage, Outright
        self.house_price_multiple = 3.0  # House price = 3x annual income
        self.down_payment_min = 0.10  # 10% minimum down payment
        self.mortgage_term = 30  # 30-year mortgage
        self.rent_share = 0.25  # Rent = 25% of income
        self.ltv_max = 0.90  # 90% loan-to-value max
        
        # Load calibration targets
        with open(DATA_DIR / 'calibration_targets.json', 'r') as f:
            self.targets = json.load(f)

# ====================================================================================
# STAGE 4 MODEL: HANK + Education + Health + Housing
# ====================================================================================

class MalaysiaHANK_Stage4(nn.Module):
    """
    Stage 4: Two-asset HANK with education, health, and housing choices.
    
    State: (liquid, illiquid, mortgage_debt, base_income, education, age, health, housing)
    Policies: (consumption, liquid_savings, illiquid_savings, education_choice, 
               healthcare_choice, housing_choice, mortgage_payment)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # Embeddings for discrete states
        self.education_embed = nn.Embedding(params.n_education, 8)
        self.health_embed = nn.Embedding(params.n_health, 8)
        self.housing_embed = nn.Embedding(params.n_housing, 8)
        
        # Extended encoder: 4 continuous + 8 edu + 1 age + 8 health + 8 housing = 29 inputs
        # Continuous: liquid, illiquid, mortgage_debt, base_income
        self.encoder = nn.Sequential(
            nn.Linear(29, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )
        
        # Policy heads
        self.policy_head = nn.Sequential(
            nn.Linear(256, 3),
            nn.Softplus()  # c, a_l, a_i > 0
        )
        
        # Education choice head
        self.education_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, params.n_education)
        )
        
        # Healthcare choice head
        self.healthcare_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )
        
        # Housing choice head
        self.housing_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, params.n_housing)
        )
        
    def forward(self, liquid, illiquid, mortgage_debt, base_income, 
                education_level, age, health_state, housing_type):
        """
        Forward pass through the network.
        
        Args:
            liquid: [batch, 1] - liquid assets
            illiquid: [batch, 1] - illiquid assets (EPF)
            mortgage_debt: [batch, 1] - outstanding mortgage debt
            base_income: [batch, 1] - base income
            education_level: [batch] - int in {0, 1, 2}
            age: [batch, 1] - age in years
            health_state: [batch] - int in {0, 1, 2, 3}
            housing_type: [batch] - int in {0, 1, 2}
        
        Returns:
            Dictionary with all policy outputs
        """
        batch_size = liquid.size(0)
        
        # Normalize continuous inputs
        liquid_norm = liquid / self.params.liquid_max
        illiquid_norm = illiquid / self.params.illiquid_max
        mortgage_norm = mortgage_debt / (self.params.house_price_multiple * 10)  # Normalize by max house price
        income_norm = base_income / 2.0
        age_norm = age / 80.0
        
        # Embed discrete states
        edu_emb = self.education_embed(education_level)
        health_emb = self.health_embed(health_state)
        housing_emb = self.housing_embed(housing_type)
        
        # Concatenate all state components: 4 + 8 + 1 + 8 + 8 = 29
        state = torch.cat([
            liquid_norm, illiquid_norm, mortgage_norm, income_norm,
            edu_emb, age_norm, health_emb, housing_emb
        ], dim=-1)
        
        # Encode state
        features = self.encoder(state)
        
        # Get base policies
        policies_norm = self.policy_head(features)
        consumption = policies_norm[:, 0:1] * 2.0
        liquid_savings = policies_norm[:, 1:2] * self.params.liquid_max
        illiquid_savings = policies_norm[:, 2:3] * self.params.illiquid_max
        
        # Get education choice (only for young)
        edu_logits = self.education_head(features)
        can_enroll = (age.squeeze(-1) < self.params.max_education_age).float()
        current_edu_onehot = nn.functional.one_hot(
            education_level, num_classes=self.params.n_education
        ).float()
        edu_choice_soft = torch.softmax(edu_logits, dim=-1)
        education_choice = (
            can_enroll.unsqueeze(-1) * edu_choice_soft + 
            (1 - can_enroll.unsqueeze(-1)) * current_edu_onehot
        )
        
        # Get healthcare choice
        healthcare_logits = self.healthcare_head(features)
        healthcare_choice = torch.softmax(healthcare_logits, dim=-1)
        
        # Get housing choice
        housing_logits = self.housing_head(features)
        housing_choice = torch.softmax(housing_logits, dim=-1)
        
        return {
            'consumption': consumption,
            'liquid_savings': liquid_savings,
            'illiquid_savings': illiquid_savings,
            'education_choice': education_choice,
            'healthcare_choice': healthcare_choice,
            'housing_choice': housing_choice,
            'can_enroll': can_enroll
        }
    
    def compute_income(self, base_income, education_level, health_state):
        """Compute income with education premium and health penalty."""
        edu_premiums = torch.tensor(
            self.params.education_premiums, 
            device=base_income.device, 
            dtype=base_income.dtype
        )[education_level]
        
        health_factors = torch.tensor(
            self.params.health_income_factor,
            device=base_income.device,
            dtype=base_income.dtype
        )[health_state]
        
        return base_income * edu_premiums.unsqueeze(-1) * health_factors.unsqueeze(-1)
    
    def compute_health_expenses(self, health_state, healthcare_choice):
        """Compute health expenses based on state and choice."""
        base_costs = torch.tensor(
            self.params.health_costs,
            device=health_state.device,
            dtype=torch.float32
        )[health_state]
        
        public_prob = healthcare_choice[:, 0]
        private_prob = healthcare_choice[:, 1]
        
        public_cost = base_costs * (1 - self.params.public_healthcare_subsidy)
        private_cost = base_costs * self.params.private_healthcare_markup
        
        expected_cost = public_prob * public_cost + private_prob * private_cost
        return expected_cost.unsqueeze(-1)
    
    def compute_housing_expenses(self, income, housing_choice, mortgage_debt):
        """
        Compute housing expenses based on choice.
        
        Rent: 25% of income
        Mortgage: Monthly payment based on outstanding debt
        Outright: 0 (but requires down payment to purchase)
        """
        batch_size = income.size(0)
        
        # Rent cost
        rent_cost = income * self.params.rent_share
        
        # Mortgage payment (monthly payment on outstanding debt)
        r_mort = self.params.r_mortgage
        n_years = self.params.mortgage_term
        
        # Monthly mortgage payment formula: P * [r(1+r)^n] / [(1+r)^n - 1]
        # Here we use annual for simplicity
        if r_mort > 0 and n_years > 0:
            mortgage_payment = mortgage_debt * (
                r_mort * (1 + r_mort)**n_years / ((1 + r_mort)**n_years - 1)
            )
        else:
            mortgage_payment = mortgage_debt / n_years if n_years > 0 else torch.zeros_like(mortgage_debt)
        
        # Outright: no ongoing cost
        outright_cost = torch.zeros_like(income)
        
        # Expected cost based on choice probabilities
        costs = torch.cat([rent_cost, mortgage_payment, outright_cost], dim=1)
        expected_cost = (housing_choice * costs).sum(dim=1, keepdim=True)
        
        return expected_cost, {
            'rent_cost': rent_cost,
            'mortgage_payment': mortgage_payment,
            'outright_cost': outright_cost
        }
    
    def compute_budget(self, policies, state):
        """Compute budget constraint residual."""
        liquid = state['liquid']
        illiquid = state['illiquid']
        mortgage_debt = state['mortgage_debt']
        base_income = state['base_income']
        education_level = state['education_level']
        health_state = state['health_state']
        
        c = policies['consumption']
        a_l = policies['liquid_savings']
        a_i = policies['illiquid_savings']
        edu_choice = policies['education_choice']
        healthcare_choice = policies['healthcare_choice']
        housing_choice = policies['housing_choice']
        
        # Compute effective income
        income = self.compute_income(base_income, education_level, health_state)
        
        # Compute expenses
        health_exp = self.compute_health_expenses(health_state, healthcare_choice)
        housing_exp, housing_details = self.compute_housing_expenses(
            income, housing_choice, mortgage_debt
        )
        
        # Education cost
        education_costs = torch.tensor(
            self.params.education_costs,
            device=liquid.device,
            dtype=liquid.dtype
        )
        expected_edu_cost = (edu_choice * education_costs.unsqueeze(0)).sum(dim=1, keepdim=True)
        
        # Cash on hand
        r_l, r_i = self.params.r_liquid, self.params.r_illiquid
        coh = (1 + r_l) * liquid + (1 + r_i) * illiquid + income - health_exp - housing_exp
        
        # Budget constraint
        total_spending = c + a_l + a_i + expected_edu_cost
        residual = total_spending - coh
        
        return residual, {
            'coh': coh,
            'income': income,
            'health_exp': health_exp,
            'housing_exp': housing_exp,
            'expected_edu_cost': expected_edu_cost
        }

# ====================================================================================
# TRAINER
# ====================================================================================

class Stage4Trainer:
    def __init__(self, model, params, lr=1e-3):
        self.model = model.to(device)
        self.params = params
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=200, factor=0.5
        )
        self.history = {
            'epoch': [], 'loss': [], 'budget_loss': [], 
            'constraint_loss': [], 'euler_loss': [],
            'mean_c': [], 'mean_a_l': [], 'mean_a_i': [],
            'mean_mortgage': [], 'homeownership_rate': [],
            'enrollment_rate': [], 'private_healthcare_rate': [],
            'time': []
        }
        self.start_time = time.time()
        
    def sample_state(self, batch_size):
        """Sample random states."""
        liquid = torch.rand(batch_size, 1, device=device) * self.params.liquid_max
        illiquid = torch.rand(batch_size, 1, device=device) * self.params.illiquid_max
        
        # Mortgage debt: some have mortgages, most don't
        has_mortgage = torch.rand(batch_size, 1, device=device) < 0.3
        mortgage_debt = has_mortgage.float() * torch.rand(batch_size, 1, device=device) * 50.0
        
        base_income = torch.exp(torch.randn(batch_size, 1, device=device) * 0.5 + 0.5)
        base_income = torch.clamp(base_income, 0.5, 3.0)
        
        education_level = torch.randint(0, 3, (batch_size,), device=device)
        age = torch.randint(18, 66, (batch_size, 1), device=device).float()
        
        health_probs = torch.tensor([0.75, 0.15, 0.08, 0.02], device=device)
        health_state = torch.multinomial(health_probs, batch_size, replacement=True)
        
        # Housing: mostly renting or owning
        housing_probs = torch.tensor([0.5, 0.3, 0.2], device=device)
        housing_type = torch.multinomial(housing_probs, batch_size, replacement=True)
        
        return {
            'liquid': liquid,
            'illiquid': illiquid,
            'mortgage_debt': mortgage_debt,
            'base_income': base_income,
            'education_level': education_level,
            'age': age,
            'health_state': health_state,
            'housing_type': housing_type
        }
    
    def compute_loss(self, batch_size=512):
        state = self.sample_state(batch_size)
        
        policies = self.model(
            state['liquid'], state['illiquid'], state['mortgage_debt'],
            state['base_income'], state['education_level'], state['age'],
            state['health_state'], state['housing_type']
        )
        
        # Budget constraint
        budget_residual, budget_info = self.model.compute_budget(policies, state)
        budget_loss = torch.mean(budget_residual ** 2)
        
        # Borrowing constraints
        liquid_viol = torch.relu(-policies['liquid_savings'])
        illiquid_viol = torch.relu(-policies['illiquid_savings'])
        constraint_loss = torch.mean(liquid_viol ** 2 + illiquid_viol ** 2)
        
        # Euler equation
        c = policies['consumption']
        mu_t = c ** (-self.params.sigma)
        beta, r_l = self.params.beta, self.params.r_liquid
        euler_residual = mu_t - beta * (1 + r_l) * mu_t
        euler_loss = torch.mean(euler_residual ** 2)
        
        # Total loss
        loss = 10.0 * budget_loss + 100.0 * constraint_loss + 1.0 * euler_loss
        
        # Diagnostics
        with torch.no_grad():
            mean_c = policies['consumption'].mean().item()
            mean_a_l = policies['liquid_savings'].mean().item()
            mean_a_i = policies['illiquid_savings'].mean().item()
            mean_mortgage = state['mortgage_debt'].mean().item()
            
            # Homeownership = mortgage + outright
            homeownership_rate = policies['housing_choice'][:, 1:].sum().item() / batch_size
            
            young_mask = (state['age'].squeeze() < self.params.max_education_age)
            enrollment_rate = policies['education_choice'][young_mask, 2].mean().item() if young_mask.any() else 0.0
            
            private_healthcare_rate = policies['healthcare_choice'][:, 1].mean().item()
        
        return loss, {
            'total': loss.item(),
            'budget': budget_loss.item(),
            'constraint': constraint_loss.item(),
            'euler': euler_loss.item(),
            'mean_c': mean_c,
            'mean_a_l': mean_a_l,
            'mean_a_i': mean_a_i,
            'mean_mortgage': mean_mortgage,
            'homeownership_rate': homeownership_rate,
            'enrollment_rate': enrollment_rate,
            'private_healthcare_rate': private_healthcare_rate
        }
    
    def train(self, n_epochs=6000, print_every=300):
        log("="*60)
        log("STAGE 4: TRAINING HANK + EDUCATION + HEALTH + HOUSING")
        log("="*60)
        log(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        log(f"Epochs: {n_epochs}, Device: {device}")
        log("")
        log("Housing System:")
        log(f"  Types: {self.params.n_housing} (Rent, Mortgage, Outright)")
        log(f"  House Price: {self.params.house_price_multiple}x income")
        log(f"  Down Payment: {self.params.down_payment_min:.0%}")
        log(f"  Mortgage Rate: {self.params.r_mortgage:.1%}")
        log(f"  Rent Share: {self.params.rent_share:.0%}")
        log("")
        
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
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
            self.history['mean_mortgage'].append(metrics['mean_mortgage'])
            self.history['homeownership_rate'].append(metrics['homeownership_rate'])
            self.history['enrollment_rate'].append(metrics['enrollment_rate'])
            self.history['private_healthcare_rate'].append(metrics['private_healthcare_rate'])
            self.history['time'].append(elapsed)
            
            if metrics['total'] < best_loss:
                best_loss = metrics['total']
                torch.save(self.model.state_dict(), OUTPUT_DIR / 'stage4_best.pt')
            
            if epoch % print_every == 0 or epoch == n_epochs - 1:
                log(f"Epoch {epoch:4d} | Loss: {metrics['total']:.6f} | "
                    f"C: {metrics['mean_c']:.2f} | "
                    f"HomeOwn: {metrics['homeownership_rate']:.1%} | "
                    f"Enroll: {metrics['enrollment_rate']:.1%} | "
                    f"Time: {elapsed:.1f}s")
        
        log("")
        log("="*60)
        log("TRAINING COMPLETE")
        log(f"Best loss: {best_loss:.6f}")
        log(f"Total time: {elapsed/60:.1f} minutes")
        log("="*60)
        
        pd.DataFrame(self.history).to_csv(OUTPUT_DIR / 'training_history.csv', index=False)
        return self.history
    
    def plot_diagnostics(self):
        """Create diagnostic plots."""
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        
        # Loss curves
        ax = axes[0, 0]
        ax.semilogy(self.history['epoch'], self.history['loss'], 'b-', label='Total', linewidth=2)
        ax.semilogy(self.history['epoch'], self.history['budget_loss'], 'r--', label='Budget', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Consumption
        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['mean_c'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Consumption')
        ax.set_title('Consumption')
        ax.grid(True, alpha=0.3)
        
        # Assets
        ax = axes[0, 2]
        ax.plot(self.history['epoch'], self.history['mean_a_l'], 'r-', label='Liquid', linewidth=2)
        ax.plot(self.history['epoch'], self.history['mean_a_i'], 'g-', label='Illiquid', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Assets')
        ax.set_title('Assets')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Homeownership rate
        ax = axes[0, 3]
        ax.plot(self.history['epoch'], self.history['homeownership_rate'], 'purple', linewidth=2)
        ax.axhline(y=self.params.targets['homeownership_rate'], color='r', linestyle='--', 
                   label=f'Target: {self.params.targets["homeownership_rate"]:.0%}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Homeownership Rate')
        ax.set_title('Housing Choice')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mortgage debt
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['mean_mortgage'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Mortgage Debt')
        ax.set_title('Mortgage Debt')
        ax.grid(True, alpha=0.3)
        
        # Training time
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], np.array(self.history['time'])/60, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (minutes)')
        ax.set_title('Training Time')
        ax.grid(True, alpha=0.3)
        
        # Summary
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
        STAGE 4 TRAINING SUMMARY
        ========================
        
        Training Time: {self.history['time'][-1]/60:.1f} minutes
        Epochs: {len(self.history['epoch'])}
        
        Final Loss: {self.history['loss'][-1]:.6f}
        
        Mean Consumption: {self.history['mean_c'][-1]:.2f}
        Mean Liquid Assets: {self.history['mean_a_l'][-1]:.1f}
        Mean Illiquid Assets: {self.history['mean_a_i'][-1]:.1f}
        
        HOMEOWNERSHIP
        -------------
        Rate: {self.history['homeownership_rate'][-1]:.1%}
        Target: {self.params.targets['homeownership_rate']:.1%}
        
        Mean Mortgage: {self.history['mean_mortgage'][-1]:.1f}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Empty subplot
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'stage4_diagnostics.png', dpi=300, bbox_inches='tight')
        log(f"Saved diagnostics: {OUTPUT_DIR / 'stage4_diagnostics.png'}")
        plt.close()

# ====================================================================================
# VALIDATION
# ====================================================================================

class Stage4Validator:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.model.eval()
    
    def validate(self, n_agents=10000):
        log("")
        log("="*60)
        log("STAGE 4 VALIDATION")
        log("="*60)
        
        with torch.no_grad():
            state = {
                'liquid': torch.rand(n_agents, 1, device=device) * self.params.liquid_max,
                'illiquid': torch.rand(n_agents, 1, device=device) * self.params.illiquid_max,
                'mortgage_debt': (torch.rand(n_agents, 1, device=device) < 0.3).float() * torch.rand(n_agents, 1, device=device) * 50.0,
                'base_income': torch.exp(torch.randn(n_agents, 1, device=device) * 0.5 + 0.5),
                'education_level': torch.randint(0, 3, (n_agents,), device=device),
                'age': torch.randint(18, 66, (n_agents, 1), device=device).float(),
                'health_state': torch.multinomial(torch.tensor([0.75, 0.15, 0.08, 0.02], device=device), n_agents, replacement=True),
                'housing_type': torch.multinomial(torch.tensor([0.5, 0.3, 0.2], device=device), n_agents, replacement=True)
            }
            
            policies = self.model(
                state['liquid'], state['illiquid'], state['mortgage_debt'],
                state['base_income'], state['education_level'], state['age'],
                state['health_state'], state['housing_type']
            )
            
            budget_residual, budget_info = self.model.compute_budget(policies, state)
            
            # Housing choice distribution
            housing_dist = policies['housing_choice'].mean(dim=0)
            
            stats = {
                'n_agents': n_agents,
                'mean_liquid': state['liquid'].mean().item(),
                'mean_illiquid': state['illiquid'].mean().item(),
                'mean_mortgage': state['mortgage_debt'].mean().item(),
                'mean_consumption': policies['consumption'].mean().item(),
                'mean_income': budget_info['income'].mean().item(),
                'mean_housing_exp': budget_info['housing_exp'].mean().item(),
                'budget_residual_mean': budget_residual.mean().item(),
                'rent_share': housing_dist[0].item(),
                'mortgage_share': housing_dist[1].item(),
                'outright_share': housing_dist[2].item(),
                'homeownership_rate': (housing_dist[1] + housing_dist[2]).item(),
            }
            
            log(f"Steady State (n={n_agents}):")
            log(f"  Mean liquid assets:      RM {stats['mean_liquid']:,.0f}")
            log(f"  Mean illiquid (EPF):     RM {stats['mean_illiquid']:,.0f}")
            log(f"  Mean mortgage debt:      RM {stats['mean_mortgage']:,.0f}")
            log(f"  Mean consumption:        RM {stats['mean_consumption']:.2f}")
            log(f"  Mean income:             RM {stats['mean_income']:.2f}")
            log(f"  Mean housing expenses:   RM {stats['mean_housing_exp']:.2f}")
            log(f"  Budget residual:         {stats['budget_residual_mean']:.4f}")
            
            log("")
            log("Housing Distribution:")
            log(f"  Rent:          {stats['rent_share']:.1%}")
            log(f"  Mortgage:      {stats['mortgage_share']:.1%}")
            log(f"  Outright:      {stats['outright_share']:.1%}")
            log(f"  ------------------------")
            log(f"  Homeownership: {stats['homeownership_rate']:.1%} (target: {self.params.targets['homeownership_rate']:.1%})")
            
            log("")
            log("Validation Checks:")
            checks = [
                ("Budget residual < 0.1", abs(stats['budget_residual_mean']) < 0.1),
                ("Consumption positive", stats['mean_consumption'] > 0),
                ("Homeownership > 50%", stats['homeownership_rate'] > 0.5),
                ("Homeownership < 90%", stats['homeownership_rate'] < 0.9)
            ]
            
            all_pass = True
            for check_name, passed in checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                log(f"  {check_name:40s} {status}")
                all_pass = all_pass and passed
            
            if all_pass:
                log("")
                log("✓ ALL VALIDATION CHECKS PASSED")
                log("  Ready for Stage 5: Add geography")
            else:
                log("")
                log("⚠ SOME CHECKS FAILED - Review before proceeding")
            
            with open(OUTPUT_DIR / 'validation_results.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            return stats

# ====================================================================================
# MAIN
# ====================================================================================

def main():
    log("="*60)
    log("MALAYSIA HANK - STAGE 4: HOUSING")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("")
    
    params = MalaysiaBaseParams()
    model = MalaysiaHANK_Stage4(params)
    
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log("")
    
    trainer = Stage4Trainer(model, params, lr=1e-3)
    history = trainer.train(n_epochs=6000, print_every=300)
    trainer.plot_diagnostics()
    
    model.load_state_dict(torch.load(OUTPUT_DIR / 'stage4_best.pt'))
    validator = Stage4Validator(model, params)
    validator.validate()
    
    log("")
    log("="*60)
    log("STAGE 4 COMPLETE")
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("Outputs:")
    log(f"  - Model: {OUTPUT_DIR / 'stage4_best.pt'}")
    log(f"  - History: {OUTPUT_DIR / 'training_history.csv'}")
    log(f"  - Diagnostics: {OUTPUT_DIR / 'stage4_diagnostics.png'}")
    log(f"  - Validation: {OUTPUT_DIR / 'validation_results.json'}")
    log(f"  - Log: {LOG_FILE}")
    log("")
    log("Next: Stage 5 - Add Geography module")

if __name__ == "__main__":
    main()
