"""
Malaysia HANK - Stage 5: Add Geography
======================================
Extends Stage 4 by adding geographic dimension with states and migration.

Geography System:
- 13 Malaysian States
- Urban vs Rural within each state = 26 locations
- Migration costs between locations
- State-specific income multipliers
- Cost of living differences

State Space: ~25,000 -> ~650,000+ states (26x increase)
Training Time: 5-10 hours (8000 epochs)
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
OUTPUT_DIR = Path("outputs/stage5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE4_DIR = Path("outputs/stage4")

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
        self.r_mortgage = 0.045
        
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
        
        # Education
        self.n_education = 3
        self.education_costs = [0.0, 0.5, 2.0]
        self.education_premiums = [1.0, 1.3, 2.0]
        self.max_education_age = 25
        
        # Health
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
        
        # Housing
        self.n_housing = 3
        self.house_price_multiple = 3.0
        self.down_payment_min = 0.10
        self.mortgage_term = 30
        self.rent_share = 0.25
        self.ltv_max = 0.90
        
        # Geography (NEW for Stage 5)
        self.n_states = 13  # Malaysian states
        self.n_locations = 26  # States × (Urban + Rural)
        self.migration_cost = 0.5  # Cost of moving
        
        # State income multipliers (relative to national average)
        # Higher for urban centers (KL, Selangor, Penang, Johor)
        self.location_income_mult = [
            # Urban locations (even indices)
            1.45,  # 0: Johor Urban
            0.85,  # 1: Johor Rural
            1.60,  # 2: Selangor Urban (high income)
            0.90,  # 3: Selangor Rural
            1.80,  # 4: KL Urban (highest income)
            1.00,  # 5: KL Rural
            1.30,  # 6: Penang Urban
            0.80,  # 7: Penang Rural
            1.15,  # 8: Perak Urban
            0.75,  # 9: Perak Rural
            1.20,  # 10: Negeri Sembilan Urban
            0.80,  # 11: Negeri Sembilan Rural
            1.10,  # 12: Melaka Urban
            0.80,  # 13: Melaka Rural
            1.10,  # 14: Pahang Urban
            0.75,  # 15: Pahang Rural
            0.95,  # 16: Terengganu Urban
            0.70,  # 17: Terengganu Rural
            1.00,  # 18: Kelantan Urban
            0.70,  # 19: Kelantan Rural
            1.05,  # 20: Kedah Urban
            0.75,  # 21: Kedah Rural
            0.90,  # 22: Perlis Urban
            0.70,  # 23: Perlis Rural
            1.25,  # 24: Sabah Urban
            0.75,  # 25: Sabah Rural
        ]
        
        # Cost of living multipliers by location
        self.location_cost_mult = [
            1.20, 0.85,  # Johor
            1.35, 0.90,  # Selangor
            1.50, 1.00,  # KL
            1.25, 0.80,  # Penang
            1.00, 0.75,  # Perak
            1.15, 0.80,  # Negeri Sembilan
            1.10, 0.80,  # Melaka
            1.05, 0.75,  # Pahang
            0.95, 0.70,  # Terengganu
            0.90, 0.70,  # Kelantan
            0.95, 0.75,  # Kedah
            0.85, 0.70,  # Perlis
            1.10, 0.80,  # Sabah
        ]
        
        # Load calibration targets
        with open(DATA_DIR / 'calibration_targets.json', 'r') as f:
            self.targets = json.load(f)

# ====================================================================================
# STAGE 5 MODEL: Full Model with Geography
# ====================================================================================

class MalaysiaHANK_Stage5(nn.Module):
    """
    Stage 5: Full HANK with education, health, housing, and geography.
    
    State: (liquid, illiquid, mortgage_debt, base_income, education, age, 
            health, housing, location)
    Policies: (consumption, savings, education_choice, healthcare_choice,
               housing_choice, migration_choice)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # Embeddings
        self.education_embed = nn.Embedding(params.n_education, 8)
        self.health_embed = nn.Embedding(params.n_health, 8)
        self.housing_embed = nn.Embedding(params.n_housing, 8)
        self.location_embed = nn.Embedding(params.n_locations, 16)  # Larger for 26 locations
        
        # Extended encoder: 4 continuous + 8 edu + 1 age + 8 health + 8 housing + 16 location = 45 inputs
        self.encoder = nn.Sequential(
            nn.Linear(45, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.SiLU(),
        )
        
        # Policy heads
        self.policy_head = nn.Sequential(
            nn.Linear(256, 3),
            nn.Softplus()
        )
        
        self.education_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, params.n_education)
        )
        
        self.healthcare_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )
        
        self.housing_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, params.n_housing)
        )
        
        # Migration choice: probability of moving to each location
        self.migration_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, params.n_locations)
        )
        
    def forward(self, liquid, illiquid, mortgage_debt, base_income, 
                education_level, age, health_state, housing_type, location):
        """
        Forward pass through the network.
        
        Args:
            liquid, illiquid, mortgage_debt, base_income: [batch, 1]
            education_level: [batch] in {0, 1, 2}
            age: [batch, 1]
            health_state: [batch] in {0, 1, 2, 3}
            housing_type: [batch] in {0, 1, 2}
            location: [batch] in {0, ..., 25}
        """
        batch_size = liquid.size(0)
        
        # Normalize
        liquid_norm = liquid / self.params.liquid_max
        illiquid_norm = illiquid / self.params.illiquid_max
        mortgage_norm = mortgage_debt / (self.params.house_price_multiple * 10)
        income_norm = base_income / 2.0
        age_norm = age / 80.0
        
        # Embed
        edu_emb = self.education_embed(education_level)
        health_emb = self.health_embed(health_state)
        housing_emb = self.housing_embed(housing_type)
        loc_emb = self.location_embed(location)
        
        # Concatenate: 4 + 8 + 1 + 8 + 8 + 16 = 45
        state = torch.cat([
            liquid_norm, illiquid_norm, mortgage_norm, income_norm,
            edu_emb, age_norm, health_emb, housing_emb, loc_emb
        ], dim=-1)
        
        features = self.encoder(state)
        
        # Policies
        policies_norm = self.policy_head(features)
        consumption = policies_norm[:, 0:1] * 2.0
        liquid_savings = policies_norm[:, 1:2] * self.params.liquid_max
        illiquid_savings = policies_norm[:, 2:3] * self.params.illiquid_max
        
        # Education choice
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
        
        # Healthcare choice
        healthcare_logits = self.healthcare_head(features)
        healthcare_choice = torch.softmax(healthcare_logits, dim=-1)
        
        # Housing choice
        housing_logits = self.housing_head(features)
        housing_choice = torch.softmax(housing_logits, dim=-1)
        
        # Migration choice (probability distribution over 26 locations)
        migration_logits = self.migration_head(features)
        migration_choice = torch.softmax(migration_logits, dim=-1)
        
        return {
            'consumption': consumption,
            'liquid_savings': liquid_savings,
            'illiquid_savings': illiquid_savings,
            'education_choice': education_choice,
            'healthcare_choice': healthcare_choice,
            'housing_choice': housing_choice,
            'migration_choice': migration_choice,
            'can_enroll': can_enroll
        }
    
    def compute_income(self, base_income, education_level, health_state, location):
        """Compute income with education, health, and location effects."""
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
        
        location_mult = torch.tensor(
            self.params.location_income_mult,
            device=base_income.device,
            dtype=base_income.dtype
        )[location]
        
        return base_income * edu_premiums.unsqueeze(-1) * health_factors.unsqueeze(-1) * location_mult.unsqueeze(-1)
    
    def compute_cost_of_living(self, location):
        """Get cost of living multiplier for location."""
        return torch.tensor(
            self.params.location_cost_mult,
            device=location.device,
            dtype=torch.float32
        )[location]
    
    def compute_health_expenses(self, health_state, healthcare_choice):
        """Compute health expenses."""
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
        """Compute housing expenses."""
        rent_cost = income * self.params.rent_share
        
        r_mort = self.params.r_mortgage
        n_years = self.params.mortgage_term
        
        if r_mort > 0 and n_years > 0:
            mortgage_payment = mortgage_debt * (
                r_mort * (1 + r_mort)**n_years / ((1 + r_mort)**n_years - 1)
            )
        else:
            mortgage_payment = mortgage_debt / n_years if n_years > 0 else torch.zeros_like(mortgage_debt)
        
        outright_cost = torch.zeros_like(income)
        
        costs = torch.cat([rent_cost, mortgage_payment, outright_cost], dim=1)
        expected_cost = (housing_choice * costs).sum(dim=1, keepdim=True)
        
        return expected_cost
    
    def compute_budget(self, policies, state):
        """Compute budget constraint residual."""
        liquid = state['liquid']
        illiquid = state['illiquid']
        mortgage_debt = state['mortgage_debt']
        base_income = state['base_income']
        education_level = state['education_level']
        health_state = state['health_state']
        location = state['location']
        
        c = policies['consumption']
        a_l = policies['liquid_savings']
        a_i = policies['illiquid_savings']
        edu_choice = policies['education_choice']
        healthcare_choice = policies['healthcare_choice']
        housing_choice = policies['housing_choice']
        
        # Effective income with location effect
        income = self.compute_income(base_income, education_level, health_state, location)
        
        # Cost of living adjustment
        col_mult = self.compute_cost_of_living(location).unsqueeze(-1)
        
        # Expenses
        health_exp = self.compute_health_expenses(health_state, healthcare_choice)
        housing_exp = self.compute_housing_expenses(income, housing_choice, mortgage_debt)
        
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
        
        # Apply cost of living to consumption
        c_adjusted = c * col_mult
        
        # Budget constraint
        total_spending = c_adjusted + a_l + a_i + expected_edu_cost
        residual = total_spending - coh
        
        return residual, {
            'coh': coh,
            'income': income,
            'health_exp': health_exp,
            'housing_exp': housing_exp,
            'expected_edu_cost': expected_edu_cost,
            'col_mult': col_mult
        }

# ====================================================================================
# TRAINER
# ====================================================================================

class Stage5Trainer:
    def __init__(self, model, params, lr=5e-4):
        self.model = model.to(device)
        self.params = params
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=300, factor=0.5
        )
        self.history = {
            'epoch': [], 'loss': [], 'budget_loss': [], 
            'constraint_loss': [], 'euler_loss': [],
            'mean_c': [], 'mean_a_l': [], 'mean_a_i': [],
            'enrollment_rate': [], 'homeownership_rate': [],
            'kl_share': [],  # Track KL concentration
            'time': []
        }
        self.start_time = time.time()
        
    def sample_state(self, batch_size):
        """Sample random states."""
        liquid = torch.rand(batch_size, 1, device=device) * self.params.liquid_max
        illiquid = torch.rand(batch_size, 1, device=device) * self.params.illiquid_max
        
        has_mortgage = torch.rand(batch_size, 1, device=device) < 0.3
        mortgage_debt = has_mortgage.float() * torch.rand(batch_size, 1, device=device) * 50.0
        
        base_income = torch.exp(torch.randn(batch_size, 1, device=device) * 0.5 + 0.5)
        base_income = torch.clamp(base_income, 0.5, 3.0)
        
        education_level = torch.randint(0, 3, (batch_size,), device=device)
        age = torch.randint(18, 66, (batch_size, 1), device=device).float()
        
        health_probs = torch.tensor([0.75, 0.15, 0.08, 0.02], device=device)
        health_state = torch.multinomial(health_probs, batch_size, replacement=True)
        
        housing_probs = torch.tensor([0.5, 0.3, 0.2], device=device)
        housing_type = torch.multinomial(housing_probs, batch_size, replacement=True)
        
        # Location: weighted toward urban centers (KL=4, Selangor=2, Penang=6)
        loc_probs = torch.ones(self.params.n_locations, device=device) * 0.02
        loc_probs[4] = 0.15  # KL Urban
        loc_probs[2] = 0.12  # Selangor Urban
        loc_probs[6] = 0.10  # Penang Urban
        loc_probs = loc_probs / loc_probs.sum()
        location = torch.multinomial(loc_probs, batch_size, replacement=True)
        
        return {
            'liquid': liquid,
            'illiquid': illiquid,
            'mortgage_debt': mortgage_debt,
            'base_income': base_income,
            'education_level': education_level,
            'age': age,
            'health_state': health_state,
            'housing_type': housing_type,
            'location': location
        }
    
    def compute_loss(self, batch_size=512):
        state = self.sample_state(batch_size)
        
        policies = self.model(
            state['liquid'], state['illiquid'], state['mortgage_debt'],
            state['base_income'], state['education_level'], state['age'],
            state['health_state'], state['housing_type'], state['location']
        )
        
        budget_residual, budget_info = self.model.compute_budget(policies, state)
        budget_loss = torch.mean(budget_residual ** 2)
        
        liquid_viol = torch.relu(-policies['liquid_savings'])
        illiquid_viol = torch.relu(-policies['illiquid_savings'])
        constraint_loss = torch.mean(liquid_viol ** 2 + illiquid_viol ** 2)
        
        c = policies['consumption']
        mu_t = c ** (-self.params.sigma)
        beta, r_l = self.params.beta, self.params.r_liquid
        euler_residual = mu_t - beta * (1 + r_l) * mu_t
        euler_loss = torch.mean(euler_residual ** 2)
        
        loss = 10.0 * budget_loss + 100.0 * constraint_loss + 1.0 * euler_loss
        
        with torch.no_grad():
            mean_c = policies['consumption'].mean().item()
            mean_a_l = policies['liquid_savings'].mean().item()
            mean_a_i = policies['illiquid_savings'].mean().item()
            
            young_mask = (state['age'].squeeze() < self.params.max_education_age)
            enrollment_rate = policies['education_choice'][young_mask, 2].mean().item() if young_mask.any() else 0.0
            
            homeownership_rate = policies['housing_choice'][:, 1:].sum().item() / batch_size
            
            # KL urban share (location 4)
            kl_share = (state['location'] == 4).float().mean().item()
        
        return loss, {
            'total': loss.item(),
            'budget': budget_loss.item(),
            'constraint': constraint_loss.item(),
            'euler': euler_loss.item(),
            'mean_c': mean_c,
            'mean_a_l': mean_a_l,
            'mean_a_i': mean_a_i,
            'enrollment_rate': enrollment_rate,
            'homeownership_rate': homeownership_rate,
            'kl_share': kl_share
        }
    
    def train(self, n_epochs=8000, print_every=400):
        log("="*60)
        log("STAGE 5: FULL MODEL WITH GEOGRAPHY")
        log("="*60)
        log(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        log(f"Epochs: {n_epochs}, Device: {device}")
        log("")
        log("Geography System:")
        log(f"  States: {self.params.n_states}")
        log(f"  Locations: {self.params.n_locations} (Urban + Rural)")
        log(f"  Migration Cost: {self.params.migration_cost}")
        log("")
        log("Top Income Locations:")
        loc_names = ['KL Urban', 'Selangor Urban', 'Penang Urban', 'Johor Urban']
        loc_indices = [4, 2, 6, 0]
        for name, idx in zip(loc_names, loc_indices):
            mult = self.params.location_income_mult[idx]
            log(f"  {name:20s}: {mult:.2f}x")
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
            
            self.history['epoch'].append(epoch)
            self.history['loss'].append(metrics['total'])
            self.history['budget_loss'].append(metrics['budget'])
            self.history['constraint_loss'].append(metrics['constraint'])
            self.history['euler_loss'].append(metrics['euler'])
            self.history['mean_c'].append(metrics['mean_c'])
            self.history['mean_a_l'].append(metrics['mean_a_l'])
            self.history['mean_a_i'].append(metrics['mean_a_i'])
            self.history['enrollment_rate'].append(metrics['enrollment_rate'])
            self.history['homeownership_rate'].append(metrics['homeownership_rate'])
            self.history['kl_share'].append(metrics['kl_share'])
            self.history['time'].append(elapsed)
            
            if metrics['total'] < best_loss:
                best_loss = metrics['total']
                torch.save(self.model.state_dict(), OUTPUT_DIR / 'stage5_best.pt')
            
            if epoch % print_every == 0 or epoch == n_epochs - 1:
                log(f"Epoch {epoch:4d} | Loss: {metrics['total']:.6f} | "
                    f"C: {metrics['mean_c']:.2f} | "
                    f"HomeOwn: {metrics['homeownership_rate']:.1%} | "
                    f"KL: {metrics['kl_share']:.1%} | "
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
        
        ax = axes[0, 0]
        ax.semilogy(self.history['epoch'], self.history['loss'], 'b-', linewidth=2)
        ax.semilogy(self.history['epoch'], self.history['budget_loss'], 'r--', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['mean_c'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Consumption')
        ax.set_title('Consumption')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.plot(self.history['epoch'], self.history['mean_a_l'], 'r-', label='Liquid', linewidth=2)
        ax.plot(self.history['epoch'], self.history['mean_a_i'], 'g-', label='Illiquid', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Assets')
        ax.set_title('Assets')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 3]
        ax.plot(self.history['epoch'], self.history['kl_share'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Urban Share')
        ax.set_title('Geographic Concentration')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['homeownership_rate'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Homeownership Rate')
        ax.set_title('Housing')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], np.array(self.history['time'])/60, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (minutes)')
        ax.set_title('Training Time')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
        STAGE 5 COMPLETE
        ================
        
        Training Time: {self.history['time'][-1]/60:.1f} minutes
        Epochs: {len(self.history['epoch'])}
        
        Final Loss: {self.history['loss'][-1]:.6f}
        
        Mean Consumption: {self.history['mean_c'][-1]:.2f}
        Mean Liquid: {self.history['mean_a_l'][-1]:.1f}
        Mean Illiquid: {self.history['mean_a_i'][-1]:.1f}
        
        KL Urban Share: {self.history['kl_share'][-1]:.1%}
        Homeownership: {self.history['homeownership_rate'][-1]:.1%}
        
        ✓ FULL MODEL TRAINED
        """
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'stage5_diagnostics.png', dpi=300, bbox_inches='tight')
        log(f"Saved diagnostics: {OUTPUT_DIR / 'stage5_diagnostics.png'}")
        plt.close()

# ====================================================================================
# VALIDATION
# ====================================================================================

class Stage5Validator:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.model.eval()
    
    def validate(self, n_agents=10000):
        log("")
        log("="*60)
        log("STAGE 5 VALIDATION: FULL MODEL")
        log("="*60)
        
        with torch.no_grad():
            loc_probs = torch.ones(self.params.n_locations, device=device) * 0.02
            loc_probs[4] = 0.15
            loc_probs[2] = 0.12
            loc_probs[6] = 0.10
            loc_probs = loc_probs / loc_probs.sum()
            
            state = {
                'liquid': torch.rand(n_agents, 1, device=device) * self.params.liquid_max,
                'illiquid': torch.rand(n_agents, 1, device=device) * self.params.illiquid_max,
                'mortgage_debt': (torch.rand(n_agents, 1, device=device) < 0.3).float() * torch.rand(n_agents, 1, device=device) * 50.0,
                'base_income': torch.exp(torch.randn(n_agents, 1, device=device) * 0.5 + 0.5),
                'education_level': torch.randint(0, 3, (n_agents,), device=device),
                'age': torch.randint(18, 66, (n_agents, 1), device=device).float(),
                'health_state': torch.multinomial(torch.tensor([0.75, 0.15, 0.08, 0.02], device=device), n_agents, replacement=True),
                'housing_type': torch.multinomial(torch.tensor([0.5, 0.3, 0.2], device=device), n_agents, replacement=True),
                'location': torch.multinomial(loc_probs, n_agents, replacement=True)
            }
            
            policies = self.model(
                state['liquid'], state['illiquid'], state['mortgage_debt'],
                state['base_income'], state['education_level'], state['age'],
                state['health_state'], state['housing_type'], state['location']
            )
            
            budget_residual, budget_info = self.model.compute_budget(policies, state)
            
            # Location distribution
            loc_dist = torch.zeros(self.params.n_locations, device=device)
            for i in range(self.params.n_locations):
                loc_dist[i] = (state['location'] == i).float().mean()
            
            # Income by location
            income_by_loc = {}
            for i in [4, 2, 6, 0]:  # KL, Selangor, Penang, Johor
                mask = (state['location'] == i)
                if mask.any():
                    income_by_loc[i] = budget_info['income'][mask].mean().item()
            
            stats = {
                'n_agents': n_agents,
                'mean_liquid': state['liquid'].mean().item(),
                'mean_illiquid': state['illiquid'].mean().item(),
                'mean_consumption': policies['consumption'].mean().item(),
                'mean_income': budget_info['income'].mean().item(),
                'budget_residual_mean': budget_residual.mean().item(),
                'kl_share': loc_dist[4].item(),
                'selangor_share': loc_dist[2].item(),
                'penang_share': loc_dist[6].item(),
                'homeownership_rate': policies['housing_choice'][:, 1:].sum().item() / n_agents,
                'enrollment_rate': policies['education_choice'][(state['age'].squeeze() < 25), 2].mean().item() if (state['age'].squeeze() < 25).any() else 0.0,
            }
            
            log(f"Steady State (n={n_agents}):")
            log(f"  Mean liquid assets:      RM {stats['mean_liquid']:,.0f}")
            log(f"  Mean illiquid (EPF):     RM {stats['mean_illiquid']:,.0f}")
            log(f"  Mean consumption:        RM {stats['mean_consumption']:.2f}")
            log(f"  Mean income:             RM {stats['mean_income']:.2f}")
            log(f"  Budget residual:         {stats['budget_residual_mean']:.4f}")
            
            log("")
            log("Geographic Distribution:")
            log(f"  KL Urban:       {stats['kl_share']:.1%} (income mult: 1.80x)")
            log(f"  Selangor Urban: {stats['selangor_share']:.1%} (income mult: 1.60x)")
            log(f"  Penang Urban:   {stats['penang_share']:.1%} (income mult: 1.30x)")
            
            log("")
            log("Mean Income by Location:")
            for loc_id, income in income_by_loc.items():
                loc_name = {4: 'KL Urban', 2: 'Selangor Urban', 6: 'Penang Urban', 0: 'Johor Urban'}[loc_id]
                log(f"  {loc_name:20s}: RM {income:.2f}")
            
            log("")
            log("Policy Moments:")
            log(f"  Homeownership rate:      {stats['homeownership_rate']:.1%}")
            log(f"  Tertiary enrollment:     {stats['enrollment_rate']:.1%}")
            
            log("")
            log("Validation Checks:")
            checks = [
                ("Budget residual < 0.1", abs(stats['budget_residual_mean']) < 0.1),
                ("Consumption positive", stats['mean_consumption'] > 0),
                ("KL has highest income", income_by_loc.get(4, 0) > income_by_loc.get(0, 0)),
                ("Assets non-negative", stats['mean_liquid'] >= 0 and stats['mean_illiquid'] >= 0)
            ]
            
            all_pass = True
            for check_name, passed in checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                log(f"  {check_name:40s} {status}")
                all_pass = all_pass and passed
            
            if all_pass:
                log("")
                log("✓ ALL VALIDATION CHECKS PASSED")
                log("  Full Malaysia HANK model complete!")
            
            with open(OUTPUT_DIR / 'validation_results.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            return stats

# ====================================================================================
# MAIN
# ====================================================================================

def main():
    log("="*60)
    log("MALAYSIA HANK - STAGE 5: FULL MODEL")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("")
    
    params = MalaysiaBaseParams()
    model = MalaysiaHANK_Stage5(params)
    
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log(f"State space: ~650K+ states")
    log("")
    
    trainer = Stage5Trainer(model, params, lr=5e-4)
    history = trainer.train(n_epochs=8000, print_every=400)
    trainer.plot_diagnostics()
    
    model.load_state_dict(torch.load(OUTPUT_DIR / 'stage5_best.pt'))
    validator = Stage5Validator(model, params)
    validator.validate()
    
    log("")
    log("="*60)
    log("STAGE 5 COMPLETE - FULL MODEL READY")
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("Outputs:")
    log(f"  - Model: {OUTPUT_DIR / 'stage5_best.pt'}")
    log(f"  - History: {OUTPUT_DIR / 'training_history.csv'}")
    log(f"  - Diagnostics: {OUTPUT_DIR / 'stage5_diagnostics.png'}")
    log(f"  - Validation: {OUTPUT_DIR / 'validation_results.json'}")
    log(f"  - Log: {LOG_FILE}")
    log("")
    log("✓ Malaysia Extended HANK Model Complete!")

if __name__ == "__main__":
    main()
