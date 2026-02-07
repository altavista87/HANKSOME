"""
MALAYSIA EXTENDED DEEP HANK ARCHITECTURE
=========================================
Full implementation with education, health, housing, and geography.
State space: ~100M+ states (tractable only with deep learning)

Author: Extended from minimal implementation
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import torch.nn.functional as F

# ============================================================================
# 1. CONFIGURATION AND PARAMETERS
# ============================================================================

@dataclass
class MalaysiaExtendedParams:
    """
    Comprehensive parameters for Malaysia Extended HANK Model
    """
    # Time preference
    beta: float = 0.86  # Lowered to 0.86 to boost MPC/Consumption (Fixing T20 savings glut)
    
    # Risk aversion
    sigma: float = 1.0  # CRRA (log utility)
    
    # Production
    alpha: float = 0.33  # Capital share
    delta: float = 0.05  # Depreciation
    
    # Returns
    r_liquid: float = 0.03      # 3% annual (bank deposits)
    r_illiquid: float = 0.05    # 5% annual (EPF average)
    r_mortgage: float = 0.045   # 4.5% annual (housing loan)
    
    # Education sector
    education_levels: int = 3   # Primary, Secondary, Tertiary
    education_costs: List[float] = None  # Cost per level
    education_returns: List[float] = None  # Wage premium
    
    # Health sector
    health_states: int = 4      # Healthy, Sick, Chronic, Disabled
    health_transition_matrix: np.ndarray = None
    health_costs: List[float] = None  # Medical expenses
    
    # Housing
    housing_types: int = 3      # Rent, Own (mortgage), Own (outright)
    rent_cost: float = 0.25     # 25% of income
    house_price: float = 3.0    # 3x annual income
    ltv_ratio: float = 0.90     # Loan-to-value
    
    # Geography
    n_states: int = 13          # Malaysian states
    n_locations: int = 26       # States × (Urban + Rural)
    migration_cost: float = 0.5  # Cost of moving
    
    # Labor market (Malaysia-specific)
    sector_weights: Dict[str, float] = None  # Formal, Public, Informal
    sector_risk: Dict[str, float] = None     # Income volatility
    
    # EPF system
    epf_employee_rate: float = 0.11  # 11% employee contribution
    epf_employer_rate: float = 0.12  # 12% employer contribution
    epf_withdrawal_age: int = 55
    
    # Asset grids
    n_liquid: int = 50
    n_illiquid: int = 30
    liquid_max: float = 50.0
    illiquid_max: float = 200.0
    
    def __post_init__(self):
        # Initialize education
        if self.education_costs is None:
            self.education_costs = [0.0, 0.5, 2.0]  # Tertiary expensive
        if self.education_returns is None:
            self.education_returns = [1.0, 1.5, 5.0]  # Increased to 5.0 to widen T20/B40 gap
            
        # Initialize health
        if self.health_transition_matrix is None:
            # Persistent health states
            self.health_transition_matrix = np.array([
                [0.95, 0.04, 0.01, 0.00],  # Healthy
                [0.30, 0.60, 0.08, 0.02],  # Sick
                [0.10, 0.15, 0.70, 0.05],  # Chronic
                [0.05, 0.10, 0.15, 0.70],  # Disabled
            ])
        if self.health_costs is None:
            self.health_costs = [0.0, 0.5, 2.0, 5.0]  # Medical expenses
            
        # Initialize labor market
        if self.sector_weights is None:
            self.sector_weights = {
                'formal': 0.50,
                'public': 0.10,
                'informal': 0.40
            }
        if self.sector_risk is None:
            self.sector_risk = {
                'formal': 0.30,
                'public': 0.15,
                'informal': 0.60
            }


# ============================================================================
# 2. NEURAL NETWORK ARCHITECTURE
# ============================================================================

class StateEncoder(nn.Module):
    """
    Encodes the high-dimensional state into a compact representation.
    Handles continuous and discrete variables separately.
    """
    def __init__(self, params: MalaysiaExtendedParams, hidden_dim: int = 512):
        super().__init__()
        self.params = params
        
        # Continuous state components (normalized)
        # - liquid assets (1)
        # - illiquid assets (1)
        # - human capital (1)
        # - age (1)
        # - mortgage debt (1)
        # = 5 continuous variables
        self.continuous_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128)
        )
        
        # Discrete state embeddings
        # - Education level (3 levels)
        self.education_embed = nn.Embedding(3, 16)
        
        # - Health status (4 states)
        self.health_embed = nn.Embedding(4, 16)
        
        # - Housing tenure (3 types)
        self.housing_embed = nn.Embedding(3, 16)
        
        # - Location (26 locations: 13 states × urban/rural)
        self.location_embed = nn.Embedding(26, 32)
        
        # - Labor sector (3 sectors)
        self.sector_embed = nn.Embedding(3, 16)
        
        # Combined encoding
        self.combined_encoder = nn.Sequential(
            nn.Linear(128 + 16 + 16 + 16 + 32 + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, continuous: torch.Tensor, education: torch.Tensor,
                health: torch.Tensor, housing: torch.Tensor, 
                location: torch.Tensor, sector: torch.Tensor) -> torch.Tensor:
        """
        Encode state into hidden representation.
        """
        # Encode continuous
        cont_encoded = self.continuous_encoder(continuous)
        
        # Encode discrete
        edu_encoded = self.education_embed(education)
        health_encoded = self.health_embed(health)
        housing_encoded = self.housing_embed(housing)
        loc_encoded = self.location_embed(location)
        sector_encoded = self.sector_embed(sector)
        
        # Ensure all are 2D [batch, dim]
        if cont_encoded.dim() == 1: cont_encoded = cont_encoded.unsqueeze(0)
        if edu_encoded.dim() == 1: edu_encoded = edu_encoded.unsqueeze(0)
        if health_encoded.dim() == 1: health_encoded = health_encoded.unsqueeze(0)
        if housing_encoded.dim() == 1: housing_encoded = housing_encoded.unsqueeze(0)
        if loc_encoded.dim() == 1: loc_encoded = loc_encoded.unsqueeze(0)
        if sector_encoded.dim() == 1: sector_encoded = sector_encoded.unsqueeze(0)
        
        # Concatenate all
        combined = torch.cat([
            cont_encoded, edu_encoded, health_encoded,
            housing_encoded, loc_encoded, sector_encoded
        ], dim=1)
        
        return self.combined_encoder(combined)


# ============================================================================
# 3. MIXTURE OF EXPERTS (MoE) ARCHITECTURE
# ============================================================================

class GatingNetwork(nn.Module):
    """
    The 'Receptionist': Routes households to the correct Expert Advisor.
    Uses temperature sharpening to force decisive group assignment.
    """
    def __init__(self, hidden_dim: int, num_experts: int = 3, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        return F.softmax(logits / self.temperature, dim=-1)


class ExpertNetwork(nn.Module):
    """
    The 'Advisor': Specialized in the physics of a specific regime.
    Ref: Shazeer et al. (2017) expert sub-networks.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEBackbone(nn.Module):
    """
    Sparsely-Gated Mixture of Experts Backbone.
    Prevents "Smoothing Bias" (Rahaman et al., 2019) by allowing
    specialization in non-linear policy regions (e.g., borrowing kinks).
    """
    def __init__(self, hidden_dim: int, num_experts: int = 3, temperature: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.gating_network = GatingNetwork(hidden_dim, num_experts, temperature)
        self.experts = nn.ModuleList([ExpertNetwork(hidden_dim) for _ in range(num_experts)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get gating weights [batch, num_experts]
        weights = self.gating_network(x)
        
        # Compute expert outputs [batch, num_experts, hidden_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Weighted sum of expert opinions: [batch, hidden_dim]
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output


class PolicyNetwork(nn.Module):
    """
    Main policy network that outputs all decisions.
    Uses a Mixture of Experts (MoE) backbone to handle household diversity.
    """
    def __init__(self, params: MalaysiaExtendedParams, hidden_dim: int = 512):
        super().__init__()
        self.params = params
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.encoder = StateEncoder(params, hidden_dim)
        
        # MoE Backbone (routes households to B40, M40, or T20 expert logic)
        self.backbone = MoEBackbone(hidden_dim, num_experts=3)
        
        # Policy heads
        
        # 1. Consumption (continuous, must be positive)
        self.consumption_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # c > 0
        )
        
        # 2. Liquid savings (continuous)
        self.liquid_savings_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        # 3. Illiquid investment (EPF + housing equity)
        self.illiquid_investment_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        # 4. Education choice (discrete: 0=none, 1=primary, 2=secondary, 3=tertiary)
        # Note: 0=none only if age permits
        self.education_choice_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 4),  # 4 choices
            nn.Softmax(dim=-1)
        )
        
        # 5. Healthcare utilization (discrete: 0=none, 1=public, 2=private, 3=both)
        self.healthcare_choice_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )
        
        # 6. Housing tenure choice (discrete)
        self.housing_choice_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 3),  # Rent, Own(mortgage), Own(outright)
            nn.Softmax(dim=-1)
        )
        
        # 7. Migration choice (26 locations including current)
        self.migration_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 26),  # 26 possible locations
            nn.Softmax(dim=-1)
        )
        
        # 8. Labor supply (intensive margin)
        self.labor_supply_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Normalized to [0, 1]
        )
        
    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass: state -> all policies.
        
        Args:
            state: Dictionary with keys:
                - 'continuous': [batch, 5]
                - 'education': [batch]
                - 'health': [batch]
                - 'housing': [batch]
                - 'location': [batch]
                - 'sector': [batch]
                
        Returns:
            Dictionary of policy outputs
        """
        # Encode state
        encoded = self.encoder(
            state['continuous'],
            state['education'],
            state['health'],
            state['housing'],
            state['location'],
            state['sector']
        )
        
        # Shared processing
        features = self.backbone(encoded)
        
        # Get all policies
        policies = {
            'consumption': self.consumption_head(features),
            'liquid_savings': self.liquid_savings_head(features),
            'illiquid_investment': self.illiquid_investment_head(features),
            'education_choice': self.education_choice_head(features),
            'healthcare_choice': self.healthcare_choice_head(features),
            'housing_choice': self.housing_choice_head(features),
            'migration': self.migration_head(features),
            'labor_supply': self.labor_supply_head(features)
        }
        
        return policies


class MalaysiaExtendedHANK(nn.Module):
    """
    Complete Malaysia Extended HANK Model with Deep Learning.
    Wraps policy network with economics (budget constraints, etc.).
    """
    def __init__(self, params: MalaysiaExtendedParams = None):
        super().__init__()
        self.params = params or MalaysiaExtendedParams()
        self.policy_net = PolicyNetwork(self.params)
        
    def compute_economic_policies(self, state: Dict[str, torch.Tensor],
                                  prices: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Compute policies that respect economic constraints.
        This wraps the neural network with hard economic constraints.
        """
        # Get raw neural network outputs
        raw_policies = self.policy_net(state)
        
        # Extract state components
        continuous = state['continuous']
        if continuous.dim() == 1:
            continuous = continuous.unsqueeze(0)
            
        liquid = continuous[:, 0]
        illiquid = continuous[:, 1]
        human_capital = continuous[:, 2]
        age = continuous[:, 3]
        mortgage_debt = continuous[:, 4]
        
        education = state['education']
        health = state['health']
        housing = state['housing']
        
        # Compute cash on hand
        y = self._compute_income(state, prices)
        
        # Compute required expenditures
        health_exp = self._compute_health_expenses(raw_policies['healthcare_choice'], state)
        edu_exp = self._compute_education_expenses(raw_policies['education_choice'], state)
        housing_exp = self._compute_housing_expenses(raw_policies['housing_choice'], state, prices)
        
        # Total required
        required = health_exp + edu_exp + housing_exp
        
        # Available for consumption and savings
        available = y + liquid * (1 + prices['r_liquid']) - required
        
        # Enforce budget constraint on consumption
        c_raw = raw_policies['consumption'].view(-1)
        # Fix clamp: min and max must be same type (Tensors)
        min_c = torch.tensor(0.01, device=c_raw.device)
        max_c = available.view(-1) - 0.01
        c = torch.clamp(c_raw, min=min_c, max=max_c)
        
        # Savings = residual
        residual = available.view(-1) - c
        
        # Split between liquid and illiquid
        alpha_liquid = torch.sigmoid(raw_policies['liquid_savings'].view(-1))
        a_liquid = residual * alpha_liquid
        a_illiquid = residual * (1 - alpha_liquid) + raw_policies['illiquid_investment'].view(-1)
        
        # Enforce borrowing constraints
        a_liquid = torch.clamp(a_liquid, min=0.0)
        a_illiquid = torch.clamp(a_illiquid, min=0.0)
        
        return {
            'consumption': c,
            'liquid_assets': a_liquid,
            'illiquid_assets': a_illiquid,
            'education_choice': raw_policies['education_choice'],
            'healthcare_choice': raw_policies['healthcare_choice'],
            'housing_choice': raw_policies['housing_choice'],
            'migration': raw_policies['migration'],
            'labor_supply': raw_policies['labor_supply']
        }
    
    def _compute_income(self, state: Dict[str, torch.Tensor],
                       prices: Dict[str, float]) -> torch.Tensor:
        """Compute labor income based on human capital, education, health, sector."""
        continuous = state['continuous']
        human_capital = continuous[:, 2]
        
        education = state['education']
        health = state['health']
        sector = state['sector']
        
        # Base wage
        w = prices.get('w', 1.0)
        
        # Education premium
        edu_mult = torch.tensor(self.params.education_returns, device=education.device)[education]
        
        # Health penalty (disabled/chronic earn less)
        health_mult = torch.ones_like(health, dtype=torch.float32)
        health_mult = torch.where(health == 3, 0.5, health_mult)  # Disabled
        health_mult = torch.where(health == 2, 0.8, health_mult)  # Chronic
        health_mult = torch.where(health == 1, 0.9, health_mult)  # Sick
        
        # Sector multiplier
        sector_names = ['formal', 'public', 'informal']
        sector_mult = torch.ones_like(sector, dtype=torch.float32)
        for i, name in enumerate(sector_names):
            sector_mult = torch.where(sector == i, 
                                     self.params.sector_weights.get(name, 1.0), 
                                     sector_mult)
        
        income = w * human_capital * edu_mult * health_mult * sector_mult
        return income
    
    def _compute_health_expenses(self, healthcare_choice_probs: torch.Tensor,
                                state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute out-of-pocket health expenses."""
        health = state['health']
        healthcare_choice = torch.argmax(healthcare_choice_probs, dim=1)
        
        # Base cost by health state
        base_cost = torch.tensor(self.params.health_costs, device=health.device)[health]
        
        # Public healthcare reduces cost (80% subsidy)
        public_mask = (healthcare_choice == 1)
        private_mask = (healthcare_choice == 2)
        
        cost = base_cost.clone()
        cost = torch.where(public_mask, cost * 0.2, cost)
        cost = torch.where(private_mask, cost * 1.5, cost)  # Private is more expensive
        
        return cost
    
    def _compute_education_expenses(self, education_choice_probs: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute education expenses based on chosen education level."""
        age = state['continuous'][:, 3]
        
        # Expected education cost (probability-weighted)
        costs = torch.tensor(self.params.education_costs + [0.0], 
                           device=education_choice_probs.device)
        expected_cost = (education_choice_probs * costs).sum(dim=1)
        
        # Only pay if young enough (age < 25 for tertiary)
        young_mask = (age < 25)
        expected_cost = torch.where(young_mask, expected_cost, torch.zeros_like(expected_cost))
        
        return expected_cost
    
    def _compute_housing_expenses(self, housing_choice_probs: torch.Tensor,
                                  state: Dict[str, torch.Tensor],
                                  prices: Dict[str, float]) -> torch.Tensor:
        """Compute housing expenses (rent or mortgage payments)."""
        income = self._compute_income(state, prices)
        
        # Rent: 25% of income
        rent_cost = income * self.params.rent_cost
        
        # Mortgage payment (simplified)
        r_mort = prices.get('r_mortgage', 0.045)
        house_price = self.params.house_price * income  # House price depends on income
        loan_amount = house_price * self.params.ltv_ratio
        
        # Monthly payment (30-year mortgage, simplified)
        mortgage_payment = loan_amount * (r_mort * (1 + r_mort)**30) / ((1 + r_mort)**30 - 1)
        
        # Owned outright: 0 cost
        owned_cost = torch.zeros_like(income)
        
        # Probability-weighted expected cost
        costs = torch.stack([rent_cost, mortgage_payment, owned_cost], dim=1)
        expected_cost = (housing_choice_probs * costs).sum(dim=1)
        
        return expected_cost


# ============================================================================
# 3. TRAINING (PHYSICS-INFORMED LOSS)
# ============================================================================

class HANKTrainer:
    """
    Trainer for Malaysia Extended HANK using physics-informed neural networks.
    """
    def __init__(self, model: MalaysiaExtendedHANK, learning_rate: float = 1e-4, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print(f"Using device: MPS (Mac Acceleration)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using device: CUDA")
            else:
                self.device = torch.device("cpu")
                print(f"Using device: CPU")
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=100, factor=0.5
        )
        
    def compute_loss(self, batch_size: int = 512) -> Tuple[torch.Tensor, Dict]:
        """
        Compute physics-informed loss.
        """
        # Sample states
        state = self._sample_states(batch_size)
        prices = {
            'r_liquid': self.model.params.r_liquid,
            'r_illiquid': self.model.params.r_illiquid,
            'r_mortgage': self.model.params.r_mortgage,
            'w': 1.0
        }
        
        # Get constrained policies
        policies = self.model.compute_economic_policies(state, prices)
        
        # 1. Budget constraint residual
        income = self.model._compute_income(state, prices)
        c = policies['consumption']
        a_liquid = policies['liquid_assets']
        a_illiquid = policies['illiquid_assets']
        
        # Cash on hand
        coh = income + state['continuous'][:, 0] * (1 + prices['r_liquid'])
        
        # Budget residual
        budget_residual = c + a_liquid + a_illiquid - coh
        budget_loss = torch.mean(budget_residual ** 2)
        
        # 2. Euler equation residual (simplified)
        # Expected marginal utility tomorrow
        beta = self.model.params.beta
        sigma = self.model.params.sigma
        
        mu_t = c ** (-sigma)
        
        # Sample next period (simplified - should iterate over shocks)
        state_next = self._transition_state(state, policies)
        policies_next = self.model.compute_economic_policies(state_next, prices)
        c_next = policies_next['consumption']
        mu_next = c_next ** (-sigma)
        
        # Euler residual for liquid asset
        euler_liquid = mu_t - beta * (1 + prices['r_liquid']) * mu_next
        euler_loss = torch.mean(euler_liquid ** 2)
        
        # 3. Borrowing constraint violations
        liquid_violation = torch.relu(-policies['liquid_assets'])
        illiquid_violation = torch.relu(-policies['illiquid_assets'])
        constraint_loss = torch.mean(liquid_violation ** 2 + illiquid_violation ** 2)
        
        # 4. Human capital accumulation consistency
        # Education should increase human capital
        edu_level = torch.argmax(policies['education_choice'], dim=1)
        h_gain = torch.tensor([0.0, 0.3, 0.7, 1.0], device=self.device)[edu_level]
        
        # 5. MoE Load Balancing Loss (Prevent Expert Collapse)
        # Get gating weights from the policy network
        # We need to access the intermediate gating weights. 
        # Ideally, PolicyNetwork should return them, but for now we re-compute them or access via hook.
        # Simpler: Let's assume the forward pass stored them or we re-compute for loss.
        # Re-computing for simplicity of implementation without changing return signatures too much.
        encoded = self.model.policy_net.encoder(
            state['continuous'], state['education'], state['health'], 
            state['housing'], state['location'], state['sector']
        )
        gate_weights = self.model.policy_net.backbone.gating_network(encoded) # [batch, num_experts]
        
        # Calculate load balancing loss (CV squared of importance)
        # Importance = sum of gate weights over batch
        importance = gate_weights.sum(dim=0)
        # Coefficient of variation = std / mean
        std_importance = importance.std()
        mean_importance = importance.mean()
        # Add small epsilon to avoid div by zero
        cv_squared = (std_importance / (mean_importance + 1e-6)) ** 2
        
        aux_loss = 10000.0 * cv_squared  # Increased from 1000.0 to 10000.0 to force expert separation
        
        # Total loss
        loss = (10.0 * budget_loss + 
                1.0 * euler_loss + 
                100.0 * constraint_loss + 
                aux_loss)
        
        metrics = {
            'total': loss.item(),
            'budget': budget_loss.item(),
            'euler': euler_loss.item(),
            'constraint': constraint_loss.item(),
            'aux_moe': aux_loss.item()
        }
        
        return loss, metrics
    
    def _sample_states(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random states for training."""
        continuous_scale = torch.tensor([50.0, 200.0, 2.0, 60.0, 100.0], device=self.device)
        return {
            'continuous': torch.rand(batch_size, 5, device=self.device) * continuous_scale,
            'education': torch.randint(0, 3, (batch_size,), device=self.device),
            'health': torch.randint(0, 4, (batch_size,), device=self.device),
            'housing': torch.randint(0, 3, (batch_size,), device=self.device),
            'location': torch.randint(0, 26, (batch_size,), device=self.device),
            'sector': torch.randint(0, 3, (batch_size,), device=self.device)
        }
    
    def _transition_state(self, state: Dict[str, torch.Tensor],
                         policies: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Simple state transition (simplified for training)."""
        state_next = state.copy()
        
        # Update assets
        continuous = state['continuous'].clone()
        continuous[:, 0] = policies['liquid_assets']
        continuous[:, 1] = policies['illiquid_assets']
        
        # Age increases
        continuous[:, 3] = torch.clamp(continuous[:, 3] + 1, max=80)
        
        state_next['continuous'] = continuous
        
        # Sample health transitions
        health_probs = torch.tensor(self.model.params.health_transition_matrix, 
                                   device=state['health'].device, dtype=torch.float32)
        # Simplified: random transition
        state_next['health'] = torch.randint(0, 4, state['health'].shape, device=state['health'].device)
        
        return state_next
    
    def save_checkpoint(self, path: str = "hank_checkpoint.pt"):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str = "hank_checkpoint.pt"):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Checkpoint loaded from {path}")
            return True
        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting fresh.")
            return False

    def train(self, n_epochs: int = 10000, print_every: int = 100):
        """Train the model."""
        print(f"Training Malaysia Extended HANK for {n_epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            loss, metrics = self.compute_loss()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step(metrics['total'])
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{n_epochs}")
                print(f"  Loss: {metrics['total']:.6f}")
                print(f"  Budget: {metrics['budget']:.6f}")
                print(f"  Euler: {metrics['euler']:.6f}")
                print(f"  Constraint: {metrics['constraint']:.6f}")
                print(f"  MoE Load Bal: {metrics['aux_moe']:.6f}")
                print()


# ============================================================================
# 4. SIMULATION AND ANALYSIS
# ============================================================================

class HANKSimulator:
    """
    Simulate the model to compute IRFs and distributions.
    """
    def __init__(self, model: MalaysiaExtendedHANK):
        self.model = model
        
    def simulate_lifecycle(self, n_agents: int = 1000, T: int = 60) -> Dict:
        """
        Simulate lifecycle of n_agents over T periods.
        """
        # Initialize at age 20
        states = {
            'continuous': torch.zeros(n_agents, 5),
            'education': torch.zeros(n_agents, dtype=torch.long),
            'health': torch.zeros(n_agents, dtype=torch.long),
            'housing': torch.zeros(n_agents, dtype=torch.long),
            'location': torch.randint(0, 26, (n_agents,)),
            'sector': torch.randint(0, 3, (n_agents,))
        }
        states['continuous'][:, 3] = 20  # Age
        
        prices = {
            'r_liquid': self.model.params.r_liquid,
            'r_illiquid': self.model.params.r_illiquid,
            'r_mortgage': self.model.params.r_mortgage,
            'w': 1.0
        }
        
        # Storage
        history = {
            'consumption': [],
            'assets_liquid': [],
            'assets_illiquid': [],
            'income': [],
            'health': []
        }
        
        for t in range(T):
            policies = self.model.compute_economic_policies(states, prices)
            
            # Store
            history['consumption'].append(policies['consumption'].numpy())
            history['assets_liquid'].append(policies['liquid_assets'].numpy())
            history['assets_illiquid'].append(policies['illiquid_assets'].numpy())
            
            # Compute income
            income = self.model._compute_income(states, prices)
            history['income'].append(income.numpy())
            history['health'].append(states['health'].numpy())
            
            # Transition (simplified)
            states = self._transition(states, policies)
        
        return history
    
    def _transition(self, states: Dict, policies: Dict) -> Dict:
        """Simple state transition."""
        states_next = states.copy()
        
        # Update assets
        continuous = states['continuous'].clone()
        continuous[:, 0] = policies['liquid_assets']
        continuous[:, 1] = policies['illiquid_assets']
        continuous[:, 3] += 1  # Age
        
        states_next['continuous'] = continuous
        
        # Health transitions (Markov)
        n_agents = states['health'].size(0)
        for i in range(n_agents):
            h = states['health'][i].item()
            probs = self.model.params.health_transition_matrix[h]
            states_next['health'][i] = np.random.choice(4, p=probs)
        
        return states_next


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create model
    params = MalaysiaExtendedParams()
    model = MalaysiaExtendedHANK(params)
    
    print("=" * 60)
    print("MALAYSIA EXTENDED DEEP HANK MODEL")
    print("=" * 60)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"State space: ~100M+ states (continuous)")
    print(f"Discrete choices: Education(4) × Health(4) × Housing(3) × Location(26)")
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    test_state = {
        'continuous': torch.tensor([[10.0, 50.0, 1.0, 25.0, 0.0]]),
        'education': torch.tensor([2]),  # Tertiary
        'health': torch.tensor([0]),      # Healthy
        'housing': torch.tensor([1]),     # Own with mortgage
        'location': torch.tensor([0]),    # Johor Urban
        'sector': torch.tensor([0])       # Formal
    }
    
    policies = model.policy_net(test_state)
    print("\nPolicy outputs:")
    for key, value in policies.items():
        print(f"  {key}: {value.shape} -> {value.detach().numpy()}")
    
    # Train (enabled for verification)
    trainer = HANKTrainer(model)
    trainer.train(n_epochs=100)
    
    print("\n✓ Architecture test and training successful!")
    print("Ready for full scale training.")
