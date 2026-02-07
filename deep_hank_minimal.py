"""
Minimal Deep HANK for Malaysia - Starting Point
Solves basic HANK with neural network (can be extended)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleHANKNet(nn.Module):
    """
    Simple 2-asset HANK network
    Can be extended to full Malaysia model
    """
    def __init__(self, n_income_shocks: int = 7, hidden_dim: int = 256):
        super().__init__()
        
        self.n_income = n_income_shocks
        
        # Encode income state (one-hot or embedding)
        self.income_embed = nn.Embedding(n_income_shocks, 16)
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(18, hidden_dim),  # liquid + illiquid + income_embed
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        
        # Policy outputs
        # 1. Consumption (must be positive)
        self.c_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # 2. Liquid asset choice
        self.a_liquid_layer = nn.Linear(hidden_dim // 2, 1)
        
        # 3. Illiquid asset choice
        self.a_illiquid_layer = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, liquid: torch.Tensor, illiquid: torch.Tensor, 
                income_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            liquid: [batch, 1] liquid assets
            illiquid: [batch, 1] illiquid assets  
            income_idx: [batch] income state index (0 to n_income-1)
            
        Returns:
            Dictionary with consumption and asset policies
        """
        # Embed income
        income_emb = self.income_embed(income_idx)  # [batch, 16]
        
        # Concatenate state
        state = torch.cat([liquid, illiquid, income_emb], dim=1)  # [batch, 18]
        
        # Forward
        features = self.network(state)
        
        # Policies
        c = self.c_layer(features)  # [batch, 1]
        a_liquid = self.a_liquid_layer(features)  # [batch, 1]
        a_illiquid = self.a_illiquid_layer(features)  # [batch, 1]
        
        return {
            'consumption': c,
            'a_liquid': a_liquid,
            'a_illiquid': a_illiquid
        }

class HANKParameters:
    """Parameters for Malaysia HANK"""
    def __init__(self):
        # Preferences
        self.beta = 0.94  # Discount factor
        self.sigma = 1.0  # CRRA (log utility)
        
        # Returns
        self.r_liquid = 0.03  # 3% annual
        self.r_illiquid = 0.05  # 5% (higher return, perhaps EPF)
        
        # Income process (Rouwenhorst)
        self.rho_e = 0.9
        self.sigma_e = 0.4
        self.n_e = 7
        
        # Borrowing constraints
        self.a_liquid_min = 0.0
        self.a_illiquid_min = 0.0
        
        # Adjustment cost for illiquid asset
        self.chi = 0.1  # Cost of moving between liquid/illiquid
        
        # Wage (normalized)
        self.w = 1.0
        
        # Generate income grid and transition matrix
        self.y_grid, self.Pi = self._create_income_process()
    
    def _create_income_process(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rouwenhorst discretization"""
        rho, sigma, n = self.rho_e, self.sigma_e, self.n_e
        
        # Create grid
        sigma_y = sigma / np.sqrt(1 - rho**2)
        psi = np.sqrt(n - 1) * sigma_y
        y_log = np.linspace(-psi, psi, n)
        
        # Transition matrix
        p = (1 + rho) / 2
        Pi = np.array([[p, 1-p], [1-p, p]], dtype=np.float32)
        
        for i in range(2, n):
            Pi_old = Pi
            len_prev = i
            
            z1 = np.zeros((len_prev + 1, len_prev + 1))
            z1[:len_prev, :len_prev] = p * Pi_old
            
            z2 = np.zeros((len_prev + 1, len_prev + 1))
            z2[:len_prev, 1:] = (1 - p) * Pi_old
            
            z3 = np.zeros((len_prev + 1, len_prev + 1))
            z3[1:, :len_prev] = (1 - p) * Pi_old
            
            z4 = np.zeros((len_prev + 1, len_prev + 1))
            z4[1:, 1:] = p * Pi_old
            
            Pi = z1 + z2 + z3 + z4
            Pi[1:-1, :] /= 2
        
        # Convert to levels and normalize
        y_grid = np.exp(y_log)
        y_grid = y_grid / np.dot(self._stationary_dist(Pi), y_grid)
        
        return torch.tensor(y_grid, dtype=torch.float32), torch.tensor(Pi, dtype=torch.float32)
    
    def _stationary_dist(self, Pi: np.ndarray) -> np.ndarray:
        """Compute stationary distribution"""
        eigenvals, eigenvecs = np.linalg.eig(Pi.T)
        idx = np.argmin(np.abs(eigenvals - 1))
        dist = np.real(eigenvecs[:, idx])
        return dist / np.sum(dist)

def compute_loss(model: SimpleHANKNet, params: HANKParameters, 
                 batch_size: int = 1024) -> Tuple[torch.Tensor, Dict]:
    """
    Compute physics-informed loss
    """
    # Sample states uniformly
    liquid = torch.rand(batch_size, 1, device=device) * 20  # 0 to 20
    illiquid = torch.rand(batch_size, 1, device=device) * 50  # 0 to 50
    income_idx = torch.randint(0, params.n_e, (batch_size,), device=device)
    
    # Get policies
    policies = model(liquid, illiquid, income_idx)
    c = policies['consumption']
    a_liquid = policies['a_liquid']
    a_illiquid = policies['a_illiquid']
    
    # Income
    y = params.y_grid[income_idx].unsqueeze(1).to(device)
    
    # Cash on hand
    R_l = 1 + params.r_liquid
    R_i = 1 + params.r_illiquid
    coh = R_l * liquid + R_i * illiquid + params.w * y
    
    # Budget constraint residual
    budget_residual = c + a_liquid + a_illiquid - coh
    budget_loss = torch.mean(budget_residual ** 2)
    
    # Borrowing constraint violations
    liquid_violation = torch.relu(params.a_liquid_min - a_liquid)
    illiquid_violation = torch.relu(params.a_illiquid_min - a_illiquid)
    constraint_loss = torch.mean(liquid_violation ** 2 + illiquid_violation ** 2)
    
    # Euler equation (simplified - next period expectation)
    # Sample next period income
    Pi_batch = params.Pi[income_idx].to(device)  # [batch, n_e]
    
    # For each possible next income
    euler_residuals = []
    for i in range(params.n_e):
        y_next = params.y_grid[i].to(device)
        prob = Pi_batch[:, i]
        
        # Next period state
        a_l_next = torch.clamp(a_liquid, min=params.a_liquid_min)
        a_i_next = torch.clamp(a_illiquid, min=params.a_illiquid_min)
        income_next = torch.full((batch_size,), i, dtype=torch.long, device=device)
        
        # Next period consumption
        policies_next = model(a_l_next, a_i_next, income_next)
        c_next = policies_next['consumption']
        
        # Marginal utility
        mu_next = c_next ** (-params.sigma)
        
        # Expected marginal utility weighted by probability
        euler_residuals.append(prob.unsqueeze(1) * mu_next)
    
    E_mu_next = torch.sum(torch.stack(euler_residuals), dim=0)
    
    # Euler residual for both assets
    mu_t = c ** (-params.sigma)
    
    # Liquid asset Euler
    euler_liquid = mu_t - params.beta * R_l * E_mu_next
    euler_loss_l = torch.mean(euler_liquid ** 2)
    
    # Illiquid asset Euler (with adjustment cost consideration)
    euler_illiquid = mu_t - params.beta * R_i * E_mu_next
    euler_loss_i = torch.mean(euler_illiquid ** 2)
    
    euler_loss = euler_loss_l + euler_loss_i
    
    # Total loss
    loss = 10.0 * budget_loss + 100.0 * constraint_loss + 1.0 * euler_loss
    
    metrics = {
        'total': loss.item(),
        'budget': budget_loss.item(),
        'constraint': constraint_loss.item(),
        'euler': euler_loss.item()
    }
    
    return loss, metrics

def train_deep_hank(n_epochs: int = 5000, print_every: int = 500):
    """Train the deep HANK model"""
    
    # Initialize
    params = HANKParameters()
    model = SimpleHANKNet(n_income_shocks=params.n_e).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    print(f"Training Deep HANK for {n_epochs} epochs...")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        loss, metrics = compute_loss(model, params)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        if epoch % print_every == 0:
            print(f"Epoch {epoch}/{n_epochs}")
            print(f"  Loss: {metrics['total']:.6f}")
            print(f"  Budget: {metrics['budget']:.6f}")
            print(f"  Constraint: {metrics['constraint']:.6f}")
            print(f"  Euler: {metrics['euler']:.6f}")
            print()
    
    return model, params

def simulate_irf(model: SimpleHANKNet, params: HANKParameters, 
                 shock_size: float = 0.01, T: int = 20):
    """
    Simulate impulse response to interest rate shock
    """
    model.eval()
    
    # Steady state (approximate - start from middle)
    liquid_ss = torch.tensor([[10.0]], device=device)
    illiquid_ss = torch.tensor([[25.0]], device=device)
    income_ss = torch.tensor([3], dtype=torch.long, device=device)  # median income
    
    # Get steady state consumption
    with torch.no_grad():
        policies_ss = model(liquid_ss, illiquid_ss, income_ss)
        c_ss = policies_ss['consumption'].item()
    
    # Simulate shock path
    r_shock = params.r_liquid + shock_size
    
    c_path = []
    liquid_path = [liquid_ss.item()]
    
    for t in range(T):
        current_liquid = torch.tensor([[liquid_path[-1]]], device=device)
        
        with torch.no_grad():
            policies = model(current_liquid, illiquid_ss, income_ss)
            c_t = policies['consumption'].item()
            a_l_t = policies['a_liquid'].item()
        
        c_path.append(c_t)
        
        # Update liquid assets (simplified law of motion)
        y_t = params.w * params.y_grid[income_ss].item()
        new_liquid = (1 + r_shock) * liquid_path[-1] + y_t - c_t
        liquid_path.append(new_liquid)
    
    # Convert to % deviations
    c_irf = [(c / c_ss - 1) * 100 for c in c_path]
    
    return c_irf

if __name__ == "__main__":
    # Train model
    model, params = train_deep_hank(n_epochs=5000)
    
    # Simulate IRF
    print("\\nSimulating IRF to 1% interest rate shock...")
    irf = simulate_irf(model, params, shock_size=0.01, T=20)
    
    print("Consumption IRF (% deviation from SS):")
    for t, c_dev in enumerate(irf):
        print(f"  Quarter {t}: {c_dev:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'deep_hank_minimal.pt')
    print("\\nModel saved to deep_hank_minimal.pt")
