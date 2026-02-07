import numpy as np

class MalaysiaParams:
    def __init__(self):
        # --- 1. Labor Market Segmentation (The "Malaysia Data Composition Plan") ---
        # Weights based on 2023 Labor Force statistics
        self.weight_formal   = 0.50  # Private sector employees (EPF covered)
        self.weight_public   = 0.10  # Civil servants (KWAP/Pension covered)
        self.weight_informal = 0.40  # Gig workers, self-employed, petty traders

        # Risk Profiles (Std Dev of idiosyncratic shocks)
        self.sigma_formal   = 0.40   # Baseline private sector risk
        self.sigma_public   = 0.15   # Low risk (stable government income)
        self.sigma_informal = 0.75   # Very High risk (volatile gig/cash income)

        # Calculate Aggregate Income Risk (Weighted Variance Approach)
        # sigma_total = sqrt( sum(w_i * sigma_i^2) )
        var_sum = (self.weight_formal * self.sigma_formal**2 + 
                   self.weight_public * self.sigma_public**2 + 
                   self.weight_informal * self.sigma_informal**2)
        
        # --- 2. Preferences & Aggregates ---
        # ADMISSIBLE CALIBRATION (Rank 4 - Research Baseline)
        # Prioritizes interest rate (r=2.35%) and Gini (0.759) over Top 1% tail.
        # Worker Mass: 97%, Elite Mass: 3%
        self.beta = 0.940      # Worker Beta
        self.beta_high = 0.948 # Elite Beta
        self.omega = 0.97      # Worker Mass
        self.premium = 0.03    # Return Premium
        
        self.sigma = 1.0      # CRRA coefficient (Log utility)
        self.frisch = 0.5     # Frisch elasticity of labor supply
        self.v = 1.0          # Disutility of labor weight

        # --- 3. Income Process ---
        self.rho_e = 0.91     
        self.sigma_e = 0.40    # Moderate Risk Profile
        self.n_e = 7

        # --- 4. Institutional Constraints (Malaysian Edge) ---
        self.epf_contribution_rate = 0.11 # Mandatory forced savings
        self.pension_wealth_equity = 0.20 # Imputed wealth for civil servants

        # --- 5. Production ---
        self.alpha = 0.33     # Capital share
        self.delta = 0.05     # Depreciation rate

        # --- 6. Government / Policy ---
        self.r_target = 0.03  # Target Real Interest Rate (3%)
        self.tax_rate = 0.0   # Flat tax rate (simplified)
        self.transfer = 0.0   # Lump sum transfers

        # --- 7. Asset Grid ---
        self.n_a = 100        # Number of asset grid points
        self.a_min = 0.0      # Borrowing constraint (0 = no borrowing)
        self.a_max = 500.0    # Max assets (Increased for Top 1% tail)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Parameter {key} not found.")

def get_calibration():
    return MalaysiaParams()
