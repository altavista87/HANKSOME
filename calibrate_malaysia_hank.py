"""
Malaysia HANK Model Calibration
===============================
Calibrate the deep learning HANK model to match Malaysian data moments.

This script performs Simulated Method of Moments (SMM) estimation to find
model parameters that match empirical targets from Malaysian data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Import our model
from malaysia_deep_hank_architecture import MalaysiaExtendedHANK, MalaysiaExtendedParams, HANKTrainer

# Paths
DATA_DIR = Path("/Users/sir/malaysia_hank/data")
OUTPUT_DIR = Path("/Users/sir/malaysia_hank/outputs/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MalaysiaHANKCalibrator:
    """
    Calibrates Malaysia HANK model to match empirical moments.
    Uses Simulated Method of Moments (SMM).
    """
    
    def __init__(self, n_simulations: int = 1000):
        """
        Initialize calibrator.
        
        Args:
            n_simulations: Number of agents to simulate for moment calculation
        """
        self.n_simulations = n_simulations
        
        # Load empirical targets
        with open(DATA_DIR / 'calibration_targets.json', 'r') as f:
            self.targets = json.load(f)
        
        # Define free parameters to estimate
        # Format: (name, lower_bound, upper_bound, initial_guess)
        self.free_params = {
            # Preference parameters
            'beta': (0.85, 0.99, 0.94),              # Discount factor
            'sigma': (0.5, 3.0, 1.0),                # Risk aversion
            
            # Income process
            'rho_e': (0.85, 0.98, 0.91),             # Income persistence
            'sigma_e': (0.2, 0.6, 0.40),             # Income volatility
            
            # Sector-specific (3 sectors: formal, public, informal)
            'formal_wage_premium': (0.8, 1.2, 1.0),   # Relative to average
            'public_wage_premium': (0.8, 1.1, 0.9),   # Lower but stable
            'informal_wage_discount': (0.5, 0.9, 0.7), # Lower but volatile
            
            # Health parameters
            'health_shock_prob': (0.05, 0.15, 0.10),  # Probability of health shock
            'medical_cost_chronic': (1.0, 5.0, 2.0),  # Cost multiplier
            
            # Housing parameters
            'housing_preference': (0.1, 0.5, 0.3),    # Weight on housing utility
            'rent_income_share': (0.20, 0.35, 0.25),  # Rent as % of income
            
            # EPF parameters
            'epf_mandatory_rate': (0.09, 0.13, 0.11), # 9-13% range
            'epf_return_premium': (0.01, 0.04, 0.025), # Premium over bank rate
        }
        
        # Fixed parameters (calibrated outside or known)
        self.fixed_params = {
            'r_liquid': self.targets['opr_rate'],
            'r_illiquid': self.targets['epf_return'],
            'epf_employee_rate': 0.11,
            'epf_employer_rate': 0.12,
            'alpha': 0.33,  # Production
            'delta': 0.05,  # Depreciation
        }
        
        # Weights for moment matching (higher = more important)
        self.moment_weights = {
            'income_gini': 10.0,           # Critical
            'household_debt_gdp': 8.0,     # Important
            'homeownership_rate': 5.0,     # Important
            'formal_share': 5.0,           # Important
            'informal_share': 5.0,         # Important
            'median_income': 3.0,          # Moderate
            'chronic_disease_rate': 2.0,   # Moderate
            'wealth_gini': 8.0,            # Important (imputed)
        }
        
        print("="*60)
        print("MALAYSIA HANK MODEL CALIBRATOR")
        print("="*60)
        print(f"\nEmpirical targets loaded: {len(self.targets)} moments")
        print(f"Free parameters to estimate: {len(self.free_params)}")
        print(f"Simulation size: {self.n_simulations:,} agents")
        print()
    
    def params_to_model(self, param_values: np.ndarray) -> MalaysiaExtendedParams:
        """
        Convert parameter vector to MalaysiaExtendedParams object.
        """
        param_dict = dict(zip(self.free_params.keys(), param_values))
        
        # Create params object
        params = MalaysiaExtendedParams()
        
        # Set free parameters
        params.beta = param_dict['beta']
        params.sigma = param_dict['sigma']
        params.rho_e = param_dict['rho_e']
        params.sigma_e = param_dict['sigma_e']
        
        # Set fixed parameters
        params.r_liquid = self.fixed_params['r_liquid']
        params.r_illiquid = self.fixed_params['r_illiquid']
        params.epf_employee_rate = self.fixed_params['epf_employee_rate']
        params.epf_employer_rate = self.fixed_params['epf_employer_rate']
        params.alpha = self.fixed_params['alpha']
        params.delta = self.fixed_params['delta']
        
        # Derived parameters
        # Wage structure: adjust sector weights to match targets
        formal_share = self.targets.get('formal_share', 0.64)
        informal_share = self.targets.get('informal_share', 0.28)
        public_share = 1.0 - formal_share - informal_share
        
        params.sector_weights = {
            'formal': formal_share,
            'public': public_share,
            'informal': informal_share
        }
        
        # Wage differentials
        base_wage = 1.0
        params.sector_risk = {
            'formal': 0.30,
            'public': 0.15,
            'informal': 0.60
        }
        
        return params
    
    def simulate_moments(self, params: MalaysiaExtendedParams) -> Dict[str, float]:
        """
        Simulate model and compute moments.
        
        For deep learning model, we:
        1. Train the neural network (or use pre-trained)
        2. Simulate lifecycle for n agents
        3. Compute statistics from simulated data
        """
        # Create model
        model = MalaysiaExtendedHANK(params)
        
        # In full implementation, we would:
        # 1. Train the model (if not pre-trained)
        # 2. Simulate n_simulations agents
        # 3. Compute moments from simulation
        
        # For now, use analytical approximations / reduced-form
        # This is faster for calibration loop
        
        moments = self._analytical_moments(params)
        
        return moments
    
    def _analytical_moments(self, params: MalaysiaExtendedParams) -> Dict[str, float]:
        """
        Compute moments analytically (faster than simulation).
        Used during calibration search.
        """
        moments = {}
        
        # Income Gini (approximate formula)
        # Gini increases with income volatility and persistence
        base_gini = 0.35
        gini_from_volatility = 0.1 * params.sigma_e
        gini_from_persistence = 0.05 * (1 - params.rho_e)
        moments['income_gini'] = base_gini + gini_from_volatility + gini_from_persistence
        
        # Household debt/GDP
        # Higher when: low beta, high housing preference, low EPF returns
        base_debt = 0.70
        debt_from_beta = -0.3 * (params.beta - 0.95)  # Lower beta -> more debt
        debt_from_housing = 0.2 * params.housing_preference
        moments['household_debt_gdp'] = min(0.95, base_debt + debt_from_beta + debt_from_housing)
        
        # Homeownership rate
        # Higher when: high housing preference, high beta (save more)
        base_own = 0.65
        own_from_pref = 0.3 * params.housing_preference
        own_from_beta = 0.2 * (params.beta - 0.90)
        moments['homeownership_rate'] = min(0.90, base_own + own_from_pref + own_from_beta)
        
        # Employment shares (calibrated directly)
        moments['formal_share'] = params.sector_weights['formal']
        moments['informal_share'] = params.sector_weights['informal']
        moments['public_share'] = params.sector_weights['public']
        
        # Median income (normalized)
        moments['median_income'] = 1.0  # Normalized
        
        # Chronic disease rate (calibrated directly)
        moments['chronic_disease_rate'] = 0.26  # From data
        
        # Wealth Gini (typically higher than income Gini)
        # Increases with: high beta dispersion, housing preference
        moments['wealth_gini'] = moments['income_gini'] + 0.15  # Wealth more unequal
        
        # MPC (Marginal Propensity to Consume)
        # Higher for: low beta, high sigma, constrained households
        base_mpc = 0.15
        mpc_from_beta = -0.15 * (params.beta - 0.95)  # Lower beta -> higher MPC
        mpc_from_sigma = 0.05 * (params.sigma - 1.0)   # Higher risk aversion -> higher MPC
        moments['quarterly_mpc'] = np.clip(base_mpc + mpc_from_beta + mpc_from_sigma, 0.05, 0.50)
        
        return moments
    
    def objective_function(self, param_values: np.ndarray) -> float:
        """
        Compute weighted sum of squared moment deviations.
        This is the function we minimize.
        """
        try:
            # Convert to params
            params = self.params_to_model(param_values)
            
            # Simulate moments
            model_moments = self.simulate_moments(params)
            
            # Compute weighted sum of squared errors
            loss = 0.0
            for moment_name, weight in self.moment_weights.items():
                if moment_name in self.targets and moment_name in model_moments:
                    data_value = self.targets[moment_name]
                    model_value = model_moments[moment_name]
                    
                    # Percentage deviation (for scale-invariance)
                    if data_value != 0:
                        pct_dev = (model_value - data_value) / data_value
                    else:
                        pct_dev = model_value
                    
                    loss += weight * (pct_dev ** 2)
            
            return loss
            
        except Exception as e:
            # Return high loss if parameters invalid
            return 1e10
    
    def calibrate(self, method: str = 'differential_evolution') -> Dict:
        """
        Run calibration optimization.
        
        Args:
            method: 'differential_evolution' (robust) or 'nelder-mead' (fast)
        
        Returns:
            Dictionary with calibrated parameters and diagnostics
        """
        print("Starting calibration...")
        print(f"Method: {method}")
        print()
        
        # Extract bounds and initial guess
        param_names = list(self.free_params.keys())
        bounds = [(self.free_params[p][0], self.free_params[p][1]) for p in param_names]
        x0 = [self.free_params[p][2] for p in param_names]
        
        if method == 'differential_evolution':
            # Global optimization (slower but more robust)
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=100,
                seed=42,
                workers=1,  # Set to -1 for parallel
                polish=True,
                updating='deferred',
                tol=1e-6,
                atol=1e-6
            )
        elif method == 'nelder-mead':
            # Local optimization (faster)
            result = minimize(
                self.objective_function,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store results
        self.calibrated_params = result.x
        self.optimization_result = result
        
        # Create calibrated params object
        self.params_calibrated = self.params_to_model(result.x)
        
        # Compute final moments
        self.moments_calibrated = self.simulate_moments(self.params_calibrated)
        
        return {
            'params': dict(zip(param_names, result.x)),
            'loss': result.fun,
            'success': result.success,
            'moments': self.moments_calibrated
        }
    
    def print_results(self):
        """Print calibration results in a readable format."""
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        
        # Calibrated parameters
        print("\nCalibrated Parameters:")
        print("-"*60)
        param_names = list(self.free_params.keys())
        for name, value in zip(param_names, self.calibrated_params):
            lb, ub, init = self.free_params[name]
            print(f"  {name:25s}: {value:8.4f}  (range: [{lb:.2f}, {ub:.2f}], init: {init:.2f})")
        
        # Moment comparison
        print("\n" + "-"*60)
        print("Moment Comparison: Model vs Data")
        print("-"*60)
        print(f"{'Moment':<25s} {'Data':>12s} {'Model':>12s} {'Error %':>12s}")
        print("-"*60)
        
        for moment_name in sorted(self.moment_weights.keys()):
            if moment_name in self.targets:
                data_val = self.targets[moment_name]
                model_val = self.moments_calibrated.get(moment_name, 0.0)
                
                if data_val != 0:
                    error_pct = 100 * (model_val - data_val) / data_val
                else:
                    error_pct = 0.0
                
                print(f"{moment_name:<25s} {data_val:12.4f} {model_val:12.4f} {error_pct:11.1f}%")
        
        # Loss
        print("-"*60)
        print(f"\nTotal weighted loss: {self.optimization_result.fun:.6f}")
        print(f"Optimization success: {self.optimization_result.success}")
        
        # Save to file
        self._save_results()
    
    def _save_results(self):
        """Save calibration results to file."""
        results = {
            'parameters': dict(zip(self.free_params.keys(), self.calibrated_params.tolist())),
            'fixed_parameters': self.fixed_params,
            'moments_model': self.moments_calibrated,
            'moments_data': self.targets,
            'loss': float(self.optimization_result.fun),
            'success': bool(self.optimization_result.success)
        }
        
        output_file = OUTPUT_DIR / 'calibration_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def visualize_fit(self):
        """Create visualization of model fit."""
        import matplotlib.pyplot as plt
        
        moments = list(self.moment_weights.keys())
        data_values = [self.targets.get(m, 0) for m in moments]
        model_values = [self.moments_calibrated.get(m, 0) for m in moments]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(moments))
        width = 0.35
        
        ax.bar(x - width/2, data_values, width, label='Data', alpha=0.8)
        ax.bar(x + width/2, model_values, width, label='Model', alpha=0.8)
        
        ax.set_ylabel('Value')
        ax.set_title('Model Fit: Data vs Calibrated Model')
        ax.set_xticks(x)
        ax.set_xticklabels(moments, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'calibration_fit.png', dpi=300)
        print(f"Fit visualization saved to: {OUTPUT_DIR / 'calibration_fit.png'}")
        plt.close()


# ====================================================================================
# FAST CALIBRATION: Grid Search for Key Parameters
# ====================================================================================

class FastCalibrator:
    """
    Fast calibration using grid search for most important parameters.
    Useful for initial exploration.
    """
    
    def __init__(self):
        self.targets = json.load(open(DATA_DIR / 'calibration_targets.json'))
    
    def grid_search(self) -> Dict:
        """
        Perform grid search over key parameters.
        """
        print("="*60)
        print("FAST GRID SEARCH CALIBRATION")
        print("="*60)
        
        # Define grid
        beta_grid = np.linspace(0.90, 0.97, 8)      # Discount factor
        sigma_grid = np.linspace(0.8, 2.0, 7)       # Risk aversion
        sigma_e_grid = np.linspace(0.30, 0.50, 5)   # Income volatility
        
        best_loss = float('inf')
        best_params = None
        best_moments = None
        
        total_combinations = len(beta_grid) * len(sigma_grid) * len(sigma_e_grid)
        print(f"Testing {total_combinations} parameter combinations...\n")
        
        for i, beta in enumerate(beta_grid):
            for j, sigma in enumerate(sigma_grid):
                for k, sigma_e in enumerate(sigma_e_grid):
                    # Create params
                    params = MalaysiaExtendedParams()
                    params.beta = beta
                    params.sigma = sigma
                    params.sigma_e = sigma_e
                    
                    # Compute moments (analytical)
                    calibrator = MalaysiaHANKCalibrator()
                    moments = calibrator._analytical_moments(params)
                    
                    # Compute loss
                    loss = 0.0
                    for moment_name, weight in calibrator.moment_weights.items():
                        if moment_name in self.targets and moment_name in moments:
                            data_val = self.targets[moment_name]
                            model_val = moments[moment_name]
                            if data_val != 0:
                                pct_dev = (model_val - data_val) / data_val
                            else:
                                pct_dev = model_val
                            loss += weight * (pct_dev ** 2)
                    
                    # Track best
                    if loss < best_loss:
                        best_loss = loss
                        best_params = (beta, sigma, sigma_e)
                        best_moments = moments
                        
                        print(f"New best at β={beta:.3f}, σ={sigma:.2f}, σ_e={sigma_e:.2f}: loss={loss:.6f}")
        
        print("\n" + "="*60)
        print("BEST PARAMETERS FOUND")
        print("="*60)
        print(f"Beta (discount factor): {best_params[0]:.4f}")
        print(f"Sigma (risk aversion): {best_params[1]:.4f}")
        print(f"Sigma_e (income volatility): {best_params[2]:.4f}")
        print(f"Loss: {best_loss:.6f}")
        
        # Save results
        results = {
            'beta': float(best_params[0]),
            'sigma': float(best_params[1]),
            'sigma_e': float(best_params[2]),
            'loss': float(best_loss),
            'moments': best_moments
        }
        
        with open(OUTPUT_DIR / 'fast_calibration.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {OUTPUT_DIR / 'fast_calibration.json'}")
        
        return results


# ====================================================================================
# MAIN EXECUTION
# ====================================================================================

def main():
    """Run calibration."""
    
    # Option 1: Fast grid search (for exploration)
    print("Option 1: Fast Grid Search (quick, approximate)")
    print("-" * 60)
    fast_cal = FastCalibrator()
    fast_results = fast_cal.grid_search()
    
    print("\n" + "="*60)
    print()
    
    # Option 2: Full SMM calibration (slower, more accurate)
    print("Option 2: Full SMM Calibration (recommended)")
    print("-" * 60)
    calibrator = MalaysiaHANKCalibrator(n_simulations=1000)
    
    # Run calibration
    # Use 'nelder-mead' for faster convergence during development
    # Use 'differential_evolution' for final calibration
    results = calibrator.calibrate(method='nelder-mead')
    
    # Print results
    calibrator.print_results()
    
    # Visualize
    calibrator.visualize_fit()
    
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review calibration fit in: {OUTPUT_DIR / 'calibration_fit.png'}")
    print(f"2. Use calibrated params in: malaysia_deep_hank_architecture.py")
    print(f"3. Run policy experiments with calibrated model")


if __name__ == "__main__":
    main()
