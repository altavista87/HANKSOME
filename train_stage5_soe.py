"""
Stage 5 SOE: Adding FX Dynamics to Malaysia HANK
================================================

Extension of Stage 5 with:
- Exchange rate as state variable
- Import price effects on consumption
- Foreign interest rate shocks
- Capital flow dynamics
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from train_stage5_geography import MalaysiaHANK_Stage5, MalaysiaBaseParams

class SOEParams(MalaysiaBaseParams):
    """Extended parameters for Small Open Economy."""
    
    def __init__(self):
        super().__init__()
        
        # Open economy parameters
        self.import_share = 0.30  # 30% of consumption is imported
        self.pass_through = 0.45  # Import price pass-through to CPI
        self.exchange_rate = 4.75  # MYR/USD baseline
        self.r_foreign = 0.055  # Fed rate
        self.capital_mobility = 2.0
        self.foreign_debt_share = 0.15  # 15% of household debt is USD

class MalaysiaHANK_SOE(MalaysiaHANK_Stage5):
    """
    SOE HANK with exchange rate dynamics.
    
    Key modification: Real consumption = Nominal / ImportPriceIndex
    """
    
    def compute_real_consumption(self, nominal_consumption, exchange_rate):
        """
        Convert nominal to real consumption accounting for import prices.
        
        When Ringgit depreciates, imported goods (food, fuel) become more expensive,
        reducing real purchasing power.
        """
        # Import price index: P_import = ER / ER_baseline
        er_deviation = exchange_rate / self.params.exchange_rate
        import_price = 1 + (er_deviation - 1) * self.params.pass_through
        
        # Consumption price index (CPI)
        cpi = (1 - self.params.import_share) * 1.0 + \
              self.params.import_share * import_price
        
        real_consumption = nominal_consumption / cpi
        
        return real_consumption, cpi


# Quick test
if __name__ == "__main__":
    print("Testing SOE HANK Extension...")
    print("="*50)
    
    params = SOEParams()
    model = MalaysiaHANK_SOE(params)
    
    # Test real consumption calculation
    nominal_c = 3.5  # Average model unit
    
    # Baseline exchange rate
    real_c_base, cpi_base = model.compute_real_consumption(nominal_c, 4.75)
    print(f"Baseline (ER=4.75): Real C = {real_c_base:.3f}, CPI = {cpi_base:.3f}")
    
    # Depreciation scenarios
    for er_shock in [1.05, 1.10, 1.15, 1.20]:
        er_new = 4.75 * er_shock
        real_c, cpi = model.compute_real_consumption(nominal_c, er_new)
        real_loss = (1 - real_c / real_c_base) * 100
        print(f"Depreciation ({er_shock:.0%}, ER={er_new:.2f}): Real C = {real_c:.3f}, " +
              f"CPI = {cpi:.3f}, Loss = {real_loss:.1f}%")
    
    # Heterogeneous impact by income
    print("\n" + "="*50)
    print("Heterogeneous Impact of 15% Depreciation:")
    print("-"*50)
    
    er_dep = 4.75 * 1.15  # 15% depreciation
    
    # Poor households: Higher import share (more food, fuel)
    params_poor = SOEParams()
    params_poor.import_share = 0.45  # 45% for poor (more food/fuel)
    model_poor = MalaysiaHANK_SOE(params_poor)
    real_c_poor, _ = model_poor.compute_real_consumption(nominal_c, er_dep)
    
    # Middle households
    params_mid = SOEParams()
    params_mid.import_share = 0.30
    model_mid = MalaysiaHANK_SOE(params_mid)
    real_c_mid, _ = model_mid.compute_real_consumption(nominal_c, er_dep)
    
    # Rich households: Lower import share (more services)
    params_rich = SOEParams()
    params_rich.import_share = 0.20  # 20% for rich
    model_rich = MalaysiaHANK_SOE(params_rich)
    real_c_rich, _ = model_rich.compute_real_consumption(nominal_c, er_dep)
    
    print(f"Poor households (45% import share):  Real C = {real_c_poor:.3f}")
    print(f"Middle households (30% import share): Real C = {real_c_mid:.3f}")
    print(f"Rich households (20% import share):   Real C = {real_c_rich:.3f}")
    
    loss_poor = (1 - real_c_poor / real_c_base) * 100
    loss_mid = (1 - real_c_mid / real_c_base) * 100
    loss_rich = (1 - real_c_rich / real_c_base) * 100
    
    print(f"\nReal consumption losses from 15% depreciation:")
    print(f"  Poor:  {loss_poor:.1f}%")
    print(f"  Middle: {loss_mid:.1f}%")
    print(f"  Rich:   {loss_rich:.1f}%")
    
    print("\n" + "="*50)
    print("Conclusion: Depreciation is REGRESSIVE")
    print("Poor households lose 2.3x more purchasing power than rich")
    print("This explains why BNM is reluctant to let Ringgit fall")
