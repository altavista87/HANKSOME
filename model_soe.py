"""
SOE HANK Model - Milestone 2
============================

Extends MalaysiaHANK_Stage5 with:
- Exchange rate as state variable
- Real consumption accounting for import prices
- Heterogeneous FX exposure by income
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from pathlib import Path

from train_stage5_geography import MalaysiaHANK_Stage5
from soe_params import SOEParams


class MalaysiaHANK_SOE(MalaysiaHANK_Stage5):
    """
    Small Open Economy HANK with FX dynamics.
    
    Key extensions:
    1. Real consumption = Nominal / CPI(exchange_rate)
    2. Heterogeneous import exposure by income quintile
    3. USD debt servicing (for rich households)
    
    CONSERVATIVE scenario: Q1 has LOWER import exposure.
    """
    
    def __init__(self, params: SOEParams):
        """
        Initialize SOE HANK model.
        
        Args:
            params: SOEParams with FX parameters
        """
        super().__init__(params)
        self.params = params  # This is now SOEParams
        
        # Validate we have SOE params
        assert hasattr(params, 'exchange_rate_baseline'), \
            "params must be SOEParams, not MalaysiaBaseParams"
        
        print(f"✓ SOE HANK initialized (scenario: CONSERVATIVE)")
        print(f"  Baseline ER: {params.exchange_rate_baseline:.2f} MYR/USD")
        print(f"  Pass-through: {params.pass_through_elasticity:.0%}")
    
    def compute_import_price_index(self, exchange_rate: float) -> float:
        """
        Compute import price index from exchange rate.
        
        P_import = (ER / ER_baseline) ^ pass_through
        
        Args:
            exchange_rate: Current MYR/USD
        
        Returns:
            Import price index (1.0 = baseline)
        """
        return self.params.compute_import_price_index(exchange_rate)
    
    def compute_cpi(self, exchange_rate: float) -> float:
        """
        Compute consumer price index.
        
        Args:
            exchange_rate: Current MYR/USD
        
        Returns:
            CPI (1.0 = baseline)
        """
        return self.params.compute_cpi(exchange_rate)
    
    def compute_real_consumption(
        self,
        nominal_consumption: torch.Tensor,
        exchange_rate: float,
        income_quintile: str = None
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Convert nominal to real consumption accounting for import prices.
        
        CONSERVATIVE scenario: Q1 has LOWER import exposure, so less affected
        by depreciation.
        
        Args:
            nominal_consumption: Nominal consumption (model units)
            exchange_rate: Current MYR/USD
            income_quintile: 'Q1', 'Q2', 'Q3', 'Q4', 'Q5' or None
        
        Returns:
            real_consumption: Real purchasing power
            cpi: Consumer price index
            import_share: Household-specific import share
        """
        # Get household-specific import share
        if income_quintile:
            import_share = self.params.get_import_share(income_quintile)
        else:
            import_share = self.params.import_share_aggregate
        
        # Import price index
        import_price = self.compute_import_price_index(exchange_rate)
        
        # Household-specific CPI
        # CPI_hh = (1 - α_hh) * P_domestic + α_hh * P_import
        # P_domestic = 1.0 (normalized)
        cpi = (1 - import_share) * 1.0 + import_share * import_price
        
        # Real consumption
        real_consumption = nominal_consumption / cpi
        
        return real_consumption, cpi, import_share
    
    def compute_usd_debt_service(
        self,
        mortgage_debt: torch.Tensor,
        exchange_rate: float,
        income_quintile: str = None
    ) -> torch.Tensor:
        """
        Compute USD debt service cost.
        
        When Ringgit depreciates, USD debt becomes more expensive to service.
        CONSERVATIVE scenario: Rich households (Q5) have more USD debt.
        
        Args:
            mortgage_debt: Total mortgage debt
            exchange_rate: Current MYR/USD
            income_quintile: For heterogeneous USD debt share
        
        Returns:
            Additional debt service cost in MYR
        """
        if income_quintile:
            usd_share = self.params.get_usd_debt_share(income_quintile)
        else:
            # Use average
            usd_share = 0.15
        
        if usd_share == 0:
            return torch.zeros_like(mortgage_debt)
        
        # USD debt amount
        usd_debt = mortgage_debt * usd_share
        
        # Exchange rate deviation
        er_deviation = exchange_rate / self.params.exchange_rate_baseline
        
        # Additional cost from depreciation
        # Base service cost: usd_debt * r_mortgage
        # With depreciation: usd_debt * r_mortgage * er_deviation
        base_service = usd_debt * self.params.r_mortgage
        actual_service = base_service * er_deviation
        
        additional_cost = actual_service - base_service
        
        return additional_cost
    
    def forward_soe(
        self,
        state: Dict[str, torch.Tensor],
        exchange_rate: float,
        r_foreign: float,
        track_quintiles: bool = False
    ) -> Dict:
        """
        Forward pass with FX state.
        
        Args:
            state: Household state (liquid, illiquid, etc.)
            exchange_rate: Current MYR/USD
            r_foreign: Foreign interest rate (Fed)
            track_quintiles: If True, compute real consumption by quintile
        
        Returns:
            policies: Extended with real_consumption, cpi, import_share
        """
        # Get base policies from parent model
        policies = self(
            state['liquid'],
            state['illiquid'],
            state['mortgage_debt'],
            state['base_income'],
            state['education_level'],
            state['age'],
            state['health_state'],
            state['housing_type'],
            state['location']
        )
        
        nominal_consumption = policies['consumption']
        
        # Compute real consumption with aggregate import share
        real_consumption, cpi, import_share = self.compute_real_consumption(
            nominal_consumption,
            exchange_rate
        )
        
        # Store in policies
        policies['real_consumption'] = real_consumption
        policies['nominal_consumption'] = nominal_consumption
        policies['cpi'] = cpi
        policies['import_share'] = import_share
        
        # USD debt service (aggregate)
        usd_service = self.compute_usd_debt_service(
            state['mortgage_debt'],
            exchange_rate
        )
        policies['usd_debt_service'] = usd_service
        
        # Track by quintile if requested
        if track_quintiles:
            policies['quintile_real_consumption'] = {}
            
            # Compute income quintiles
            income = self.compute_income(
                state['base_income'],
                state['education_level'],
                state['health_state'],
                state['location']
            )
            
            # Define quintile boundaries
            q_thresholds = torch.quantile(
                income,
                torch.tensor([0.2, 0.4, 0.6, 0.8], device=income.device)
            )
            
            quintiles = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            masks = [
                income < q_thresholds[0],
                (income >= q_thresholds[0]) & (income < q_thresholds[1]),
                (income >= q_thresholds[1]) & (income < q_thresholds[2]),
                (income >= q_thresholds[2]) & (income < q_thresholds[3]),
                income >= q_thresholds[3]
            ]
            
            for quintile, mask in zip(quintiles, masks):
                if mask.sum() > 0:
                    q_nominal = nominal_consumption[mask].mean()
                    q_real, q_cpi, q_alpha = self.compute_real_consumption(
                        q_nominal, exchange_rate, quintile
                    )
                    policies['quintile_real_consumption'][quintile] = {
                        'nominal': q_nominal.item(),
                        'real': q_real.item(),
                        'cpi': q_cpi,
                        'import_share': q_alpha
                    }
        
        # Store FX state
        policies['exchange_rate'] = exchange_rate
        policies['r_foreign'] = r_foreign
        
        return policies
    
    def simulate_fx_shock_impact(
        self,
        state: Dict[str, torch.Tensor],
        er_shock: float,
        T: int = 10
    ) -> Dict:
        """
        Quick simulation of FX shock impact on different quintiles.
        
        Args:
            state: Household state
            er_shock: Exchange rate shock (e.g., 1.10 for 10% depreciation)
            T: Number of periods
        
        Returns:
            impact: Consumption impact by quintile
        """
        baseline_er = self.params.exchange_rate_baseline
        shocked_er = baseline_er * er_shock
        
        # Baseline
        baseline_policies = self.forward_soe(
            state, baseline_er, self.params.r_foreign_baseline, track_quintiles=True
        )
        
        # Shocked
        shocked_policies = self.forward_soe(
            state, shocked_er, self.params.r_foreign_baseline, track_quintiles=True
        )
        
        # Compute impacts
        impact = {}
        
        # Aggregate impact
        base_real = baseline_policies['real_consumption'].mean().item()
        shock_real = shocked_policies['real_consumption'].mean().item()
        impact['aggregate'] = (shock_real / base_real - 1) * 100
        
        # By quintile
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            if q in baseline_policies['quintile_real_consumption']:
                base_q = baseline_policies['quintile_real_consumption'][q]['real']
                shock_q = shocked_policies['quintile_real_consumption'][q]['real']
                impact[q] = (shock_q / base_q - 1) * 100
        
        return impact


def test_model_soe():
    """Test SOE HANK model."""
    print("\n" + "="*60)
    print("TESTING SOE HANK MODEL - MILESTONE 2")
    print("="*60)
    
    # Initialize
    params = SOEParams()
    model = MalaysiaHANK_SOE(params)
    
    # Test 1: Import price mechanics
    print("\n1. Testing import price mechanics:")
    for er_shock in [0.95, 1.00, 1.10, 1.20]:
        er = params.exchange_rate_baseline * er_shock
        imp_price = model.compute_import_price_index(er)
        cpi = model.compute_cpi(er)
        print(f"   ER {er_shock:.0%}: Import={imp_price:.3f}, CPI={cpi:.3f}")
    
    # Test 2: Real consumption by quintile
    print("\n2. Testing heterogeneous real consumption:")
    nominal_c = torch.tensor([3.5])
    
    for er_shock in [1.00, 1.15]:  # Baseline vs 15% depreciation
        er = params.exchange_rate_baseline * er_shock
        print(f"\n   Exchange rate {er_shock:.0%} (ER={er:.2f}):")
        
        for q in ['Q1', 'Q3', 'Q5']:
            real_c, cpi, alpha = model.compute_real_consumption(nominal_c, er, q)
            print(f"     {q} (α={alpha:.0%}): Real C={real_c.item():.3f}, CPI={cpi:.3f}")
    
    # Test 3: USD debt service
    print("\n3. Testing USD debt service (15% depreciation):")
    mortgage = torch.tensor([50.0])  # Model units
    er_dep = params.exchange_rate_baseline * 1.15
    
    for q in ['Q1', 'Q3', 'Q5']:
        service = model.compute_usd_debt_service(mortgage, er_dep, q)
        usd_share = params.get_usd_debt_share(q)
        print(f"   {q} (USD debt={usd_share:.0%}): " +
              f"Additional service={service.item():.2f}")
    
    # Test 4: Quick shock simulation
    print("\n4. Testing FX shock impact simulation:")
    
    # Create dummy state
    n = 1000
    device = torch.device('cpu')
    state = {
        'liquid': torch.rand(n, 1) * 50,
        'illiquid': torch.rand(n, 1) * 200,
        'mortgage_debt': torch.rand(n, 1) * 100,
        'base_income': torch.exp(torch.randn(n, 1) * 0.5 + 0.5),
        'education_level': torch.randint(0, 3, (n,)),
        'age': torch.randint(18, 66, (n, 1)).float(),
        'health_state': torch.randint(0, 4, (n,)),
        'housing_type': torch.randint(0, 3, (n,)),
        'location': torch.randint(0, 26, (n,))
    }
    
    impact = model.simulate_fx_shock_impact(state, er_shock=1.15)
    
    print(f"   15% Ringgit depreciation impact on real consumption:")
    print(f"   Aggregate: {impact['aggregate']:+.2f}%")
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        if q in impact:
            print(f"   {q}: {impact[q]:+.2f}%")
    
    # Key finding
    q1_impact = impact.get('Q1', 0)
    q5_impact = impact.get('Q5', 0)
    if q1_impact and q5_impact:
        diff = q1_impact - q5_impact
        print(f"\n   Key finding: Q1 impact is {abs(diff):.2f}pp " +
              f"{'less' if diff > 0 else 'more'} severe than Q5")
        print(f"   (CONSERVATIVE scenario: Q1 less exposed to imports)")
    
    print("\n" + "="*60)
    print("✓ Milestone 2 Complete: SOE HANK Model Working!")
    print("="*60)
    
    return model


if __name__ == "__main__":
    test_model_soe()
