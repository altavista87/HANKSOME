"""
SOE HANK Parameters - Milestone 1 (COMPLETE)
=============================================

Extension of MalaysiaBaseParams with Open Economy features.
Uses CONSERVATIVE scenario: Q1 has LOWER import exposure.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

# Import base parameters
from train_stage5_geography import MalaysiaBaseParams

DATA_DIR = Path("/Users/sir/malaysia_hank/data")


@dataclass
class SOEParams(MalaysiaBaseParams):
    """
    Small Open Economy parameters for Malaysia HANK.
    
    CONSERVATIVE SCENARIO (Selected):
    - Q1 has LOWER import exposure (22%) - buys local goods
    - Q3 has HIGHEST import exposure (32%) - consumption upgrading
    - Q5 has moderate exposure (25%) - high services share
    
    Rationale: Poor households consume local rice (SST 5%), local brands,
    and informal services. Cannot afford imported goods.
    """
    
    # File paths for FX data
    fx_data_dir: Path = field(default=DATA_DIR, init=False)
    
    # Exchange rate parameters
    exchange_rate_baseline: float = 4.75  # MYR/USD (2024 average)
    exchange_rate_persistence: float = 0.85  # AR(1) coefficient
    exchange_rate_volatility: float = 0.05  # Monthly std dev
    
    # Import price parameters
    import_share_aggregate: float = 0.30  # 30% of consumption
    pass_through_elasticity: float = 0.45  # ER → CPI pass-through
    import_price_persistence: float = 0.80
    
    # Capital flow parameters
    r_foreign_baseline: float = 0.055  # Fed funds rate
    capital_mobility: float = 2.0  # UIP deviation elasticity
    capital_flow_volatility: float = 0.03  # % of GDP
    
    # BNM intervention
    fx_intervention_strength: float = 0.30  # 0=free float, 1=fixed
    foreign_reserves_months: float = 6.0  # Import coverage
    
    # CONSERVATIVE scenario: Q1 less exposed (see research notes)
    import_share_by_quintile: Dict[str, float] = field(default_factory=lambda: {
        'Q1': 0.22,  # Poor: local staples, informal sector
        'Q2': 0.28,  # Lower-middle: starting to buy processed
        'Q3': 0.32,  # Middle: HIGHEST (processed food, fuel-intensive)
        'Q4': 0.28,  # Upper-middle: more services
        'Q5': 0.25,  # Rich: high services, but luxury imports
    })
    
    # USD debt exposure (poor households have less foreign debt)
    usd_debt_share_by_quintile: Dict[str, float] = field(default_factory=lambda: {
        'Q1': 0.02,  # Poor: no access to USD credit
        'Q2': 0.05,
        'Q3': 0.10,
        'Q4': 0.18,
        'Q5': 0.30,  # Rich: can borrow in USD
    })
    
    # Commodity exports (Malaysia-specific)
    palm_oil_share_exports: float = 0.08  # 8% of exports
    lng_share_exports: float = 0.12  # 12% of exports
    electronics_share_exports: float = 0.40  # 40% of exports
    
    # Geography-specific exposure
    urban_import_share: float = 0.28
    rural_import_share: float = 0.32  # Slightly higher (fuel for transport)
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        super().__init__()
        self._load_fx_data_summary()
        self._validate_parameters()
    
    def _load_fx_data_summary(self):
        """Load and summarize FX data for calibration validation."""
        self.fx_data_summary = {}
        
        # Exchange rate data
        er_file = self.fx_data_dir / 'exchange_rate_monthly.csv'
        if er_file.exists():
            er_df = pd.read_csv(er_file)
            er_df['date'] = pd.to_datetime(er_df['date'])
            self.fx_data_summary['exchange_rate'] = {
                'mean': er_df['myr_usd'].mean(),
                'std': er_df['myr_usd'].std(),
                'min': er_df['myr_usd'].min(),
                'max': er_df['myr_usd'].max(),
                'latest': er_df['myr_usd'].iloc[-1],
                'volatility_annual': er_df['myr_usd'].pct_change().std() * np.sqrt(12),
            }
        
        # Import price data
        ip_file = self.fx_data_dir / 'import_price_index.csv'
        if ip_file.exists():
            ip_df = pd.read_csv(ip_file)
            ip_df['date'] = pd.to_datetime(ip_df['date'])
            self.fx_data_summary['import_prices'] = {
                'mean': ip_df['import_price_index'].mean(),
                'total_change': (ip_df['import_price_index'].iloc[-1] / 
                                ip_df['import_price_index'].iloc[0] - 1),
                'volatility': ip_df['import_price_index'].pct_change().std() * np.sqrt(12),
            }
        
        # Fed funds rate
        fed_file = self.fx_data_dir / 'fed_funds_rate.csv'
        if fed_file.exists():
            fed_df = pd.read_csv(fed_file)
            self.fx_data_summary['fed_rate'] = {
                'current': fed_df['fed_funds_rate'].iloc[-1],
                'max': fed_df['fed_funds_rate'].max(),
                'change_from_low': (fed_df['fed_funds_rate'].iloc[-1] - 
                                   fed_df['fed_funds_rate'].min()),
            }
    
    def _validate_parameters(self):
        """Validate parameter ranges."""
        assert 0 <= self.import_share_aggregate <= 1, "Import share must be in [0,1]"
        assert 0 <= self.pass_through_elasticity <= 1, "Pass-through must be in [0,1]"
        assert self.capital_mobility >= 0, "Capital mobility must be non-negative"
        assert 0 <= self.fx_intervention_strength <= 1, "Intervention strength must be in [0,1]"
    
    def get_import_share(self, income_quintile: str = None, location_type: str = None) -> float:
        """
        Get import share for specific household type.
        
        CONSERVATIVE scenario: Q1 has LOWER import exposure.
        """
        if income_quintile and income_quintile in self.import_share_by_quintile:
            base = self.import_share_by_quintile[income_quintile]
        else:
            base = self.import_share_aggregate
        
        # Location adjustment: rural slightly higher (transport fuel)
        if location_type == 'urban':
            return base * 0.95
        elif location_type == 'rural':
            return base * 1.10
        
        return base
    
    def get_usd_debt_share(self, income_quintile: str) -> float:
        """Get USD debt share for income quintile."""
        return self.usd_debt_share_by_quintile.get(income_quintile, 0.0)
    
    def compute_usd_debt_service(
        self,
        mortgage_debt: float,
        exchange_rate: float,
        income_quintile: str = None
    ) -> float:
        """
        Compute USD debt service cost.
        
        When Ringgit depreciates, USD debt becomes more expensive.
        
        Args:
            mortgage_debt: Total mortgage debt
            exchange_rate: Current MYR/USD
            income_quintile: For heterogeneous USD debt share
        
        Returns:
            Additional service cost in MYR due to depreciation
        """
        if income_quintile:
            usd_share = self.get_usd_debt_share(income_quintile)
        else:
            usd_share = 0.15  # Average
        
        if usd_share == 0 or mortgage_debt == 0:
            return 0.0
        
        # USD debt amount
        usd_debt = mortgage_debt * usd_share
        
        # Exchange rate deviation
        er_deviation = exchange_rate / self.exchange_rate_baseline
        
        # Base service cost
        base_service = usd_debt * self.r_mortgage
        
        # With depreciation
        actual_service = base_service * er_deviation
        
        # Additional cost
        additional_cost = actual_service - base_service
        
        return additional_cost
    
    def compute_import_price_index(self, exchange_rate: float) -> float:
        """
        Compute import price index given exchange rate.
        
        P_import = (ER / ER_baseline) ^ pass_through
        """
        er_ratio = exchange_rate / self.exchange_rate_baseline
        # Pass-through: not all exchange rate change hits consumers
        import_price = er_ratio ** self.pass_through_elasticity
        return import_price
    
    def compute_cpi(self, exchange_rate: float) -> float:
        """
        Compute consumer price index with import content.
        
        CPI = (1-α) * P_domestic + α * P_import
        where α = import_share
        """
        import_price = self.compute_import_price_index(exchange_rate)
        alpha = self.import_share_aggregate
        
        cpi = (1 - alpha) * 1.0 + alpha * import_price
        return cpi
    
    def print_summary(self):
        """Print parameter summary."""
        print("\n" + "="*60)
        print("SOE HANK PARAMETERS - CONSERVATIVE SCENARIO")
        print("="*60)
        
        print("\n1. Exchange Rate Parameters:")
        print(f"   Baseline MYR/USD: {self.exchange_rate_baseline:.2f}")
        print(f"   Persistence (AR1): {self.exchange_rate_persistence:.2f}")
        
        print("\n2. Import Price Parameters:")
        print(f"   Aggregate import share: {self.import_share_aggregate:.1%}")
        print(f"   Pass-through elasticity: {self.pass_through_elasticity:.2f}")
        
        print("\n3. Heterogeneous Import Exposure (CONSERVATIVE):")
        print("   Q1 (Poor):    22% - Local staples, informal sector")
        print("   Q2:           28% - Lower-middle, processed foods")
        print("   Q3 (Middle):  32% - HIGHEST (consumption upgrading)")
        print("   Q4:           28% - Upper-middle, more services")
        print("   Q5 (Rich):    25% - High services, luxury imports")
        print(f"   Q5/Q1 ratio: {self.import_share_by_quintile['Q5']/self.import_share_by_quintile['Q1']:.2f}x")
        
        print("\n4. USD Debt Exposure:")
        for q, share in self.usd_debt_share_by_quintile.items():
            print(f"   {q}: {share:.0%}")
        
        print("\n5. Capital Flow Parameters:")
        print(f"   Foreign interest rate: {self.r_foreign_baseline:.1%}")
        print(f"   Capital mobility: {self.capital_mobility:.1f}")
        print(f"   BNM intervention: {self.fx_intervention_strength:.0%}")
        
        if hasattr(self, 'fx_data_summary'):
            print("\n6. Data Validation:")
            if 'exchange_rate' in self.fx_data_summary:
                er = self.fx_data_summary['exchange_rate']
                print(f"   Historical ER mean: {er['mean']:.2f}")
        
        print("="*60)


def test_soe_params():
    """Test SOE parameters."""
    print("Testing SOE Parameters (CONSERVATIVE Scenario)...")
    
    params = SOEParams()
    params.print_summary()
    
    # Test import price calculation
    print("\nTesting import price mechanics:")
    for er_shock in [0.95, 1.00, 1.10, 1.20]:
        er = params.exchange_rate_baseline * er_shock
        imp_price = params.compute_import_price_index(er)
        cpi = params.compute_cpi(er)
        print(f"   ER {er_shock:.0%} ({er:.2f}): Import price={imp_price:.3f}, CPI={cpi:.3f}")
    
    # Test heterogeneous exposure
    print("\nTesting heterogeneous exposure:")
    for q in ['Q1', 'Q3', 'Q5']:
        urban = params.get_import_share(q, 'urban')
        rural = params.get_import_share(q, 'rural')
        usd_debt = params.get_usd_debt_share(q)
        print(f"   {q}: Urban={urban:.1%}, Rural={rural:.1%}, USD debt={usd_debt:.0%}")
    
    print("\n✓ SOE Parameters (CONSERVATIVE) ready for Milestone 2!")
    return params


if __name__ == "__main__":
    test_soe_params()
