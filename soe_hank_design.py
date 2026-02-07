"""
Small Open Economy HANK - Design Document
==========================================

Extension of Stage 5 HANK to include FX dynamics for Malaysia.

Key Features:
- Two-sector consumption (domestic + imported)
- Exchange rate as state variable
- Capital flows responding to interest differentials
- Terms of trade shocks
- Import price pass-through to inflation
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class SOEParams:
    """Parameters for Small Open Economy HANK."""
    
    # --- Original HANK parameters (inherited) ---
    beta: float = 0.92
    sigma: float = 1.5
    r_liquid: float = 0.03
    # ... etc
    
    # --- NEW: Open Economy Parameters ---
    
    # Exchange rate dynamics
    exchange_rate_initial: float = 4.75  # MYR/USD (approx 2024)
    exchange_rate_persistence: float = 0.85  # AR(1) coefficient
    exchange_rate_volatility: float = 0.05  # Std dev of shocks
    
    # Import share in consumption (Malaysia: ~30% of GDP)
    import_share: float = 0.30
    
    # Import price pass-through (how fast FX → consumer prices)
    # Literature: 0.3-0.6 for emerging markets
    pass_through: float = 0.45
    
    # Capital flow sensitivity to interest differential
    # How much capital flows in when r_MY > r_US
    capital_mobility: float = 2.0  # Elasticity
    
    # Foreign interest rate (Fed funds rate proxy)
    r_foreign: float = 0.055  # 5.5% (current Fed rate)
    r_foreign_persistence: float = 0.9
    
    # Terms of trade (commodity prices)
    # Malaysia exports: Palm oil, LNG, electronics
    # Malaysia imports: Food, fuel, capital goods
    commodity_price_initial: float = 1.0  # Normalized
    commodity_price_volatility: float = 0.08
    
    # Export demand elasticity
    export_elasticity: float = -1.5  # Price elasticity
    
    # Central bank FX intervention
    # BNM can smooth exchange rate volatility
    fx_intervention_strength: float = 0.3  # 0 = free float, 1 = fixed
    
    # Foreign debt (household + government)
    foreign_debt_gdp_ratio: float = 0.35  # Malaysia: ~35%
    foreign_debt_usd_share: float = 0.15  # 15% of debt is USD-denominated


class SOE_HANK:
    """
    Small Open Economy HANK with FX dynamics.
    
    State variables (added to base model):
    - exchange_rate: MYR/USD (higher = depreciation)
    - r_foreign: Foreign interest rate (Fed)
    - commodity_price: Terms of trade index
    - foreign_debt: USD-denominated debt stock
    
    Shocks:
    - FX shock (capital flow sudden stop)
    - Fed rate shock (US monetary tightening)
    - Commodity price shock (palm oil/LNG price drop)
    - Terms of trade shock
    """
    
    def __init__(self, params: SOEParams):
        self.params = params
        
    def compute_import_price_index(self, exchange_rate: float) -> float:
        """
        Import price index depends on exchange rate.
        P_import = P_world * ExchangeRate
        
        With pass-through:
        Domestic CPI effect = pass_through * (ΔER / ER)
        """
        # Normalize: ER = 4.75 is baseline
        er_deviation = exchange_rate / self.params.exchange_rate_initial
        
        # Import price inflation
        import_inflation = (er_deviation - 1) * self.params.pass_through
        
        return 1 + import_inflation
    
    def compute_real_consumption(
        self, 
        nominal_consumption: float,
        exchange_rate: float
    ) -> float:
        """
        Real consumption = Domestic consumption + Imported consumption
        
        C_real = [(1-α) * C_domestic^(ρ) + α * C_import^(ρ)]^(1/ρ)
        
        where α = import_share, ρ = elasticity of substitution
        
        Simplified: Real consumption is eroded by import prices
        """
        import_price = self.compute_import_price_index(exchange_rate)
        
        # Consumption basket: (1-α) domestic at price 1, α imported at price P_import
        # Real consumption = Nominal / PriceIndex
        price_index = (1 - self.params.import_share) * 1.0 + \
                      self.params.import_share * import_price
        
        return nominal_consumption / price_index
    
    def compute_capital_flows(
        self,
        r_domestic: float,
        r_foreign: float,
        exchange_rate: float,
        expected_depreciation: float
    ) -> float:
        """
        Capital flows based on uncovered interest parity (UIP) deviation.
        
        K_flow = mobility * (r_domestic - r_foreign - expected_depreciation)
        
        Positive = capital inflow (demand for MYR) → appreciation pressure
        Negative = capital outflow (sudden stop) → depreciation
        """
        interest_differential = r_domestic - r_foreign
        
        # UIP condition: r_MY = r_US + expected depreciation
        # Deviation from UIP drives flows
        uip_deviation = interest_differential - expected_depreciation
        
        capital_flow = self.params.capital_mobility * uip_deviation
        
        return capital_flow
    
    def update_exchange_rate(
        self,
        current_er: float,
        capital_flow: float,
        trade_balance: float,
        fx_shock: float = 0.0
    ) -> float:
        """
        Exchange rate dynamics with BNM intervention.
        
        ΔER/ER = -capital_flow + trade_balance_effect + fx_shock - intervention
        
        Negative sign: Inflows appreciate currency (lower ER = stronger MYR)
        """
        # Capital flow effect
        capital_effect = -0.1 * capital_flow  # Normalized coefficient
        
        # Trade balance effect (surplus appreciates)
        trade_effect = 0.05 * trade_balance
        
        # BNM intervention (smooths volatility)
        intervention = self.params.fx_intervention_strength * (
            current_er - self.params.exchange_rate_initial
        )
        
        # Total change
        delta_er = capital_effect + trade_effect + fx_shock - intervention
        
        # New exchange rate (AR(1) persistence)
        new_er = current_er * (1 + delta_er) * self.params.exchange_rate_persistence + \
                 self.params.exchange_rate_initial * (1 - self.params.exchange_rate_persistence)
        
        return new_er
    
    def compute_household_budget_soe(
        self,
        liquid_assets: torch.Tensor,
        illiquid_assets: torch.Tensor,
        income: torch.Tensor,
        consumption: torch.Tensor,
        exchange_rate: float,
        foreign_debt: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extended budget constraint with:
        - Import price effects on real consumption
        - Foreign debt servicing (if USD debt)
        """
        # Import price effect on purchasing power
        real_consumption = self.compute_real_consumption(
            consumption.item() if isinstance(consumption, torch.Tensor) else consumption,
            exchange_rate
        )
        
        # Foreign debt servicing (if any USD debt)
        if foreign_debt is not None and self.params.foreign_debt_usd_share > 0:
            # USD debt becomes more expensive when ER rises (depreciation)
            er_deviation = exchange_rate / self.params.exchange_rate_initial
            foreign_debt_service = foreign_debt * self.params.r_liquid * er_deviation
        else:
            foreign_debt_service = 0.0
        
        # Net resources available
        net_resources = income - foreign_debt_service
        
        # Savings (for asset accumulation)
        savings = net_resources - consumption
        
        return savings, foreign_debt_service


# ============================================================================
# Example Policy Simulations with FX
# ============================================================================

def simulate_fed_tightening_soe():
    """
    Simulate Fed rate hike impact on Malaysian households.
    
    Scenario: Fed raises rates by 100bps
    BNM dilemma: Follow (hike OPR) or let Ringgit depreciate?
    """
    params = SOEParams()
    model = SOE_HANK(params)
    
    print("Scenario: Fed Tightening (100bps hike)")
    print("="*50)
    
    # Initial state
    r_domestic = 0.03  # 3% OPR
    r_foreign = 0.055  # Fed at 5.5%
    exchange_rate = 4.75  # MYR/USD
    
    # Fed shock
    r_foreign_new = 0.065  # +100bps
    
    # Option 1: BNM follows (hikes OPR)
    print("\nOption 1: BNM follows Fed (+100bps OPR)")
    r_domestic_opt1 = 0.04  # 4% OPR
    expected_dep = 0  # No expected depreciation
    
    capital_flow = model.compute_capital_flows(
        r_domestic_opt1, r_foreign_new, exchange_rate, expected_dep
    )
    print(f"  Capital flow: {capital_flow:.2f} (% of GDP)")
    print(f"  Interest burden on households: +{((r_domestic_opt1 - 0.03) * 100):.0f}bps")
    
    # Option 2: BNM holds (lets Ringgit depreciate)
    print("\nOption 2: BNM holds OPR, lets Ringgit adjust")
    r_domestic_opt2 = 0.03  # Hold at 3%
    # Expected depreciation compensates for rate differential
    expected_dep = r_domestic_opt2 - r_foreign_new  # -3.5%
    
    new_er = exchange_rate * 1.05  # 5% depreciation
    import_price = model.compute_import_price_index(new_er)
    
    print(f"  Exchange rate: {exchange_rate:.2f} → {new_er:.2f} (depreciation)")
    print(f"  Import price index: {import_price:.3f} (+{(import_price-1)*100:.1f}%)")
    print(f"  Real consumption impact: -{(1 - 1/import_price)*100:.1f}%")
    
    print("\n" + "="*50)
    print("Conclusion: BNM faces impossible trinity tradeoff")


def simulate_commodity_shock():
    """
    Simulate palm oil/LNG price collapse (Malaysia's main exports).
    """
    params = SOEParams()
    model = SOE_HANK(params)
    
    print("\nScenario: Commodity Price Collapse (-30%)")
    print("="*50)
    
    # Palm oil and LNG prices drop 30%
    # → Trade balance deteriorates
    # → Ringgit depreciation pressure
    # → Import prices rise
    
    commodity_shock = -0.30
    trade_balance_impact = -0.04  # -4% of GDP
    
    print(f"  Export revenue impact: {commodity_shock:.0%}")
    print(f"  Trade balance impact: {trade_balance_impact:.1%} of GDP")
    print(f"  Exchange rate pressure: depreciation")
    print(f"  Q1 household impact: Higher food prices (rice, cooking oil)")


if __name__ == "__main__":
    simulate_fed_tightening_soe()
    simulate_commodity_shock()
