"""
SOE HANK Parameters - REVISED with Research-Based Calibration
==============================================================

Two scenarios for import share by income:
1. CONSERVATIVE: Q1 has lower import share (can't afford imports)
2. ALTERNATIVE: Q1 has higher import share (higher food/fuel ratio)
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

DATA_DIR = Path("/Users/sir/malaysia_hank/data")

@dataclass
class SOEParamsScenario:
    """
    FX exposure parameters with scenario selection.
    
    Scenarios:
    - 'conservative': Q1 has LOWER import share (default, research-based)
    - 'food_fuel': Q1 has HIGHER import share (higher food/fuel consumption)
    - 'flat': All quintiles have same import share
    """
    
    scenario: str = 'conservative'
    
    # Exchange rate parameters
    exchange_rate_baseline: float = 4.75
    exchange_rate_persistence: float = 0.85
    
    # Import price parameters
    import_share_aggregate: float = 0.30
    pass_through_elasticity: float = 0.45
    
    # Capital flow
    r_foreign_baseline: float = 0.055
    capital_mobility: float = 2.0
    
    def __post_init__(self):
        self._set_import_shares()
        self._save_scenario()
    
    def _set_import_shares(self):
        """Set import shares based on scenario."""
        
        if self.scenario == 'conservative':
            # Based on: Q1 consumes local goods, can't afford imports
            # Q3 has highest (consumption upgrading to processed foods)
            self.import_share_by_quintile = {
                'Q1': 0.22,  # Poor: local staples, informal sector
                'Q2': 0.28,  # Lower-middle: starting to buy processed
                'Q3': 0.32,  # Middle: HIGHEST (processed food, fuel-intensive)
                'Q4': 0.28,  # Upper-middle: more services
                'Q5': 0.25,  # Rich: high services, but luxury imports
            }
            self.rationale = """
            Q1 consumes local rice (SST 5%), local brands, informal services.
            Can't afford imported goods. Import exposure increases through middle
            class (Q3) as households upgrade to processed foods, then declines
            for rich (Q5) as services share increases.
            """
            
        elif self.scenario == 'food_fuel':
            # Based on: Q1 has higher food/fuel share, which have import content
            self.import_share_by_quintile = {
                'Q1': 0.38,  # High food/fuel share
                'Q2': 0.35,
                'Q3': 0.30,  # Aggregate average
                'Q4': 0.26,
                'Q5': 0.22,  # High services share
            }
            self.rationale = """
            Q1 spends 55% on food vs 20% for Q5. Food has import content
            (rice, cooking oil, wheat). Fuel also import-intensive.
            Services (consumed more by rich) are mostly domestic.
            """
            
        elif self.scenario == 'flat':
            # No heterogeneity - test sensitivity
            self.import_share_by_quintile = {
                'Q1': 0.30, 'Q2': 0.30, 'Q3': 0.30, 'Q4': 0.30, 'Q5': 0.30,
            }
            self.rationale = "No heterogeneity - sensitivity test"
            
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
    
    def _save_scenario(self):
        """Save scenario to JSON for documentation."""
        data = {
            'scenario': self.scenario,
            'rationale': self.rationale,
            'import_shares': self.import_share_by_quintile,
            'parameters': {
                'exchange_rate_baseline': self.exchange_rate_baseline,
                'pass_through': self.pass_through_elasticity,
                'r_foreign': self.r_foreign_baseline,
            }
        }
        
        output_file = DATA_DIR / f'soe_scenario_{self.scenario}.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved scenario to {output_file}")
    
    def print_comparison(self):
        """Print comparison of all scenarios."""
        print("\n" + "="*70)
        print("IMPORT SHARE SCENARIOS COMPARISON")
        print("="*70)
        
        scenarios = ['conservative', 'food_fuel', 'flat']
        print(f"\n{'Quintile':<10} {'Conservative':>15} {'Food/Fuel':>15} {'Flat':>15}")
        print("-"*70)
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            values = []
            for s in scenarios:
                p = SOEParamsScenario(scenario=s)
                values.append(p.import_share_by_quintile[q])
            print(f"{q:<10} {values[0]:>14.0%} {values[1]:>14.0%} {values[2]:>14.0%}")
        
        print("\nImplied Q5/Q1 ratio:")
        for s in scenarios:
            p = SOEParamsScenario(scenario=s)
            ratio = p.import_share_by_quintile['Q5'] / p.import_share_by_quintile['Q1']
            print(f"  {s:<15}: {ratio:.2f}x (Q5/Q1)")
        
        print("\n" + "="*70)
        print("SCENARIO DESCRIPTIONS:")
        print("-"*70)
        
        for s in scenarios:
            p = SOEParamsScenario(scenario=s)
            print(f"\n{s.upper()}:")
            print(f"  {p.rationale.strip()}")
        
        print("="*70)


def test_scenarios():
    """Test all scenarios."""
    print("Testing SOE Parameter Scenarios...")
    
    # Create and compare all scenarios
    comparator = SOEParamsScenario(scenario='conservative')
    comparator.print_comparison()
    
    print("\n✓ All scenarios created successfully!")
    print("\nRECOMMENDATION:")
    print("  Use 'conservative' as baseline (Q1 less exposed)")
    print("  Test sensitivity with 'food_fuel' scenario")
    print("  Document which assumption drives results")


if __name__ == "__main__":
    test_scenarios()
