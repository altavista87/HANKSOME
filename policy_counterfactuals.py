"""
SOE HANK Policy Counterfactuals - Milestone 5
==============================================

Policy experiments:
1. BNM policy tradeoffs (OPR vs FX stability)
2. Poverty impact quantification
3. Food subsidy during depreciation
4. Optimal policy frontier
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json

from soe_params import SOEParams
from calibrate_soe import SOECalibrator

OUTPUT_DIR = Path("/Users/sir/malaysia_hank/outputs/soe_simulation")


class PolicyCounterfactuals:
    """
    Run policy counterfactuals for SOE HANK.
    """
    
    def __init__(self, params: SOEParams):
        self.params = params
        self.calibrator = SOECalibrator(params)
        
        # Poverty line (Malaysia PLI 2022)
        self.poverty_line_monthly = 2208  # RM per household
        self.model_to_rm = 1500  # Conversion factor (calibrated)
    
    def compute_poverty_rate(
        self,
        income: torch.Tensor,
        cpi: float = 1.0
    ) -> Tuple[float, float]:
        """
        Compute poverty rate given income and price level.
        
        Args:
            income: Nominal income (model units)
            cpi: Price level (1.0 = baseline)
        
        Returns:
            poverty_rate: % below poverty line
            mean_income_real: Real mean income
        """
        # Convert to RM
        income_rm = income * self.model_to_rm
        
        # Real income (accounting for CPI)
        real_income_rm = income_rm / cpi
        
        # Poverty rate
        below_poverty = (real_income_rm < self.poverty_line_monthly).float()
        poverty_rate = below_poverty.mean().item()
        
        mean_income_real = real_income_rm.mean().item()
        
        return poverty_rate, mean_income_real
    
    def bnm_policy_tradeoff(
        self,
        shock_name: str = 'capital_flight',
        T: int = 40
    ) -> Dict:
        """
        Analyze BNM's tradeoff: OPR hike vs FX stability vs Q1 welfare.
        
        Three policy responses:
        1. PASSIVE: Let Ringgit depreciate, hold OPR
        2. DEFEND FX: Hike OPR to stabilize Ringgit
        3. DEFEND Q1: Hike OPR + food subsidy
        
        Args:
            shock_name: Shock to analyze
            T: Horizon
        
        Returns:
            comparison: Results for each policy
        """
        print(f"\n{'='*70}")
        print(f"BNM POLICY TRADEOFF ANALYSIS")
        print(f"Shock: {shock_name}")
        print(f"{'='*70}")
        
        # Create representative household population
        n = 5000
        state = {
            'base_income': torch.exp(torch.randn(n) * 0.5 + 0.5),
            'mortgage_debt': torch.rand(n) * 100,
        }
        
        # Baseline poverty
        baseline_poverty, baseline_income = self.compute_poverty_rate(
            state['base_income']
        )
        
        print(f"\nBaseline (pre-shock):")
        print(f"  Poverty rate: {baseline_poverty:.1%}")
        print(f"  Mean income: RM {baseline_income:.0f}")
        
        # Define policies
        policies = {
            'passive': {
                'name': 'PASSIVE',
                'description': 'Hold OPR, let Ringgit adjust',
                'er_defense': 0.0,  # No defense
                'opr_hike': 0.0,
                'food_subsidy': 0.0,
            },
            'defend_fx_50': {
                'name': 'DEFEND FX (50bps)',
                'description': 'Hike OPR 50bps, partial ER defense',
                'er_defense': 0.5,  # 50% less depreciation
                'opr_hike': 0.005,
                'food_subsidy': 0.0,
            },
            'defend_fx_100': {
                'name': 'DEFEND FX (100bps)',
                'description': 'Hike OPR 100bps, full ER defense',
                'er_defense': 1.0,  # Full defense
                'opr_hike': 0.010,
                'food_subsidy': 0.0,
            },
            'defend_q1': {
                'name': 'DEFEND Q1',
                'description': 'Hike OPR 50bps + 20% food subsidy for Q1',
                'er_defense': 0.5,
                'opr_hike': 0.005,
                'food_subsidy': 0.20,  # 20% food price subsidy
            },
        }
        
        results = {}
        
        for policy_key, policy in policies.items():
            print(f"\n{'-'*70}")
            print(f"POLICY: {policy['name']}")
            print(f"{policy['description']}")
            print(f"{'-'*70}")
            
            # Get mechanical IRF with adjustments
            irf = self.calibrator.run_mechanical_irf(shock_name, T)
            
            # Apply policy adjustments
            # 1. ER defense reduces depreciation impact
            er_defense = policy['er_defense']
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                if q in irf['by_quintile']:
                    irf['by_quintile'][q] = [
                        x * (1 - er_defense) for x in irf['by_quintile'][q]
                    ]
            
            # 2. OPR hike increases debt service (especially for rich)
            opr_hike = policy['opr_hike']
            # Simplified: OPR hike affects all households with debt
            # Rich have more debt, so affected more
            opr_impact = {
                'Q1': -opr_hike * 0.5 * 100,   # Less debt
                'Q2': -opr_hike * 0.7 * 100,
                'Q3': -opr_hike * 0.9 * 100,
                'Q4': -opr_hike * 1.1 * 100,
                'Q5': -opr_hike * 1.3 * 100,   # More debt
            }
            
            # 3. Food subsidy (only for Q1)
            food_subsidy = policy['food_subsidy']
            
            # Compute final impacts
            final_impacts = {}
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                if q in irf['by_quintile']:
                    fx_impact = min(irf['by_quintile'][q])  # Most negative
                    debt_impact = opr_impact.get(q, 0)
                    
                    # Food subsidy only for Q1
                    subsidy_impact = food_subsidy * 100 if q == 'Q1' else 0
                    
                    final_impacts[q] = fx_impact + debt_impact + subsidy_impact
            
            # Aggregate impact (weighted average)
            agg_impact = np.mean(list(final_impacts.values()))
            
            # Estimate poverty impact
            # Assume income distribution shifts by final_impacts
            q1_impact = final_impacts.get('Q1', 0)
            
            # Simplified: poverty increases if Q1 income falls
            # Real income after shock
            income_after = state['base_income'] * (1 + q1_impact/100)
            
            # CPI after shock (worst case for Q1)
            max_dep = max(irf['exchange_rate']) / self.params.exchange_rate_baseline
            cpi_after = self.params.compute_cpi(max_dep * (1 - er_defense))
            
            poverty_after, mean_income_after = self.compute_poverty_rate(
                income_after, cpi_after
            )
            
            poverty_change = (poverty_after - baseline_poverty) * 100  # pp change
            
            # Store results
            results[policy_key] = {
                'name': policy['name'],
                'description': policy['description'],
                'fx_impacts': {q: min(irf['by_quintile'][q]) for q in irf['by_quintile']},
                'opr_impacts': opr_impact,
                'final_impacts': final_impacts,
                'aggregate_impact': agg_impact,
                'poverty_rate': poverty_after,
                'poverty_change_pp': poverty_change,
                'mean_income_rm': mean_income_after,
                'er_defense': er_defense,
                'opr_hike_bps': opr_hike * 10000,
                'food_subsidy': food_subsidy,
            }
            
            # Print summary
            print(f"\n  Impacts on real consumption:")
            for q in ['Q1', 'Q3', 'Q5']:
                if q in final_impacts:
                    print(f"    {q}: {final_impacts[q]:+.2f}%")
            
            print(f"\n  Poverty impact:")
            print(f"    Baseline: {baseline_poverty:.1%}")
            print(f"    After shock: {poverty_after:.1%}")
            print(f"    Change: {poverty_change:+.2f}pp")
            
            print(f"\n  Policy instruments:")
            print(f"    ER defense: {er_defense:.0%}")
            print(f"    OPR hike: {opr_hike*10000:.0f}bps")
            print(f"    Food subsidy: {food_subsidy:.0%}")
        
        # Summary comparison
        print(f"\n{'='*70}")
        print("POLICY COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n{'Policy':<20} {'Q1 Impact':>12} {'Poverty':>10} {'ΔPoverty':>10}")
        print(f"{'-'*70}")
        for key, res in results.items():
            print(f"{res['name']:<20} {res['final_impacts']['Q1']:>+11.2f}% " +
                  f"{res['poverty_rate']:>9.1%} {res['poverty_change_pp']:>+9.2f}pp")
        
        # Find best policy for Q1
        best_q1 = min(results.items(), key=lambda x: x[1]['final_impacts']['Q1'])
        print(f"\nBest for Q1 (poor): {best_q1[1]['name']}")
        
        # Find best for poverty reduction
        best_poverty = min(results.items(), key=lambda x: x[1]['poverty_change_pp'])
        print(f"Best for poverty: {best_poverty[1]['name']}")
        
        return results
    
    def optimal_policy_frontier(
        self,
        shock_name: str = 'capital_flight',
        T: int = 40
    ) -> pd.DataFrame:
        """
        Compute optimal policy frontier.
        
        Tradeoff between:
        - X-axis: Exchange rate stability (lower depreciation)
        - Y-axis: Q1 welfare (less negative impact)
        
        Returns:
            frontier: DataFrame of policy frontier points
        """
        print(f"\n{'='*70}")
        print(f"OPTIMAL POLICY FRONTIER")
        print(f"Shock: {shock_name}")
        print(f"{'='*70}")
        
        # Grid of policies
        er_defense_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
        opr_hike_levels = [0.0, 0.0025, 0.005, 0.0075, 0.01]
        
        frontier_points = []
        
        n = 3000
        state = {
            'base_income': torch.exp(torch.randn(n) * 0.5 + 0.5),
            'mortgage_debt': torch.rand(n) * 100,
        }
        
        # Baseline IRF
        irf = self.calibrator.run_mechanical_irf(shock_name, T)
        baseline_dep = (max(irf['exchange_rate']) / 
                       self.params.exchange_rate_baseline - 1)
        
        for er_defense in er_defense_levels:
            for opr_hike in opr_hike_levels:
                # Compute impacts
                q1_fx_impact = min(irf['by_quintile']['Q1']) * (1 - er_defense)
                
                # OPR impact (higher for rich, but affects Q1 too)
                q1_opr_impact = -opr_hike * 0.5 * 100
                
                q1_total = q1_fx_impact + q1_opr_impact
                
                # Actual depreciation
                actual_dep = baseline_dep * (1 - er_defense)
                
                frontier_points.append({
                    'er_defense': er_defense,
                    'opr_hike_bps': opr_hike * 10000,
                    'depreciation': actual_dep * 100,
                    'q1_impact': q1_total,
                    'policy': f"D{int(er_defense*100)}_O{int(opr_hike*10000)}"
                })
        
        df = pd.DataFrame(frontier_points)
        
        # Find efficient frontier
        # A policy is dominated if another has both less depreciation and better Q1 impact
        efficient = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if i != j:
                    # Other has less depreciation AND better Q1 impact
                    if (other['depreciation'] <= row['depreciation'] and 
                        other['q1_impact'] >= row['q1_impact'] and
                        (other['depreciation'] < row['depreciation'] or 
                         other['q1_impact'] > row['q1_impact'])):
                        dominated = True
                        break
            if not dominated:
                efficient.append(row)
        
        efficient_df = pd.DataFrame(efficient)
        
        print(f"\nPolicy frontier ({len(efficient_df)} efficient points):")
        print(f"\n{'Policy':<12} {'Depreciation':>13} {'Q1 Impact':>11}")
        print(f"{'-'*40}")
        for _, row in efficient_df.sort_values('depreciation').iterrows():
            print(f"{row['policy']:<12} {row['depreciation']:>+12.1f}% {row['q1_impact']:>+10.2f}%")
        
        # Extreme points
        min_dep = efficient_df.loc[efficient_df['depreciation'].idxmin()]
        max_q1 = efficient_df.loc[efficient_df['q1_impact'].idxmax()]
        
        print(f"\nExtreme points:")
        print(f"  Min depreciation: {min_dep['policy']} " +
              f"({min_dep['depreciation']:+.1f}%, {min_dep['q1_impact']:+.2f}%)")
        print(f"  Max Q1 welfare: {max_q1['policy']} " +
              f"({max_q1['depreciation']:+.1f}%, {max_q1['q1_impact']:+.2f}%)")
        
        return efficient_df
    
    def food_subsidy_analysis(
        self,
        shock_name: str = 'capital_flight'
    ) -> Dict:
        """
        Analyze food subsidy as counter-cyclical policy during depreciation.
        
        Compare:
        1. No subsidy
        2. Universal subsidy (all households)
        3. Targeted subsidy (Q1 only)
        4. Targeted subsidy (B40 - Q1+Q2)
        
        Args:
            shock_name: Shock scenario
        
        Returns:
            comparison: Results for each subsidy design
        """
        print(f"\n{'='*70}")
        print(f"FOOD SUBSIDY ANALYSIS")
        print(f"Shock: {shock_name}")
        print(f"{'='*70}")
        
        # Subsidy scenarios
        subsidies = {
            'none': {
                'name': 'No Subsidy',
                'target': None,
                'rate': 0.0,
            },
            'universal_10': {
                'name': 'Universal 10%',
                'target': 'all',
                'rate': 0.10,
            },
            'q1_30': {
                'name': 'Q1 Targeted 30%',
                'target': 'Q1',
                'rate': 0.30,
            },
            'b40_20': {
                'name': 'B40 Targeted 20%',
                'target': 'B40',  # Q1 + Q2
                'rate': 0.20,
            },
        }
        
        n = 3000
        state = {
            'base_income': torch.exp(torch.randn(n) * 0.5 + 0.5),
            'mortgage_debt': torch.rand(n) * 100,
        }
        
        # Get IRF
        irf = self.calibrator.run_mechanical_irf(shock_name, T=20)
        baseline_poverty, _ = self.compute_poverty_rate(state['base_income'])
        
        results = {}
        
        for sub_key, sub in subsidies.items():
            # Base impact (no subsidy)
            q1_base_impact = min(irf['by_quintile']['Q1'])
            
            # Apply subsidy
            if sub['target'] == 'all':
                # All quintiles benefit
                subsidy_effect = {q: sub['rate'] * 100 for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']}
            elif sub['target'] == 'Q1':
                subsidy_effect = {'Q1': sub['rate'] * 100}
            elif sub['target'] == 'B40':
                subsidy_effect = {'Q1': sub['rate'] * 100, 'Q2': sub['rate'] * 100}
            else:
                subsidy_effect = {}
            
            # Final Q1 impact
            q1_final = q1_base_impact + subsidy_effect.get('Q1', 0)
            
            # Estimate poverty (simplified)
            poverty_change = (q1_final / 100) * (-0.5)  # 1% income = 0.5pp poverty change
            
            # Cost estimation (% of GDP)
            if sub['target'] == 'all':
                cost = sub['rate'] * 0.30  # 30% of consumption is food
            elif sub['target'] == 'Q1':
                cost = sub['rate'] * 0.30 * 0.20  # 20% of households
            elif sub['target'] == 'B40':
                cost = sub['rate'] * 0.30 * 0.40  # 40% of households
            else:
                cost = 0
            
            results[sub_key] = {
                'name': sub['name'],
                'target': sub['target'],
                'rate': sub['rate'],
                'q1_impact': q1_final,
                'poverty_reduction_pp': -poverty_change,
                'cost_pct_gdp': cost * 100,
                'cost_per_poverty_pp': cost * 100 / max(poverty_change, 0.01),
            }
            
            print(f"\n{sub['name']}:")
            print(f"  Q1 impact: {q1_final:+.2f}%")
            print(f"  Poverty reduction: {-poverty_change:+.2f}pp")
            print(f"  Cost: {cost*100:.2f}% of GDP")
            print(f"  Cost per pp poverty reduction: {results[sub_key]['cost_per_poverty_pp']:.2f}% GDP")
        
        # Efficiency ranking
        print(f"\n{'='*70}")
        print("SUBSIDY EFFICIENCY RANKING")
        print(f"{'='*70}")
        
        efficiency = sorted(
            results.items(),
            key=lambda x: x[1]['cost_per_poverty_pp']
        )
        
        print(f"\n{'Rank':<6} {'Policy':<20} {'Cost/pp':>12}")
        print(f"{'-'*40}")
        for i, (key, res) in enumerate(efficiency, 1):
            print(f"{i:<6} {res['name']:<20} {res['cost_per_poverty_pp']:>11.2f}%")
        
        return results


def test_policy_counterfactuals():
    """Test policy counterfactuals."""
    print("\n" + "="*70)
    print("TESTING MILESTONE 5: POLICY COUNTERFACTUALS")
    print("="*70)
    
    params = SOEParams()
    pc = PolicyCounterfactuals(params)
    
    # 1. BNM Policy Tradeoff
    print("\n" + "="*70)
    print("1. BNM POLICY TRADEOFF ANALYSIS")
    print("="*70)
    tradeoff_results = pc.bnm_policy_tradeoff('capital_flight', T=20)
    
    # 2. Optimal Policy Frontier
    print("\n" + "="*70)
    print("2. OPTIMAL POLICY FRONTIER")
    print("="*70)
    frontier = pc.optimal_policy_frontier('capital_flight', T=20)
    
    # 3. Food Subsidy Analysis
    print("\n" + "="*70)
    print("3. FOOD SUBSIDY ANALYSIS")
    print("="*70)
    subsidy_results = pc.food_subsidy_analysis('capital_flight')
    
    # Save all results
    output = {
        'tradeoff': tradeoff_results,
        'frontier': frontier.to_dict(),
        'subsidy': subsidy_results,
    }
    
    with open(OUTPUT_DIR / 'policy_counterfactuals.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("✓ Milestone 5 Complete!")
    print(f"Results saved to {OUTPUT_DIR / 'policy_counterfactuals.json'}")
    print("="*70)
    
    return pc


if __name__ == "__main__":
    test_policy_counterfactuals()
