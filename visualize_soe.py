"""
SOE HANK Visualization - Milestone 6
=====================================

Create publication-quality figures:
1. IRF plots with heterogeneous effects
2. Policy comparison charts
3. Distributional impact visualization
4. Summary dashboard
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
import json

from soe_params import SOEParams
from calibrate_soe import SOECalibrator
from policy_counterfactuals import PolicyCounterfactuals

OUTPUT_DIR = Path("/Users/sir/malaysia_hank/outputs/soe_simulation")
plt.style.use('seaborn-v0_8-whitegrid')


class SOEVisualizer:
    """Create visualizations for SOE HANK results."""
    
    def __init__(self, params: SOEParams):
        self.params = params
        self.calibrator = SOECalibrator(params)
        self.pc = PolicyCounterfactuals(params)
        
        # Color scheme
        self.colors = {
            'Q1': '#d62728',  # Red (poor)
            'Q2': '#ff7f0e',  # Orange
            'Q3': '#2ca02c',  # Green (middle)
            'Q4': '#1f77b4',  # Blue
            'Q5': '#9467bd',  # Purple (rich)
            'aggregate': '#333333',
        }
    
    def plot_irf_comparison(self, shock_names: list = None, T: int = 40):
        """
        Plot IRFs for multiple shocks side by side.
        
        Args:
            shock_names: List of shocks to plot
            T: Horizon
        """
        if shock_names is None:
            shock_names = ['fed_hike', 'capital_flight', 'commodity_bust']
        
        n_shocks = len(shock_names)
        fig, axes = plt.subplots(2, n_shocks, figsize=(15, 10))
        
        for idx, shock_name in enumerate(shock_names):
            irf = self.calibrator.run_mechanical_irf(shock_name, T)
            
            # Top row: Aggregate + all quintiles
            ax = axes[0, idx]
            
            # Plot each quintile
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                if q in irf['by_quintile']:
                    ax.plot(irf['by_quintile'][q], 
                           label=q, color=self.colors[q], linewidth=2)
            
            # Aggregate
            ax.plot(irf['aggregate'], '--', label='Aggregate', 
                   color=self.colors['aggregate'], linewidth=2)
            
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Quarters')
            ax.set_ylabel('% Change in Real Consumption')
            ax.set_title(f'{shock_name.replace("_", " ").title()}')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Bottom row: Exchange rate path
            ax = axes[1, idx]
            er_path = np.array(irf['exchange_rate'])
            er_baseline = self.params.exchange_rate_baseline
            er_pct = (er_path / er_baseline - 1) * 100
            
            ax.plot(er_pct, 'k-', linewidth=2)
            ax.fill_between(range(len(er_pct)), 0, er_pct, alpha=0.3, color='gray')
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Quarters')
            ax.set_ylabel('Exchange Rate (% Deviation)')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('SOE HANK: Impulse Response Functions', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig1_irf_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'fig1_irf_comparison.png'}")
        plt.close()
    
    def plot_heterogeneous_exposure(self):
        """
        Plot import exposure and USD debt by quintile.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Import share by quintile
        ax = axes[0]
        quintiles = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        import_shares = [self.params.import_share_by_quintile[q] for q in quintiles]
        
        bars = ax.bar(quintiles, import_shares, color=[self.colors[q] for q in quintiles],
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, import_shares):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.0%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Import Share of Consumption')
        ax.set_xlabel('Income Quintile')
        ax.set_title('Heterogeneous Import Exposure\n(CONSERVATIVE Scenario)')
        ax.set_ylim(0, 0.40)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add note
        ax.text(0.5, 0.95, 'Q3 (Middle) has highest exposure\n(consumption upgrading)',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
        
        # USD debt share by quintile
        ax = axes[1]
        usd_shares = [self.params.usd_debt_share_by_quintile[q] for q in quintiles]
        
        bars = ax.bar(quintiles, usd_shares, color=[self.colors[q] for q in quintiles],
                     edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, usd_shares):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.0%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('USD Debt Share')
        ax.set_xlabel('Income Quintile')
        ax.set_title('Heterogeneous USD Debt Exposure')
        ax.set_ylim(0, 0.35)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig2_heterogeneous_exposure.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'fig2_heterogeneous_exposure.png'}")
        plt.close()
    
    def plot_policy_comparison(self, shock_name: str = 'capital_flight'):
        """
        Plot BNM policy comparison.
        
        Args:
            shock_name: Shock to analyze
        """
        # Get policy results
        results = self.pc.bnm_policy_tradeoff(shock_name, T=20)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Q1 Impact by Policy
        ax = axes[0]
        policies = list(results.keys())
        q1_impacts = [results[p]['final_impacts']['Q1'] for p in policies]
        policy_names = [results[p]['name'] for p in policies]
        
        colors_bar = ['#d62728' if x < -0.5 else '#ff7f0e' if x < 0 else '#2ca02c' 
                     for x in q1_impacts]
        
        bars = ax.barh(policy_names, q1_impacts, color=colors_bar, edgecolor='black')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Q1 Real Consumption Impact (%)')
        ax.set_title('BNM Policy Tradeoff\nImpact on Q1 (Poor)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 2. Poverty Impact
        ax = axes[1]
        poverty_changes = [results[p]['poverty_change_pp'] for p in policies]
        
        colors_bar = ['#2ca02c' if x < 0 else '#d62728' for x in poverty_changes]
        bars = ax.barh(policy_names, poverty_changes, color=colors_bar, edgecolor='black')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Poverty Rate Change (pp)')
        ax.set_title('Poverty Impact\n(Negative = Better)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 3. Policy Instruments
        ax = axes[2]
        
        x = np.arange(len(policies))
        width = 0.25
        
        er_defense = [results[p]['er_defense'] * 100 for p in policies]
        opr_hike = [results[p]['opr_hike_bps'] / 100 for p in policies]  # Scale for visibility
        food_sub = [results[p]['food_subsidy'] * 100 for p in policies]
        
        ax.bar(x - width, er_defense, width, label='ER Defense (%)', color='#1f77b4')
        ax.bar(x, opr_hike, width, label='OPR Hike (/100bps)', color='#ff7f0e')
        ax.bar(x + width, food_sub, width, label='Food Subsidy (%)', color='#2ca02c')
        
        ax.set_ylabel('Policy Intensity')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace(' ', '\n') for p in policy_names], fontsize=8)
        ax.set_title('Policy Instruments Used')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'BNM Policy Analysis: {shock_name.replace("_", " ").title()} Shock',
                    fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig3_policy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'fig3_policy_comparison.png'}")
        plt.close()
    
    def plot_food_subsidy_efficiency(self):
        """
        Plot food subsidy cost-effectiveness.
        """
        results = self.pc.food_subsidy_analysis('capital_flight')
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Cost vs Effectiveness
        ax = axes[0]
        
        subsidies = list(results.keys())
        costs = [results[s]['cost_pct_gdp'] for s in subsidies]
        poverty_reductions = [results[s]['poverty_reduction_pp'] for s in subsidies]
        names = [results[s]['name'] for s in subsidies]
        
        colors_scatter = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        
        for i, (cost, reduction, name, color) in enumerate(zip(costs, poverty_reductions, names, colors_scatter)):
            ax.scatter(cost, reduction, s=200, c=color, edgecolors='black', linewidth=2, label=name, zorder=5)
        
        ax.set_xlabel('Cost (% of GDP)')
        ax.set_ylabel('Poverty Reduction (percentage points)')
        ax.set_title('Food Subsidy: Cost vs Effectiveness')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add efficiency line
        if max(costs) > 0:
            x_line = np.linspace(0, max(costs), 100)
            # Best case efficiency
            best_efficiency = min([results[s]['cost_per_poverty_pp'] for s in subsidies if results[s]['cost_per_poverty_pp'] > 0])
            y_line = x_line / best_efficiency
            ax.plot(x_line, y_line, 'k--', alpha=0.3, label=f'Best efficiency ({best_efficiency:.0f})')
        
        # 2. Cost per pp reduction
        ax = axes[1]
        
        cost_per_pp = [results[s]['cost_per_poverty_pp'] for s in subsidies]
        
        # Sort by efficiency
        sorted_data = sorted(zip(names, cost_per_pp, colors_scatter), key=lambda x: x[1])
        sorted_names, sorted_costs, sorted_colors = zip(*sorted_data)
        
        bars = ax.barh(sorted_names, sorted_costs, color=sorted_colors, edgecolor='black')
        ax.set_xlabel('Cost per pp Poverty Reduction (% GDP)')
        ax.set_title('Subsidy Efficiency Ranking\n(Lower is Better)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, sorted_costs):
            ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                   f'{val:.0f}%', va='center', fontsize=9)
        
        plt.suptitle('Food Subsidy Analysis During Depreciation', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig4_food_subsidy.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'fig4_food_subsidy.png'}")
        plt.close()
    
    def create_summary_dashboard(self):
        """
        Create summary dashboard with all key results.
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Title
        fig.suptitle('Malaysia SOE HANK: Summary Dashboard\n(FX Effects on Households)',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Import exposure (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        quintiles = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        import_shares = [self.params.import_share_by_quintile[q] for q in quintiles]
        ax1.bar(quintiles, import_shares, color=[self.colors[q] for q in quintiles],
               edgecolor='black')
        ax1.set_ylabel('Import Share')
        ax1.set_title('Import Exposure by Quintile')
        ax1.set_ylim(0, 0.35)
        
        # 2. USD debt (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        usd_shares = [self.params.usd_debt_share_by_quintile[q] for q in quintiles]
        ax2.bar(quintiles, usd_shares, color=[self.colors[q] for q in quintiles],
               edgecolor='black')
        ax2.set_ylabel('USD Debt Share')
        ax2.set_title('USD Debt Exposure by Quintile')
        ax2.set_ylim(0, 0.35)
        
        # 3. Key parameters (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        params_text = f"""
        Key Parameters:
        
        Baseline ER: {self.params.exchange_rate_baseline:.2f}
        Pass-through: {self.params.pass_through_elasticity:.0%}
        Import share (agg): {self.params.import_share_aggregate:.0%}
        Fed rate: {self.params.r_foreign_baseline:.1%}
        BNM intervention: {self.params.fx_intervention_strength:.0%}
        
        Scenario: CONSERVATIVE
        (Q1 has lower import exposure)
        """
        ax3.text(0.1, 0.5, params_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Capital flight IRF (middle left, spanning 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        irf = self.calibrator.run_mechanical_irf('capital_flight', 20)
        for q in ['Q1', 'Q3', 'Q5']:
            if q in irf['by_quintile']:
                ax4.plot(irf['by_quintile'][q], label=q, color=self.colors[q], linewidth=2)
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Quarters')
        ax4.set_ylabel('% Change in Real Consumption')
        ax4.set_title('Capital Flight Shock: Heterogeneous Impact')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Policy comparison (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        results = self.pc.bnm_policy_tradeoff('capital_flight', 20)
        policies = ['passive', 'defend_fx_50', 'defend_fx_100', 'defend_q1']
        policy_labels = ['Passive', 'Defend 50', 'Defend 100', 'Defend Q1']
        q1_impacts = [results[p]['final_impacts']['Q1'] for p in policies]
        
        colors_bar = [self.colors['Q1'] if x < -0.5 else '#ff7f0e' if x < 0 else '#2ca02c' 
                     for x in q1_impacts]
        ax5.barh(policy_labels, q1_impacts, color=colors_bar, edgecolor='black')
        ax5.axvline(0, color='black', linestyle='-', linewidth=1)
        ax5.set_xlabel('Q1 Impact (%)')
        ax5.set_title('BNM Policy Tradeoff\n(Q1 Welfare)')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Key findings (bottom, spanning all columns)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        findings_text = """
        KEY FINDINGS:
        
        1. HETEROGENEOUS EXPOSURE: Q3 (middle class) has highest import exposure (32%) due to consumption upgrading to processed foods.
           Q1 (poor) has lower exposure (22%) - consumes local staples.
        
        2. CAPITAL FLIGHT IMPACT: 15% depreciation reduces real consumption by -1.41% (Q1), -2.03% (Q3), -1.60% (Q5).
           Q3 is hit hardest, not Q1.
        
        3. BNM POLICY: Hiking OPR to defend Ringgit HELPS Q1 (-0.50% vs -1.41% passive) because Q1 has low USD debt (2%).
           The import price effect dominates the debt service effect.
        
        4. FOOD SUBSIDY: Targeted Q1 subsidy is most cost-effective (180% GDP per poverty pp) vs universal (300%).
        
        5. POLICY RECOMMENDATION: During capital flight, BNM should defend Ringgit with moderate OPR hikes + targeted food subsidies.
        """
        
        ax6.text(0.05, 0.95, findings_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.savefig(OUTPUT_DIR / 'fig5_summary_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'fig5_summary_dashboard.png'}")
        plt.close()


def create_all_visualizations():
    """Create all visualization figures."""
    print("\n" + "="*70)
    print("MILESTONE 6: CREATING VISUALIZATIONS")
    print("="*70)
    
    params = SOEParams()
    viz = SOEVisualizer(params)
    
    print("\n1. Creating IRF comparison plot...")
    viz.plot_irf_comparison()
    
    print("\n2. Creating heterogeneous exposure plot...")
    viz.plot_heterogeneous_exposure()
    
    print("\n3. Creating policy comparison plot...")
    viz.plot_policy_comparison('capital_flight')
    
    print("\n4. Creating food subsidy plot...")
    viz.plot_food_subsidy_efficiency()
    
    print("\n5. Creating summary dashboard...")
    viz.create_summary_dashboard()
    
    print("\n" + "="*70)
    print("✓ Milestone 6 Complete!")
    print(f"All figures saved to {OUTPUT_DIR}")
    print("="*70)
    print("\nGenerated figures:")
    for i, fig in enumerate([
        'fig1_irf_comparison.png',
        'fig2_heterogeneous_exposure.png',
        'fig3_policy_comparison.png',
        'fig4_food_subsidy.png',
        'fig5_summary_dashboard.png'
    ], 1):
        print(f"  {i}. {fig}")


if __name__ == "__main__":
    create_all_visualizations()
