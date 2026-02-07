"""
Visualize Malaysia HANK Calibration Data
========================================
Create publication-ready plots from scraped data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

DATA_DIR = Path("/Users/sir/malaysia_hank/data")
OUTPUT_DIR = Path("/Users/sir/malaysia_hank/outputs/data_visualization")
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_macro_trends():
    """Plot GDP, debt, and interest rate trends."""
    
    # Load data
    gdp = pd.read_csv(DATA_DIR / 'dosm_gdp.csv')
    debt = pd.read_csv(DATA_DIR / 'bnm_household_debt.csv')
    opr = pd.read_csv(DATA_DIR / 'bnm_opr_rates_fallback.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # GDP Growth
    ax = axes[0, 0]
    ax.plot(gdp['year'], gdp['gdp_growth'], 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(gdp['year'], gdp['gdp_growth'], 0, 
                    where=(gdp['gdp_growth'] < 0), alpha=0.3, color='red')
    ax.set_xlabel('Year')
    ax.set_ylabel('GDP Growth (%)')
    ax.set_title('Malaysia GDP Growth Rate')
    ax.grid(True, alpha=0.3)
    
    # Household Debt to GDP
    ax = axes[0, 1]
    ax.plot(debt['year'], debt['household_debt_gdp'], 'g-s', linewidth=2, markersize=6)
    ax.fill_between(debt['year'], debt['household_debt_gdp'], alpha=0.3, color='green')
    ax.set_xlabel('Year')
    ax.set_ylabel('Household Debt (% of GDP)')
    ax.set_title('Malaysia Household Debt to GDP Ratio')
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # OPR History
    ax = axes[1, 0]
    opr['date'] = pd.to_datetime(opr['date'])
    ax.plot(opr['date'], opr['opr'], 'r-', linewidth=2)
    ax.fill_between(opr['date'], opr['opr'], alpha=0.3, color='red')
    ax.set_xlabel('Year')
    ax.set_ylabel('OPR (%)')
    ax.set_title('Bank Negara Malaysia - Overnight Policy Rate')
    ax.grid(True, alpha=0.3)
    
    # Debt Composition
    ax = axes[1, 1]
    x = debt['year']
    width = 0.25
    ax.bar(x - width, debt['housing_loan_share'], width, label='Housing', color='steelblue')
    ax.bar(x, debt['personal_loan_share'], width, label='Personal', color='coral')
    ax.bar(x + width, debt['credit_card_share'], width, label='Credit Card', color='lightgreen')
    ax.set_xlabel('Year')
    ax.set_ylabel('Share of Total Debt (%)')
    ax.set_title('Household Debt Composition')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'macro_trends.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'macro_trends.png'}")
    plt.close()


def plot_labor_market():
    """Plot labor force statistics."""
    
    labor = pd.read_csv(DATA_DIR / 'dosm_labor_force.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sector composition
    ax = axes[0]
    x = labor['year']
    ax.plot(x, labor['formal_sector_share'], 'b-o', label='Formal', linewidth=2)
    ax.plot(x, labor['informal_sector_share'], 'r-s', label='Informal', linewidth=2)
    ax.plot(x, labor['public_sector_share'], 'g-^', label='Public', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Employment Share (%)')
    ax.set_title('Malaysia Labor Market Structure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 80)
    
    # Unemployment rate
    ax = axes[1]
    ax.plot(labor['year'], labor['unemployment_rate'], 'purple', marker='o', linewidth=2, markersize=8)
    ax.fill_between(labor['year'], labor['unemployment_rate'], alpha=0.3, color='purple')
    ax.axhline(y=labor['unemployment_rate'].mean(), color='r', linestyle='--', 
               alpha=0.5, label=f'Mean: {labor["unemployment_rate"].mean():.1f}%')
    ax.set_xlabel('Year')
    ax.set_ylabel('Unemployment Rate (%)')
    ax.set_title('Malaysia Unemployment Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'labor_market.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'labor_market.png'}")
    plt.close()


def plot_epf_analysis():
    """Plot EPF statistics."""
    
    epf = pd.read_csv(DATA_DIR / 'epf_statistics.csv')
    dividend = pd.read_csv(DATA_DIR / 'epf_dividend_history.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total savings growth
    ax = axes[0, 0]
    ax.plot(epf['year'], epf['total_savings_billion_rm'], 'b-o', linewidth=2, markersize=6)
    ax.fill_between(epf['year'], epf['total_savings_billion_rm'], alpha=0.3, color='blue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Savings (Billion RM)')
    ax.set_title('EPF Total Savings Growth')
    ax.grid(True, alpha=0.3)
    
    # Average vs Median savings
    ax = axes[0, 1]
    ax.plot(epf['year'], epf['average_savings_per_member_rm'], 'b-o', 
            label='Average', linewidth=2, markersize=6)
    ax.plot(epf['year'], epf['median_savings_age_54_rm'], 'r-s', 
            label='Median (Age 54)', linewidth=2, markersize=6)
    ax.fill_between(epf['year'], epf['average_savings_per_member_rm'], 
                    epf['median_savings_age_54_rm'], alpha=0.2, color='gray')
    ax.set_xlabel('Year')
    ax.set_ylabel('Savings (RM)')
    ax.set_title('EPF Average vs Median Savings')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Members below RM 10k at age 54 (inadequate savings)
    ax = axes[1, 0]
    ax.bar(epf['year'], epf['members_below_10k_age_54_pct'], color='coral', alpha=0.7)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('EPF Members with < RM 10,000 at Age 54')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Historical dividend rates
    ax = axes[1, 1]
    ax.plot(dividend['year'], dividend['dividend_rate'], 'g-o', linewidth=2, markersize=4)
    ax.axhline(y=dividend['dividend_rate'].mean(), color='r', linestyle='--', 
               alpha=0.5, label=f'Average: {dividend["dividend_rate"].mean():.2f}%')
    ax.fill_between(dividend['year'], dividend['dividend_rate'], alpha=0.3, color='green')
    ax.set_xlabel('Year')
    ax.set_ylabel('Dividend Rate (%)')
    ax.set_title('EPF Historical Dividend Rates (2000-2024)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'epf_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'epf_analysis.png'}")
    plt.close()


def plot_inequality():
    """Plot inequality measures."""
    
    hies = pd.read_csv(DATA_DIR / 'dosm_hies_summary.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gini coefficient
    ax = axes[0]
    ax.plot(hies['year'], hies['income_gini'], 'b-o', linewidth=2, markersize=8)
    ax.fill_between(hies['year'], hies['income_gini'], 0.35, alpha=0.3, color='blue')
    ax.axhline(y=0.4, color='g', linestyle='--', alpha=0.5, label='Gini = 0.4 (threshold)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Malaysia Income Inequality (Gini)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.35, 0.45)
    
    # Income shares
    ax = axes[1]
    x = hies['year']
    width = 0.35
    ax.bar(x - width/2, hies['bottom40_income_share'], width, label='Bottom 40%', color='lightblue')
    ax.bar(x + width/2, hies['top20_income_share'], width, label='Top 20%', color='darkblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Income Share (%)')
    ax.set_title('Income Distribution: Bottom 40% vs Top 20%')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'inequality.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'inequality.png'}")
    plt.close()


def plot_housing_health():
    """Plot housing and health data."""
    
    housing = pd.read_csv(DATA_DIR / 'napic_housing.csv')
    health = pd.read_csv(DATA_DIR / 'moh_health.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # House prices
    ax = axes[0, 0]
    ax.plot(housing['year'], housing['median_house_price_kl_rm']/1000, 'b-o', 
            label='Kuala Lumpur', linewidth=2, markersize=6)
    ax.plot(housing['year'], housing['median_house_price_national_rm']/1000, 'r-s', 
            label='National', linewidth=2, markersize=6)
    ax.set_xlabel('Year')
    ax.set_ylabel('Median Price (RM \'000)')
    ax.set_title('Malaysia Median House Prices')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Homeownership
    ax = axes[0, 1]
    ax.plot(housing['year'], housing['homeownership_rate'], 'g-^', 
            linewidth=2, markersize=8, color='forestgreen')
    ax.fill_between(housing['year'], housing['homeownership_rate'], 
                    alpha=0.3, color='forestgreen')
    ax.set_xlabel('Year')
    ax.set_ylabel('Homeownership Rate (%)')
    ax.set_title('Malaysia Homeownership Rate')
    ax.set_ylim(70, 85)
    ax.grid(True, alpha=0.3)
    
    # Chronic disease prevalence
    ax = axes[1, 0]
    ax.bar(health['year'], health['chronic_disease_prevalence'], 
           color='coral', alpha=0.7, width=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Malaysia Chronic Disease Prevalence')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Healthcare utilization
    ax = axes[1, 1]
    x = health['year']
    width = 1.5
    ax.bar(x - width/2, health['public_healthcare_utilization'], width, 
           label='Public', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, health['private_healthcare_utilization'], width, 
           label='Private', color='coral', alpha=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Utilization Share (%)')
    ax.set_title('Healthcare Utilization: Public vs Private')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'housing_health.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'housing_health.png'}")
    plt.close()


def create_calibration_summary():
    """Create a visual summary of calibration targets."""
    
    with open(DATA_DIR / 'calibration_targets.json', 'r') as f:
        targets = json.load(f)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select key targets for visualization
    key_targets = {
        'Income Gini': targets['income_gini'],
        'Household Debt/GDP': targets['household_debt_gdp'],
        'Formal Employment': targets['formal_share'],
        'Informal Employment': targets['informal_share'],
        'Homeownership Rate': targets['homeownership_rate'],
        'Public Healthcare': targets['public_healthcare_share'],
        'Chronic Disease Rate': targets['chronic_disease_rate'],
        'OPR Rate': targets['opr_rate'],
        'EPF Return': targets['epf_return'],
    }
    
    names = list(key_targets.keys())
    values = list(key_targets.values())
    
    colors = ['steelblue' if v < 0.5 else 'coral' for v in values]
    
    bars = ax.barh(names, values, color=colors, alpha=0.7)
    ax.set_xlabel('Value (ratio or rate)')
    ax.set_title('Malaysia HANK Model - Key Calibration Targets', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'calibration_targets_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'calibration_targets_summary.png'}")
    plt.close()


def main():
    """Generate all visualizations."""
    
    print("="*60)
    print("MALAYSIA HANK DATA VISUALIZATION")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    plot_macro_trends()
    plot_labor_market()
    plot_epf_analysis()
    plot_inequality()
    plot_housing_health()
    create_calibration_summary()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
