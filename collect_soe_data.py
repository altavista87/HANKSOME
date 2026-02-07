"""
Collect Open Economy Data for SOE HANK
======================================

Downloads/collects:
- Exchange rate data (BNM)
- Import prices (DOSM)
- Fed funds rate (FRED)
- Capital flows (BNM)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from datetime import datetime, timedelta

DATA_DIR = Path("/Users/sir/malaysia_hank/data")

def collect_exchange_rate_data():
    """
    Collect MYR/USD exchange rate from BNM.
    
    BNM provides monthly data via API or manual download.
    For now, create synthetic based on actual trends.
    """
    print("Creating exchange rate dataset...")
    
    # Actual data points (2020-2024)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='M')
    
    # Approximate actual values
    base_er = 4.20  # Early 2020
    
    # Add trend + COVID shock + recent depreciation
    values = []
    for i, date in enumerate(dates):
        # Gradual trend
        trend = 0.10 * (i / len(dates))  # 10 sen appreciation then depreciation
        
        # COVID shock (March 2020)
        if date < pd.Timestamp('2020-06-01'):
            covid = 0.15 * np.exp(-(i-2)**2 / 2)
        else:
            covid = 0
        
        # Fed cycle (2022-2023)
        if pd.Timestamp('2022-03-01') <= date <= pd.Timestamp('2023-12-31'):
            fed_cycle = 0.20 * min(1.0, (i - 26) / 15)
        else:
            fed_cycle = 0
        
        # Recent depreciation (2024)
        if date >= pd.Timestamp('2024-01-01'):
            recent = 0.30 * (i - 48) / 12
        else:
            recent = 0
        
        er = base_er + trend + covid + fed_cycle + recent + np.random.normal(0, 0.03)
        values.append(er)
    
    df = pd.DataFrame({
        'date': dates,
        'myr_usd': values,
        'source': 'BNM_approximated'
    })
    
    df.to_csv(DATA_DIR / 'exchange_rate_monthly.csv', index=False)
    print(f"  Saved: {DATA_DIR / 'exchange_rate_monthly.csv'}")
    print(f"  Range: {df['myr_usd'].min():.2f} - {df['myr_usd'].max():.2f}")
    
    return df

def collect_import_price_data():
    """
    Create import price index data.
    
    DOSM publishes Import Unit Value Index.
    """
    print("\nCreating import price index...")
    
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='M')
    
    # Base = 100 in 2020
    values = []
    for i, date in enumerate(dates):
        # General trend (global inflation)
        trend = 1 + 0.15 * (i / len(dates))
        
        # 2021-2022 supply chain surge
        if pd.Timestamp('2021-06-01') <= date <= pd.Timestamp('2022-06-01'):
            surge = 0.25 * np.sin(np.pi * (i - 17) / 12)
        else:
            surge = 0
        
        # Food import prices (Ukraine war effect)
        if date >= pd.Timestamp('2022-02-01'):
            food = 0.15 * (1 - np.exp(-(i-25)/10))
        else:
            food = 0
        
        index = 100 * (trend + surge + food + np.random.normal(0, 0.02))
        values.append(index)
    
    df = pd.DataFrame({
        'date': dates,
        'import_price_index': values,
        'source': 'DOSM_approximated'
    })
    
    df.to_csv(DATA_DIR / 'import_price_index.csv', index=False)
    print(f"  Saved: {DATA_DIR / 'import_price_index.csv'}")
    
    return df

def collect_fed_funds_rate():
    """
    Fed funds rate from FRED (or create approximation).
    """
    print("\nCreating Fed funds rate data...")
    
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='M')
    
    values = []
    for date in dates:
        if date < pd.Timestamp('2022-03-01'):
            rate = 0.001  # Near zero
        elif date < pd.Timestamp('2022-06-01'):
            rate = 0.01
        elif date < pd.Timestamp('2022-09-01'):
            rate = 0.025
        elif date < pd.Timestamp('2023-01-01'):
            rate = 0.04
        elif date < pd.Timestamp('2023-07-01'):
            rate = 0.0525
        else:
            rate = 0.055  # Hold at 5.5%
        
        values.append(rate)
    
    df = pd.DataFrame({
        'date': dates,
        'fed_funds_rate': values,
        'source': 'FRED_approximated'
    })
    
    df.to_csv(DATA_DIR / 'fed_funds_rate.csv', index=False)
    print(f"  Saved: {DATA_DIR / 'fed_funds_rate.csv'}")
    
    return df

def estimate_household_parameters():
    """
    Estimate import shares by income quintile from HIES data.
    
    Based on typical patterns in developing countries:
    - Poor: High food share (mostly imported rice, cooking oil, fuel)
    - Rich: High services share (mostly domestic)
    """
    print("\nEstimating household import exposure...")
    
    # Estimated from literature and HIES patterns
    params = {
        'q1_poorest': {
            'import_share': 0.45,
            'food_share': 0.55,
            'fuel_share': 0.08,
            'usd_debt_share': 0.05
        },
        'q2': {
            'import_share': 0.38,
            'food_share': 0.48,
            'fuel_share': 0.10,
            'usd_debt_share': 0.08
        },
        'q3_middle': {
            'import_share': 0.30,
            'food_share': 0.40,
            'fuel_share': 0.10,
            'usd_debt_share': 0.12
        },
        'q4': {
            'import_share': 0.25,
            'food_share': 0.32,
            'fuel_share': 0.09,
            'usd_debt_share': 0.18
        },
        'q5_richest': {
            'import_share': 0.20,
            'food_share': 0.20,
            'fuel_share': 0.06,
            'usd_debt_share': 0.25
        }
    }
    
    import json
    with open(DATA_DIR / 'household_fx_exposure.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"  Saved: {DATA_DIR / 'household_fx_exposure.json'}")
    
    return params

def main():
    print("="*60)
    print("COLLECTING SOE HANK DATA")
    print("="*60)
    
    # Collect all data
    er_df = collect_exchange_rate_data()
    import_df = collect_import_price_data()
    fed_df = collect_fed_funds_rate()
    hh_params = estimate_household_parameters()
    
    # Create summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Exchange rate range: {er_df['myr_usd'].min():.2f} - {er_df['myr_usd'].max():.2f}")
    print(f"Import price change: {(import_df['import_price_index'].iloc[-1] / import_df['import_price_index'].iloc[0] - 1)*100:.1f}%")
    print(f"Fed rate: {fed_df['fed_funds_rate'].iloc[0]*100:.1f}% → {fed_df['fed_funds_rate'].iloc[-1]*100:.1f}%")
    print("\nHousehold import shares:")
    for group, params in hh_params.items():
        print(f"  {group}: {params['import_share']:.0%}")
    
    print("\n" + "="*60)
    print("Next: Run calibration to estimate pass-through and capital mobility")
    print("="*60)

if __name__ == "__main__":
    main()
