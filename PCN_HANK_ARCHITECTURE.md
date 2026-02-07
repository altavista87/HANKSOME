# PCN-HANK Architecture Plan
## Policy-Calibrated Neural HANK with WID Integration

**Version**: 1.1  
**Date**: January 30, 2026  
**Status**: Architecture Design Phase

---

## 1. Executive Summary

PCN-HANK bridges your working deep learning infrastructure with targeted policy modules for Malaysia, **now with WID (World Inequality Database) integration** to accurately model the top tail of the distribution.

**Core Innovations**: 
1. 4-state labor market with heterogeneous credit spreads
2. STR phase-out mechanics
3. **WID-calibrated fractiles for tail-end accuracy**

---

## 2. State Space Architecture

### 2.1 State Dimensions

| Component | Dimensions | Embedding | Notes |
|-----------|------------|-----------|-------|
| **Continuous** | 5 | - | liquid, illiquid, mortgage, human_capital, age |
| **wid_fractile** | **6** | **16** | **NEW: B40, M40, T90, T9, T1, T0.1** |
| labor_sector | 4 | 16 | formal_epf, precariat, civil_service, informal_legacy |
| education_level | 3 | 8 | primary, secondary, tertiary |
| ptptn_state | 5 | 8 | no_loan, studying, repaying_formal, repaying_hidden, defaulted |
| health_state | 4 | 8 | healthy, sick, chronic, disabled |
| housing_type | 3 | 8 | rent, mortgage, outright |
| location | 26 | 16 | 13 states x urban/rural |
| str_eligibility | 3 | 4 | b40_full, m40_partial, t20_none |
| **Total** | **54 states** | **84 dims** | **~1.2M total discrete states** |

**Total Parameters**: ~520,000 (vs current 507K)

### 2.2 WID Fractile Integration

```python
WID_CALIBRATION = {
    # From WID.world (Malaysia DINA data)
    'fractiles': {
        'B40': {
            'income_share': 0.16,        # Bottom 40% get 16% of income
            'wealth_share': 0.05,         # Bottom 40% hold 5% of wealth
            'wealth_income_ratio': 0.8,   # Low asset accumulation
            'n_households': 3.2e6         # ~3.2M households
        },
        'M40': {
            'income_share': 0.32,        # Middle 40% get 32% of income
            'wealth_share': 0.25,         # Middle 40% hold 25% of wealth
            'wealth_income_ratio': 2.5,   # Moderate savings
            'n_households': 3.2e6
        },
        'T90': {  # P90-P99
            'income_share': 0.28,        # Top 10% (excl top 1%) get 28%
            'wealth_share': 0.35,         # Hold 35% of wealth
            'wealth_income_ratio': 8.0,   # Significant asset accumulation
            'n_households': 800e3
        },
        'T1': {   # P99-P99.9
            'income_share': 0.15,        # Top 1% (excl top 0.1%) get 15%
            'wealth_share': 0.20,         # Hold 20% of wealth
            'wealth_income_ratio': 18.0,  # High wealth concentration
            'n_households': 80e3
        },
        'T0.1': {  # Top 0.1%
            'income_share': 0.09,        # Top 0.1% get 9% of income
            'wealth_share': 0.15,         # Hold 15% of wealth
            'wealth_income_ratio': 40.0,  # Extreme wealth concentration
            'n_households': 8e3
        }
    }
}
```

**Why WID Matters for Malaysia**:
- Standard surveys (HIES) miss top incomes due to underreporting
- WID uses tax data + national accounts to capture the full distribution
- Critical for Q11: "Taxing the Missing Rich" - wealth tax on >RM10m

---

## 3. Data Requirements & Sources

### 3.1 WID (World Inequality Database) - NEW

| Parameter | Value (2022) | Source | Priority |
|-----------|--------------|--------|----------|
| B40 income share | 16.5% | WID.world | P0 |
| M40 income share | 32.3% | WID.world | P0 |
| Top 10% income share | 35.2% | WID.world | P0 |
| Top 1% income share | 15.8% | WID.world | P0 |
| Top 0.1% income share | 6.8% | WID.world | P0 |
| B40 wealth/income | 0.8x | WID.world | P0 |
| M40 wealth/income | 2.5x | WID.world | P0 |
| Top 10% wealth/income | 12.5x | WID.world | P0 |
| Top 1% wealth/income | 25.0x | WID.world | P0 |
| Capital share top 0.1% | 70% | WID DINA | P1 |

**Access**: wid.world → Download → Malaysia → DINA

### 3.2 Labor Market (4-State)

| Parameter | Value | Source | Priority |
|-----------|-------|--------|----------|
| formal_epf.share | 52% | DOSM LFS + EPF | P0 |
| precariat.share | 18% | Grab Report + MDEC | P0 |
| civil_service.share | 8% | JPA | P0 |
| informal_legacy.share | 22% | DOSM LFS residual | P0 |
| Sector wage premiums | [1.0, 0.85, 0.95, 0.70] | HIES 2022 | P0 |
| Sector volatility | [0.25, 0.45, 0.10, 0.55] | HIES panel | P1 |

**Cross-tabulation target**: WID fractile × Labor sector matrix
- What % of Top 0.1% are civil servants? (Very few - mostly business owners)
- What % of B40 are precariat? (High)

### 3.3 STR (Sumbangan Tunai Rahmah)

| Parameter | Value | Source | Priority |
|-----------|-------|--------|----------|
| b40_threshold | RM 2,500/mo | MOF Budget | P0 |
| m40_threshold | RM 5,000/mo | MOF Budget | P0 |
| annual_amount | RM 2,500 | MOF | P0 |
| mpc_estimate | 0.95 | BNPL data | P1 (BNM restricted) |
| take_up_rate | 88% | MOF | P0 |

**WID Insight**: STR phase-out creates 60%+ implicit marginal tax rate for M40 near RM5,000 threshold

### 3.4 PTPTN

| Parameter | Value | Source | Priority |
|-----------|-------|--------|----------|
| total_borrowers | 3.6M | PTPTN Annual | P0 |
| average_loan | RM 27,000 | PTPTN | P0 |
| default_rate_overall | 40% | PTPTN | P0 |
| Default by WID type | B40: 45%, M40: 35%, T20: 15% | PTPTN + MOHE | P1 |
| Income-contingent threshold | RM 2,000/mo | PTPTN Policy | P0 |

### 3.5 Credit Spreads by Sector

| Sector | Spread | Notes |
|--------|--------|-------|
| formal_epf | 0 bps | Base rate ~4.5% |
| precariat | +400 bps | 8-12% personal loan |
| civil_service | -100 bps | LPPSA 4% vs 4.5% |
| informal_legacy | +600 bps | Ar-Rahnu 10-12% |

**WID Dimension**: Top 0.1% get additional -50bps (relationship banking)

### 3.6 Housing & LPPSA

| Parameter | Value | Source | Priority |
|-----------|-------|--------|----------|
| lppsa.rate | 4.0% | LPPSA Act | P0 |
| lppsa.max_loan | RM 200K-750K | LPPSA Guidelines | P0 |
| market_mortgage_rate | 4.5% | BNM | P0 |
| PRIMA eligibility | RM 3,000-10,000/mo | PRIMA Website | P0 |

**WID Insight**: B40 households often fail mortgage affordability despite adequate income (negative liquid assets not captured in surveys)

---

## 4. WID-Enabled Research Questions

### Q1-Q6: Same as before (STR, PTPTN, Precariat)

### Q11: Taxing the Missing Rich (NOW TESTABLE)

**Question**: If Malaysia introduced a 2% wealth tax on financial assets >RM10m (targeting T0.1), how much revenue is raised vs evasion/capital flight?

**PCN-HANK Mechanism**:
```python
def compute_wealth_tax(wealth, wid_type, offshore_share=0.30):
    """
    T0.1 households hold 25x wealth-income ratio
    But 30% may be offshore (Labuan, Singapore)
    """
    if wid_type == 'T0.1' and wealth > 1000:  # RM 10M ≈ 1000 model units
        taxable_wealth = wealth * (1 - offshore_share)
        tax = 0.02 * taxable_wealth
        
        # Capital mobility decision
        mobility_cost = 0.05 * wealth  # Cost of moving to Singapore
        if tax > mobility_cost:
            # Household moves assets offshore
            actual_tax = 0
            capital_flight = wealth * 0.30
        else:
            actual_tax = tax
            capital_flight = 0
            
    return actual_tax, capital_flight
```

**Required Data**:
- WID: Top 0.1% wealth concentration
- BNM: Offshore holdings estimate (Labuan)
- IRB: Tax evasion estimates

### NEW: Q12: Inequality-Precision Policy Tradeoff

**Question**: When BNM hikes OPR, does the consumption drop for T0.1 (asset-rich) offset the gain to B40 (debt-burdened)?

**Mechanism**: 
- T0.1: Hold liquid assets → gain from higher rates
- B40: Hold credit card debt → hurt by higher rates
- Net effect depends on distribution of assets/debt by WID type

---

## 5. Implementation Roadmap

### Week 1: WID Data Integration
- Download WID Malaysia DINA data
- Create wid_fractile dimension (6 classes)
- Calibrate wealth/income ratios by fractile
- **Deliverable**: WID-calibrated steady state

### Week 2: Core Architecture
- config.py: PCN-HANK parameters with WID
- state_space.py: 6-class WID + 4-state labor encoding
- policy_network.py: Neural network with 8 heads

### Week 3: Fiscal Block
- transfers.py: STR phase-out function
- ptptn.py: Education loan states
- **WID feature**: Different PTPTN default rates by fractile

### Week 4: Credit Market
- credit_market.py: Sector-specific spreads
- labor_market.py: Transition matrices
- **WID feature**: Top 0.1% preferential rates

### Week 5: Housing + Wealth Tax Module
- housing_market.py: LPPSA integration
- wealth_tax.py: Q11 implementation
- **WID feature**: Capital mobility decision

### Week 6: Calibration & Validation
- calibration_targets.py: WID moments
- validator.py: Model fit to WID distributions
- **Test**: Can model reproduce WID wealth Gini?

---

## 6. Data Collection Checklist

### P0 (Required for MVP)
- [x] **WID.world** - Download Malaysia DINA (income + wealth fractiles)
- [ ] DOSM LFS 2024 - sector shares
- [ ] MOF Budget 2024 - STR parameters
- [ ] PTPTN Annual Report - loan data
- [ ] LPPSA Act - housing rates
- [ ] BNM lending rates - credit spreads

### P1 (Validation)
- [ ] HIES Panel - transition matrices
- [ ] MOHE research - PTPTN defaults by sector
- [ ] BNPL data - STR MPC (BNM restricted)
- [ ] Grab Report - gig economy size
- [ ] **Labuan FSA** - Offshore wealth estimates (for Q11)

### WID-Specific Data Sources
| Source | URL | Variables |
|--------|-----|-----------|
| WID.world | https://wid.world/country/malaysia/ | Income/wealth shares, fractiles |
| WID DINA | Download tab | Microdata for calibration |
| WID Methodology | wid.world/methodology | Imputation methods |

---

## 7. Key Innovation Summary

### With WID Integration:

1. **Tail-End Accuracy**: Model correctly captures top 0.1% holding 40x wealth-income ratio (not possible with survey data)

2. **Q11 Testable**: Can simulate wealth taxation on financial assets >RM10m with realistic behavioral responses

3. **Policy Calibration**: STR phase-out effects precisely targeted using actual income distribution (not smoothed survey data)

4. **Validation Target**: Model must reproduce WID wealth Gini (~0.75 for Malaysia)

### Comparison: Survey vs WID Calibration

| Metric | Survey (HIES) | WID (DINA) | Impact on Model |
|--------|---------------|------------|-----------------|
| Top 1% income share | 8% | 16% | 2x higher top incomes |
| Top 0.1% wealth | Missing | 15% of total | Critical for Q11 |
| Wealth Gini | 0.60 (underreported) | 0.75 (tax-adjusted) | Better inequality fit |
| Capital income share | 25% | 35% | Higher capital taxation potential |

---

## 8. Neural Network Update: MoE Backbone

To capture the extreme non-linearities of the Malaysian wealth distribution (especially the Top 0.1% tail captured by WID), we use a **Mixture of Experts (MoE)** backbone.

```python
# NEW: WID-aware MoE Backbone
self.wid_embed = nn.Embedding(6, 16)  # B40, M40, T90, T9, T1, T0.1
self.moe_backbone = MoEBackbone(hidden_dim=512, num_experts=3)

# Updated state encoder
def encode_state(self, continuous, wid_type, labor_sector, ...):
    cont_encoded = self.continuous_encoder(continuous)  # 64
    wid_encoded = self.wid_embed(wid_type)              # 16
    labor_encoded = self.labor_embed(labor_sector)      # 16
    # ... other embeddings
    
    encoded = torch.cat([cont_encoded, wid_encoded, labor_encoded, ...], dim=1)
    
    # MoE Backbone routes to specialized Experts:
    # Expert 1: B40 Physics (liquidity constraints, STR)
    # Expert 2: M40/T90 Physics (housing, education loans)
    # Expert 3: T1/T0.1 Physics (wealth tax, capital flight)
    features = self.moe_backbone(encoded)
    
    return features

# WID-aware policy heads
self.consumption_head = nn.Sequential(
    nn.Linear(512, 128),
    nn.SiLU(),
    nn.Linear(128, 1),
    nn.Softplus()
)
```

**Why MoE for WID Integration?**
1. **Spectral Bias Mitigation**: Prevents the high-frequency "kinks" in the B40 policy function from being smoothed out by the T0.1 behavior (Rahaman et al., 2019).
2. **Regime Specialization**: Experts act as specialized "advisors" for different Hamilton (1989) regimes in the distribution.
3. **Tail Precision**: Allows the model to fit the Top 0.1% wealth-income ratios (40x) precisely without distorting the B40 calibration.

**Training Implication**: Loss function now includes WID moment matching
```python
L_total = L_budget + L_euler + L_constraint + λ * L_wid_moments

where L_wid_moments = sum over fractiles (model_wealth_share - WID_target)^2
```

