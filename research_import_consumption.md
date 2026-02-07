# Research Notes: Import Consumption by Income in Malaysia

## Key Empirical Question
Do poor (Q1) households consume MORE or LESS imported goods as a share of their budget?

## Theoretical Arguments

### Argument A: Poor households consume MORE imports (my initial assumption)
- Higher food share → more rice, cooking oil, wheat (imported)
- Higher fuel share → petrol/diesel (imported/refined)
- Limited domestic substitutes
- Can't afford premium local products

### Argument B: Poor households consume FEWER imports (your intuition)
- Buy local/traditional goods (cheaper)
- Can't afford imported products
- Consume domestically produced staples (local rice, vegetables)
- Informal sector goods (local services)

## What We Need to Know

### For Malaysia specifically:

1. **Rice consumption**
   - B40 buys local rice (SST 5%, subsidized)
   - T20 buys imported rice (Thai, Japanese)
   - → Q1 may consume LESS imported rice

2. **Cooking oil**
   - Mostly palm oil (domestic production)
   - But packaging may be imported
   - → Relatively domestic

3. **Fuel**
   - Petrol/diesel subsidized
   - But crude oil imported
   - → Everyone faces similar import content

4. **Processed foods**
   - B40: Local brands, traditional foods
   - T20: International brands, imported specialties
   - → Q1 may consume LESS imports

5. **Services**
   - B40: Local services, informal sector
   - T20: International travel, imported services
   - → Q1 definitely consumes LESS imported services

## Data Sources to Check

### DOSM HIES 2022 (Household Income & Expenditure Survey)
- Has expenditure by category and income group
- Categories: Food at home, Food away, Transport, etc.
- But doesn't explicitly label "imported"

### Input-Output Tables (Malaysia)
- Shows import content by sector
- Can combine with HIES to estimate import intensity

### Academic Research
1. "Import intensity of household consumption in Malaysia"
2. "Who bears the burden of depreciation?" (distributional effects)
3. BNM working papers on exchange rate pass-through

## Hypothesis Refinement

### Revised hypothesis: U-shaped relationship
- **Q1 (Very poor)**: Low import share
  - Consume local staples, informal sector
  - Can't afford imported goods
  
- **Q2-Q3 (Lower-middle)**: HIGHEST import share
  - Moving to processed foods, branded goods
  - Starting to consume fuel more intensively
  
- **Q4-Q5 (Upper-middle, Rich)**: Lower import share for goods, BUT
  - High import share for services (travel, education)
  - Overall may be lower or similar

## Practical Approach for Model

Given uncertainty, we should:

1. **Use conservative estimates** from literature
2. **Allow sensitivity analysis** (try both scenarios)
3. **Focus on differential impact** rather than absolute levels

### Literature estimates from similar countries:

**Indonesia (similar structure)**:
- Poor: 15-20% import content
- Middle: 25-30% import content
- Rich: 20-25% import content (services offset goods)

**Thailand**:
- Similar pattern to Indonesia
- Poor households less exposed to imports

**Latin America studies**:
- Generally find poor households less exposed to import prices
- Except for fuel/food staples

## Recommendation

**Conservative calibration for Malaysia:**

```python
import_share_by_quintile = {
    'Q1': 0.25,  # Lower than aggregate (can't afford imports)
    'Q2': 0.30,  # Approaching average
    'Q3': 0.32,  # Slightly above average (consumption upgrading)
    'Q4': 0.28,  # Declining goods share
    'Q5': 0.22,  # High services share (mostly domestic)
}
```

**BUT with fuel/food adjustment:**
- Add 0.10 to all quintiles for fuel import content
- Q1 still pays world prices for fuel via subsidies

**Alternative: Flat with adjustment**
- All quintiles: 30% baseline
- Q1: +0.05 (higher fuel/food share)
- Q5: -0.05 (higher services share)

This needs empirical validation from HIES + Input-Output tables.
