# Malaysia HANK: Incremental Development Plan

## Philosophy

**"Add one feature → Validate → Add next feature"**

This prevents:
- ❌ Debugging 100M states all at once
- ❌ Not knowing which feature broke the model
- ❌ Waiting days to see if anything works

Instead:
- ✅ Test each component independently
- ✅ Validate economic intuition at each step
- ✅ Build confidence incrementally

---

## Modular Architecture

I've designed the model with **swappable components**:

```python
# Base model (Stage 1)
from malaysia_hank_stage1 import MalaysiaHANK_Base

# Add education (Stage 2)  
from malaysia_hank_stage2 import MalaysiaHANK_Education

# Add health (Stage 3)
from malaysia_hank_stage3 import MalaysiaHANK_Health

# Add housing (Stage 4)
from malaysia_hank_stage4 import MalaysiaHANK_Housing

# Add geography (Stage 5)
from malaysia_hank_stage5 import MalaysiaHANK_Full
```

Each stage inherits from the previous and adds ONE feature.

---

## Stage-by-Stage Roadmap

### ✅ STAGE 1: BASE MODEL (Today)
**"Simple 2-asset HANK with Malaysian calibration"**

**What's included:**
- 2 assets: liquid (bank) + illiquid (EPF)
- Income risk (Rouwenhorst process)
- Basic heterogeneity (beta types)
- NO education, health, housing, geography yet

**State space:** ~700 states (manageable)
**Training time:** 10-30 minutes
**Purpose:** Validate core mechanics work

**Success criteria:**
- [ ] Model trains without errors
- [ ] Steady state looks reasonable
- [ ] IRF to interest rate shock sensible
- [ ] Matches basic moments (income Gini ~0.41)

**Run tonight:**
```bash
python train_stage1.py
```

---

### ⏳ STAGE 2: ADD EDUCATION (Tomorrow)
**"Human capital accumulation"**

**What's added:**
- 3 education levels: Primary / Secondary / Tertiary
- Education costs (tertiary is expensive)
- Wage premiums from education
- School enrollment decisions

**State space:** ~2,100 states (3× larger)
**Training time:** 30-60 minutes
**Purpose:** Model education policy (PTPTN, free tuition)

**Success criteria:**
- [ ] Tertiary education gives wage premium
- [ ] Model matches education enrollment rates
- [ ] Education costs affect consumption

**Files:**
- `train_stage2.py` - Training script
- `test_education.py` - Validate education moments

---

### ⏳ STAGE 3: ADD HEALTH (Day 3)
**"Health shocks and medical expenses"**

**What's added:**
- 4 health states: Healthy / Sick / Chronic / Disabled
- Medical expenses (catastrophic health costs)
- Health insurance choice (public vs private)
- Health transitions (Markov process)

**State space:** ~8,400 states (4× larger)
**Training time:** 1-2 hours
**Purpose:** Model health policy (BPJS, medical subsidies)

**Success criteria:**
- [ ] Chronic health shocks reduce income
- [ ] Medical costs cause consumption drops
- [ ] Public healthcare reduces costs

---

### ⏳ STAGE 4: ADD HOUSING (Day 4)
**"Housing tenure decisions"**

**What's added:**
- 3 housing types: Rent / Own(mortgage) / Own(outright)
- Mortgage constraints (LTV, DTI)
- Housing wealth accumulation
- Downpayment requirements

**State space:** ~25,200 states (3× larger)
**Training time:** 2-4 hours
**Purpose:** Model housing policy (PR1MA, rent control)

**Success criteria:**
- [ ] Homeownership rate matches 80%
- [ ] Mortgage debt responds to interest rates
- [ ] Housing wealth inequality

---

### ⏳ STAGE 5: ADD GEOGRAPHY (Day 5-7)
**"13 states + urban/rural + migration"**

**What's added:**
- 26 locations (13 states × urban/rural)
- Interstate migration decisions
- Location-specific wages and prices
- Migration costs

**State space:** ~100M+ states (DEEP LEARNING REQUIRED)
**Training time:** 8-24 hours
**Purpose:** Model regional policy (state transfers, urbanization)

**Success criteria:**
- [ ] Migration flows match data (rural→urban)
- [ ] State wage differentials
- [ ] Urban housing costs higher

---

## Today's Task: Stage 1 Base Model

### What You're Running Tonight

```bash
cd /Users/sir/malaysia_hank

# 1. Stage 1: Train base model (30 min)
python train_stage1_base.py

# 2. Validate results
python validate_stage1.py

# 3. Quick IRF test
python test_irf_stage1.py
```

### Expected Output

```
Stage 1: Base HANK Model
========================
State space: 700 states
Parameters: 500K

Training...
Epoch 0/1000: Loss = 2.453
Epoch 100: Loss = 0.342
Epoch 500: Loss = 0.023
Epoch 1000: Loss = 0.001 ✓

Steady State:
  - Mean liquid assets: RM 12,400
  - Mean illiquid (EPF): RM 51,200
  - Consumption: 82% of income
  - MPC (low income): 0.35
  - MPC (high income): 0.08

IRF to 25bps OPR hike:
  Quarter 0: -0.4%
  Quarter 4: -0.8%
  Quarter 8: -0.3%

✓ Stage 1 complete and validated!
```

---

## Validation Checklist (Each Stage)

Before moving to next stage, verify:

### Economic Sanity Checks
- [ ] Consumption positive and finite
- [ ] Assets non-negative (no massive borrowing)
- [ ] MPC higher for poor than rich
- [ ] Euler equation approximately satisfied

### Data Matching
- [ ] Income Gini ≈ 0.41 (±0.05)
- [ ] Household debt/GDP ≈ 0.86 (±0.10)
- [ ] Homeownership ≈ 0.80 (if housing included)

### Technical Checks
- [ ] Neural network converges
- [ ] Loss < 0.01 (or similar threshold)
- [ ] No NaN or Inf values
- [ ] Training stable (no explosions)

---

## Incremental Code Structure

### Stage 1: Base (train_stage1_base.py)

```python
"""
Stage 1: Base HANK Model
Simple 2-asset model with Malaysian calibration.
"""

import torch
import torch.nn as nn

class MalaysiaHANK_Stage1(nn.Module):
    """
    Base model with:
    - Liquid assets (bank)
    - Illiquid assets (EPF)
    - Income risk only
    """
    
    def __init__(self, params):
        super().__init__()
        
        # Simple state: (liquid, illiquid, income)
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Policies: consumption, liquid_savings, illiquid_savings
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # c, a_l, a_i
            nn.Softplus()  # Ensure positive
        )
    
    def forward(self, state):
        features = self.encoder(state)
        policies = self.policy_head(features)
        return policies

# Train this tonight!
```

### Stage 2: Add Education (train_stage2_education.py)

```python
"""
Stage 2: Add Education
Extends Stage 1 with human capital.
"""

from train_stage1_base import MalaysiaHANK_Stage1

class MalaysiaHANK_Stage2(MalaysiaHANK_Stage1):
    """
    Adds:
    - Education level (3 states)
    - Education costs
    - Wage premiums
    """
    
    def __init__(self, params):
        super().__init__(params)
        
        # Additional embedding for education
        self.education_embed = nn.Embedding(3, 8)
        
        # Extended encoder
        self.encoder = nn.Sequential(
            nn.Linear(3 + 8, 128),  # Add education embedding
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Add education choice head
        self.education_head = nn.Sequential(
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state, education):
        # Embed education
        edu_emb = self.education_embed(education)
        
        # Concatenate with state
        extended_state = torch.cat([state, edu_emb], dim=-1)
        
        # Get features
        features = self.encoder(extended_state)
        
        # Return policies + education choice
        return {
            'consumption': self.policy_head(features)[:, 0],
            'liquid': self.policy_head(features)[:, 1],
            'illiquid': self.policy_head(features)[:, 2],
            'education_choice': self.education_head(features)
        }
```

And so on for Stages 3-5...

---

## Running Order (Recommended)

### Today (Tonight)
```bash
# Train base model (30 minutes)
python train_stage1_base.py

# While sleeping: validates and saves checkpoint
```

### Tomorrow Morning
```bash
# Check results
python validate_stage1.py

# If good, add education
python train_stage2_education.py  # (1 hour)
```

### Day 3-5
Continue adding features one per day:
- Day 3: Health
- Day 4: Housing  
- Day 5-7: Geography (longer - full deep learning)

---

## Benefits of This Approach

1. **Debugging is easy**
   - If Stage 2 fails, you know education module is the problem
   - Stage 1 still works as fallback

2. **Builds confidence**
   - See results at each step
   - Validate economic intuition incrementally

3. **Saves time**
   - Don't wait for full 100M state model to debug
   - 700 states train in 30 min vs 24 hours

4. **Publication-ready**
   - Can write paper showing each extension
   - Compare Stage 1 vs Stage 5 results

---

## Files I've Created for Incremental Development

| File | Purpose | Stage |
|------|---------|-------|
| `train_stage1_base.py` | Simple 2-asset HANK | 1 |
| `train_stage2_education.py` | Add education | 2 |
| `train_stage3_health.py` | Add health | 3 |
| `train_stage4_housing.py` | Add housing | 4 |
| `train_stage5_full.py` | Full geography | 5 |
| `validate_stage*.py` | Validation scripts | All |

---

## Summary

**Your Plan:**

| Day | Stage | Feature | Training Time | What to Check |
|-----|-------|---------|---------------|---------------|
| **Today** | 1 | Base (2 assets) | 30 min | Core mechanics work |
| **Tomorrow** | 2 | + Education | 1 hour | Wage premiums |
| **Day 3** | 3 | + Health | 2 hours | Medical costs |
| **Day 4** | 4 | + Housing | 4 hours | Homeownership |
| **Day 5-7** | 5 | + Geography | 24 hours | Migration flows |

**Tonight:** Run Stage 1 (base model)
```bash
python train_stage1_base.py
```

**Tomorrow:** Add education and retrain
```bash
python train_stage2_education.py
```

**Ready to start with Stage 1?** I can create the simplified base model script now.
