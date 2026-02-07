# HANKSOME MoE v4 (Active Development)

**Previously known as:** `malaysia_hank`

This is the current, active version of the Malaysia Extended HANK Model using Deep Learning and Mixture of Experts (MoE).

## Key Features
- **100M+ State Space:** Handles Education, Health, Housing, and Geography.
- **Mixture of Experts:** B40, M40, T20 specialized networks.
- **Deep Learning Solver:** PyTorch-based physics-informed loss.

## Usage
1. **Train Model:** `python malaysia_deep_hank_architecture.py`
2. **Benchmark:** `python compare_to_bnm.py`

## Latest Changes (Jan 2026)
- Renamed from `malaysia_hank` to `HANKSOME_MoE_v4`.
- **Calibration Fix:** Lowered beta to 0.86 and increased Tertiary education returns to 5.0 to widen Consumption Ratio (Target: 3.5x).
