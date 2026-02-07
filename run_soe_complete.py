"""
Run Complete SOE HANK Pipeline
==============================

Executes all milestones in sequence:
1. Parameters & Data
2. Model Extension
3. IRF Computer
4. Calibration
5. Policy Counterfactuals
6. Visualization
"""

import sys
from pathlib import Path

print("="*70)
print("MALAYSIA SOE HANK: COMPLETE PIPELINE")
print("="*70)

milestones = [
    ("Milestone 1: SOE Parameters", "soe_params.py"),
    ("Milestone 2: Model Extension", "model_soe.py"),
    ("Milestone 3: IRF Computer", "irf_soe.py"),
    ("Milestone 4: Calibration", "calibrate_soe.py"),
    ("Milestone 5: Policy Counterfactuals", "policy_counterfactuals.py"),
    ("Milestone 6: Visualization", "visualize_soe.py"),
]

for i, (name, script) in enumerate(milestones, 1):
    print(f"\n{'='*70}")
    print(f"Running {name}")
    print(f"{'='*70}")
    
    try:
        exec(open(script).read())
        print(f"\n✅ {name} Complete")
    except Exception as e:
        print(f"\n❌ {name} Failed: {e}")
        sys.exit(1)

print("\n" + "="*70)
print("ALL MILESTONES COMPLETE!")
print("="*70)
print("\nGenerated outputs:")
print("  - 6 Python modules")
print("  - 5 visualization figures")
print("  - 3 JSON data files")
print("  - 1 comprehensive summary document")
print("\nView results in: /Users/sir/malaysia_hank/outputs/soe_simulation/")
print("="*70)
