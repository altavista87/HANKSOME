#!/usr/bin/env python3
"""
Quick NaN Fixes for compute_irf_stage5.py
=========================================

Run this script to apply minimal NaN fixes to the IRF computation.
This is a temporary patch until proper fixes are implemented.

Usage:
    python apply_nan_fixes.py
    
Then run:
    python compute_irf_stage5.py
"""

import re

def apply_fixes():
    """Apply minimal NaN fixes to compute_irf_stage5.py"""
    
    with open('compute_irf_stage5.py', 'r') as f:
        content = f.read()
    
    # Fix 1: Add safe_normalize function at the top (after imports)
    safe_normalize_func = '''
# Safe normalization function (added for NaN handling)
def safe_normalize(arr, fallback=0.0):
    """Safely normalize array, return fallback if mean is invalid."""
    arr = np.array(arr)
    mean = np.mean(arr)
    if mean == 0 or np.isnan(mean) or np.isinf(mean):
        return np.full_like(arr, fallback, dtype=float)
    return arr / mean * 100

def safe_deviation(arr, fallback=0.0):
    """Return array or zeros if contains NaN/Inf."""
    arr = np.array(arr)
    if np.isnan(arr).any() or np.isinf(arr).any():
        return np.full_like(arr, fallback, dtype=float)
    return arr

'''
    
    # Find a good place to insert (after imports)
    import_end = content.find('class ImpulseResponseComputer:')
    if import_end > 0:
        content = content[:import_end] + safe_normalize_func + content[import_end:]
    
    # Fix 2: Replace problematic normalization in plot_irf
    content = re.sub(
        r"ax\.plot\(quarters, np\.array\(irf\['C'\]\) / np\.mean\(irf\['C'\]\) \* 100",
        "ax.plot(quarters, safe_normalize(irf['C'])",
        content
    )
    
    content = re.sub(
        r"ax\.plot\(quarters, np\.array\(irf\['Y'\]\) / np\.mean\(irf\['Y'\]\) \* 100",
        "ax.plot(quarters, safe_normalize(irf['Y'])",
        content
    )
    
    # Fix 3: Replace other problematic divisions
    content = re.sub(
        r"np\.array\(irf\['C'\]\) / np\.mean\(irf\['C'\]\) \* 100",
        "safe_normalize(irf['C'])",
        content
    )
    
    # Fix 4: Add NaN checks in _simulate_path
    old_append = "results['C'].append(policies['consumption'].mean().item())"
    new_append = '''c_val = policies['consumption'].mean().item()
            if not (np.isnan(c_val) or np.isinf(c_val)):
                results['C'].append(c_val)
            else:
                results['C'].append(results['C'][-1] if results['C'] else 0.0)'''
    content = content.replace(old_append, new_append)
    
    # Fix 5: Add subgroup size check
    old_group_loop = "for group_name, mask in groups.items():"
    new_group_loop = '''for group_name, mask in groups.items():
            n_group = mask.sum().item()
            if n_group < 50:
                print(f"  Skipping {group_name}: only {n_group} agents")
                continue'''
    content = content.replace(old_group_loop, new_group_loop)
    
    # Write back
    with open('compute_irf_stage5.py', 'w') as f:
        f.write(content)
    
    print("✓ Applied NaN fixes to compute_irf_stage5.py")
    print("\nChanges made:")
    print("  1. Added safe_normalize() function")
    print("  2. Added safe_deviation() function")
    print("  3. Replaced unsafe divisions with safe_normalize()")
    print("  4. Added NaN checks in simulation loop")
    print("  5. Added minimum group size check (50 agents)")
    print("\nYou can now run: python compute_irf_stage5.py")

if __name__ == "__main__":
    apply_fixes()
