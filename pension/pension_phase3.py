#!/usr/bin/env python3
"""
PENSION Phase 3 (Markov Chain Analysis)
- Transition probability matrices for each of the 7 targets
- Chi-Square goodness-of-fit test against uniform distribution
- Next draw prediction using transition from Draw #315
"""

import pandas as pd
import numpy as np
from scipy.stats import chisquare
import os

# ── 1. Load Data ──────────────────────────────────────────────────────────
DATA_PATH = "data/pension.csv"
df = pd.read_csv(DATA_PATH)
df = df.sort_values("round").reset_index(drop=True)

n_draws = len(df)
print(f"Total pension draws loaded: {n_draws}")
print(f"Data range: Draw 1 to Draw {n_draws}")

# ── 2. Define Targets ─────────────────────────────────────────────────---
targets_info = {
    "group": {"values": df["group"].values, "n_states": 5, "state_range": (1, 5)},
    "n1": {"values": df["n1"].values, "n_states": 10, "state_range": (0, 9)},
    "n2": {"values": df["n2"].values, "n_states": 10, "state_range": (0, 9)},
    "n3": {"values": df["n3"].values, "n_states": 10, "state_range": (0, 9)},
    "n4": {"values": df["n4"].values, "n_states": 10, "state_range": (0, 9)},
    "n5": {"values": df["n5"].values, "n_states": 10, "state_range": (0, 9)},
    "n6": {"values": df["n6"].values, "n_states": 10, "state_range": (0, 9)},
}

results = []  # Store results for logging

print(f"\n{'='*60}")
print("MARKOV CHAIN ANALYSIS")
print(f"{'='*60}")

# ── 3. Build Transition Matrices & Chi-Square Tests ──────────────────────
next_prediction = {}

for target_name, info in targets_info.items():
    values = info["values"]
    n_states = info["n_states"]
    min_val, max_val = info["state_range"]
    
    # Build transition count matrix
    trans_matrix = np.zeros((n_states, n_states), dtype=int)
    
    for t in range(1, n_draws):
        prev_state = values[t - 1] - min_val  # shift to 0-indexed
        curr_state = values[t] - min_val
        trans_matrix[prev_state, curr_state] += 1
    
    # Total transitions observed
    total_transitions = n_draws - 1
    
    # Expected counts under uniform: for each row, expected = row_sum / n_states
    row_sums = trans_matrix.sum(axis=1)
    
    # Chi-Square test per row
    chi2_total = 0.0
    dof_total = 0
    row_results = []
    
    print(f"\n  --- {target_name} (n_states={n_states}) ---")
    
    for row_idx in range(n_states):
        observed = trans_matrix[row_idx]
        expected_uniform = np.full(n_states, row_sums[row_idx] / n_states)
        
        # Only test rows with at least 5 expected per cell (Chi-Square validity)
        if row_sums[row_idx] > 0 and np.all(expected_uniform >= 1):
            chi2, p_value = chisquare(observed, f_exp=expected_uniform)
            chi2_total += chi2
            dof_total += (n_states - 1)
            
            is_significant = "***" if p_value < 0.05 else ""
            row_results.append({
                "row": row_idx + min_val,
                "count": row_sums[row_idx],
                "chi2": chi2,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "expected": expected_uniform,
                "observed": observed,
            })
            
            if p_value < 0.05:
                print(f"    From state={row_idx + min_val}: χ²={chi2:.4f}, p={p_value:.4f} {is_significant}")
        else:
            if row_sums[row_idx] > 0:
                print(f"    From state={row_idx + min_val}: count={row_sums[row_idx]} (too few for test)")
    
    # Overall Chi-Square (pooled across all rows)
    mean_p = np.mean([r["p_value"] for r in row_results]) if row_results else 1.0
    min_p = min([r["p_value"] for r in row_results]) if row_results else 1.0
    sig_rows = sum(1 for r in row_results if r["significant"])
    
    result = {
        "target": target_name,
        "n_states": n_states,
        "total_transitions": total_transitions,
        "mean_p": mean_p,
        "min_p": min_p,
        "significant_rows": sig_rows,
        "total_rows_tested": len(row_results),
        "chi2_total": chi2_total,
        "dof_total": dof_total,
    }
    
    if len(row_results) > 0:
        # Pooled p-value from summed chi-square
        from scipy.stats import chi2 as chi2_dist
        pooled_p = 1.0 - chi2_dist.cdf(chi2_total, dof_total)
        result["pooled_p"] = pooled_p
        print(f"    --- Pooled: χ²={chi2_total:.4f}, df={dof_total}, p={pooled_p:.6f}" + 
              (" ***" if pooled_p < 0.05 else ""))
        print(f"    Significant rows: {sig_rows}/{len(row_results)} (p<0.05)")
    else:
        result["pooled_p"] = 1.0
    
    results.append(result)
    
    # ── 4. Predict Draw #316 based on Draw #315 ──
    last_state = values[-1] - min_val  # 0-indexed
    last_state_value = values[-1]
    
    # Get transition probabilities from last state
    if row_sums[last_state] > 0:
        probs = trans_matrix[last_state] / row_sums[last_state]
    else:
        probs = np.ones(n_states) / n_states  # uniform if no history
    
    # Most likely next state
    most_likely = np.argmax(probs) + min_val
    next_prediction[target_name] = int(most_likely)
    
    print(f"    Draw #{n_draws} ({target_name}) = {last_state_value}")
    print(f"    Transitions from state {last_state_value}: {dict(enumerate(trans_matrix[last_state], start=min_val))}")
    print(f"    Probabilities: ", end="")
    top3 = np.argsort(probs)[-3:][::-1]
    for i, idx in enumerate(top3):
        print(f"{idx + min_val}={probs[idx]:.3f}", end="; " if i < 2 else "")
    print()
    print(f"    >>> Next draw prediction: {most_likely}")
    print(f"    (Uniform would give probability: {1/n_states:.4f} vs predicted: {probs[most_likely]:.4f})")

# ── 5. Summary Table ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY: Chi-Square Tests vs Uniform Distribution")
print(f"{'='*60}")
print(f"{'Target':<10} {'States':<8} {'Pooled p':<12} {'Sig Rows':<10} {'Uniform?':<12}")
print("-" * 60)

all_pooled_p = []
for r in results:
    is_uniform = "YES" if r["pooled_p"] >= 0.05 else "NO ***"
    print(f"{r['target']:<10} {r['n_states']:<8} {r['pooled_p']:<12.6f} {r['significant_rows']}/{r['total_rows_tested']:<9} {is_uniform}")
    all_pooled_p.append(r["pooled_p"])

print("-" * 60)
# Count how many targets have memory (p < 0.05 → not uniform → has memory)
memory_count = sum(1 for p in all_pooled_p if p < 0.05)
print(f"Targets with significant memory (p<0.05): {memory_count}/{len(results)}")

# ── 6. Final Prediction for Draw #316 ────────────────────────────────────
print(f"\n{'='*60}")
print(f"PREDICTION FOR DRAW #316 (using Markov Chain)")
print(f"{'='*60}")
print(f"  Based on Draw #315 values:")
print(f"    group={df['group'].values[-1]}, digits=[{df['n1'].values[-1]}, {df['n2'].values[-1]}, "
      f"{df['n3'].values[-1]}, {df['n4'].values[-1]}, {df['n5'].values[-1]}, {df['n6'].values[-1]}]")

print(f"\n  Markov Chain Prediction for Draw #316:")
print(f"    Class:  {next_prediction['group']}")
print(f"    Digits: [{next_prediction['n1']}, {next_prediction['n2']}, {next_prediction['n3']}, "
      f"{next_prediction['n4']}, {next_prediction['n5']}, {next_prediction['n6']}]")

# ── 7. Log to pension_log.tsv ────────────────────────────────────────────
LOG_FILE = "pension/pension_log.tsv"
pooled_p_str = ";".join([f"{r['target']}={r['pooled_p']:.4f}" for r in results])
sig_summary = f"{memory_count}/{len(results)} targets have memory (p<0.05)"
log_entry = {
    "phase": 3,
    "description": "Pension Phase 3 - Markov Chain + Chi-Square Test",
    "targets": "7 targets (group, n1-n6)",
    "method": f"Transition Matrix (5x5 group, 10x10 digits) + Chi-Square vs uniform, alpha=0.05",
    "pooled_p_values": pooled_p_str,
    "sig_summary": sig_summary,
    "next_prediction": f"group={next_prediction['group']},n1={next_prediction['n1']},n2={next_prediction['n2']},n3={next_prediction['n3']},n4={next_prediction['n4']},n5={next_prediction['n5']},n6={next_prediction['n6']}",
}
header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

# ── 8. Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PENSION PHASE 3 COMPLETE - SUMMARY")
print(f"{'='*60}")
print(f"  Data: Draws 1 to {n_draws} ({n_draws - 1} transitions)")
print(f"  Method: Markov Chain Transition Matrix + Chi-Square Test")
print(f"  Null Hypothesis: Transitions follow a uniform distribution (no memory)")
print(f"  Alpha: 0.05")
print(f"\n  Key Finding:")
if memory_count > 0:
    print(f"    ✅ {memory_count}/{len(results)} targets show statistically significant memory (p<0.05)")
    print(f"    → The drawing machine(s) may have physical bias/memory")
else:
    print(f"    ❌ No targets show statistically significant memory")
    print(f"    → Transitions appear consistent with uniform random draws")
print(f"\n  Draw #316 Markov Prediction:")
print(f"    Class: {next_prediction['group']}")
print(f"    Digits: [{next_prediction['n1']}, {next_prediction['n2']}, {next_prediction['n3']}, "
      f"{next_prediction['n4']}, {next_prediction['n5']}, {next_prediction['n6']}]")
print("\nPension Phase 3 complete.")