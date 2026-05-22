#!/usr/bin/env python3
"""
Lotto 6/45 Phase 2 (Macro Patterns)
- Phase 1 features: Frequency (last 10/30), Gap
- Phase 2 macro features: Sum of Numbers, Odd/Even Ratio (combination-level filtering)
- Strategy: Generate top-N candidate combinations, score + filter by macro patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from collections import Counter
import os

# ── 1. Load Data ──────────────────────────────────────────────────────────
DATA_PATH = "data/lotto.csv"
df = pd.read_csv(DATA_PATH)
df = df.sort_values("round").reset_index(drop=True)

draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values.tolist()
n_draws = len(draws)
print(f"Total draws loaded: {n_draws}")

# ── 2. Compute Macro Pattern Distributions from ALL historical data ──────
all_sums = [sum(d) for d in draws]
all_odd_counts = [sum(1 for n in d if n % 2 == 1) for d in draws]

sum_mean = np.mean(all_sums)
sum_std = np.std(all_sums)
print(f"\n[Macro] Sum distribution: mean={sum_mean:.1f}, std={sum_std:.1f}")
print(f"[Macro] Sum range (95%): [{sum_mean - 2*sum_std:.0f}, {sum_mean + 2*sum_std:.0f}]")

odd_even_dist = Counter(all_odd_counts)
print(f"[Macro] Odd/Even distribution (odd count out of 6):")
for odd_c in range(7):
    pct = odd_even_dist[odd_c] / len(all_odd_counts) * 100
    print(f"         {odd_c} odd / {6 - odd_c} even: {odd_even_dist[odd_c]} draws ({pct:.1f}%)")

# ── 3. Feature Engineering (Phase 1 features) ────────────────────────────
def compute_features_for_draw(draw_idx, draws, window_10=10, window_30=30):
    history = draws[:draw_idx]
    last_seen = {}
    for i, drawn in enumerate(history):
        for num in drawn:
            last_seen[num] = i

    def freq_last_n(history, num, n):
        return sum(1 for drawn in history[-n:] if num in drawn)

    features = []
    for num in range(1, 46):
        f10 = freq_last_n(history, num, window_10)
        f30 = freq_last_n(history, num, window_30)
        if num in last_seen:
            gap = (len(history) - 1) - last_seen[num]
        else:
            gap = len(history)
        features.append([f10, f30, gap])
    return np.array(features)

def target_vector(drawn_numbers):
    vec = np.zeros(45, dtype=int)
    for num in drawn_numbers:
        vec[num - 1] = 1
    return vec

# ── 4. Train / Test Split ────────────────────────────────────────────────
TEST_SIZE = 50
train_draws = draws[:-TEST_SIZE]
test_draws = draws[-TEST_SIZE:]

print(f"\nTrain draws: {len(train_draws)}, Test draws: {len(test_draws)}")

MIN_HISTORY = 30
X_train_list, y_train_list = [], []

for idx in range(MIN_HISTORY, len(train_draws)):
    feats = compute_features_for_draw(idx, draws[:len(train_draws)])
    target = target_vector(train_draws[idx])
    X_train_list.append(feats)
    y_train_list.append(target)

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)

print(f"Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")

# ── 5. Train Random Forest ───────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Random Forest trained.")

# ── 6. Phase 2: Macro-Aware Combination Selection ───────────────────────
def generate_top_combinations(probs, top_n=12, max_combos=5000):
    """
    From the top `top_n` numbers by probability, generate all 6-number combos.
    Returns at most `max_combos` combinations (sorted by probability product).
    """
    # Indices of top_n numbers (0..44)
    top_indices = np.argsort(probs)[-top_n:][::-1]
    top_numbers = [int(i + 1) for i in top_indices]

    # Generate all C(top_n, 6) combinations
    all_combos = list(combinations(top_numbers, 6))

    # Score each combo by sum of individual probabilities
    combo_scores = []
    for combo in all_combos:
        score = sum(probs[n - 1] for n in combo)
        combo_scores.append((combo, score))

    # Sort by score descending, keep top max_combos
    combo_scores.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in combo_scores[:max_combos]]


def macro_filter_score(combo):
    """
    Score a 6-number combination based on macro pattern fitness.
    Higher is better.
    Returns a score multiplier (≥ 0).
    """
    s = sum(combo)
    odd_count = sum(1 for n in combo if n % 2 == 1)
    even_count = 6 - odd_count

    # Sum fitness: Gaussian score centered at mean
    sum_z = abs(s - sum_mean) / sum_std
    sum_score = max(0, 1.0 - sum_z / 3.0)  # decays to 0 at ~3 sigma

    # Odd/Even fitness: based on historical frequency
    # Most common: 3 odd/3 even, then 4/2, 2/4, 5/1, 1/5, 6/0, 0/6
    oe_freq = odd_even_dist.get(odd_count, 0) / len(all_odd_counts)
    oe_score = oe_freq * 6  # scale so common patterns get ~1.0+

    # Combined macro score (weighted)
    macro_score = 0.5 * sum_score + 0.5 * min(oe_score, 1.0)
    return macro_score


def select_best_combo(probs, population_size=500):
    """
    Select the best 6-number combination using:
    1. Generate candidate combos from top-N probability numbers
    2. Score each combo: probability sum × macro_filter_score
    """
    # Try multiple top_n values to explore the space
    candidates = set()
    for top_n in [10, 12, 14, 16, 18]:
        combos = generate_top_combinations(probs, top_n=top_n, max_combos=2000)
        for c in combos:
            candidates.add(c)

    # Score all unique candidates
    best_score = -1
    best_combo = None
    for combo in candidates:
        prob_score = sum(probs[n - 1] for n in combo)
        macro = macro_filter_score(combo)
        total_score = prob_score * (0.6 + 0.4 * macro)  # blend
        if total_score > best_score:
            best_score = total_score
            best_combo = combo

    return sorted(best_combo)


# ── 7. Backtest on Test Set ──────────────────────────────────────────────
total_hits_phase2 = 0
total_draws_evaluated = 0
last_train_idx = len(train_draws)
test_features_for_next = compute_features_for_draw(last_train_idx, draws)

print(f"\n{'='*60}")
print("Phase 2 Backtest Results (Macro-Aware Selection)")
print(f"{'='*60}")

for i, actual_drawn in enumerate(test_draws):
    feats = compute_features_for_draw(last_train_idx + i, draws)
    probs = rf.predict_proba(feats)[:, 1]

    # Phase 2: macro-aware combination selection
    combo = select_best_combo(probs)
    actual_set = set(actual_drawn)
    hits = sum(1 for n in combo if n in actual_set)
    total_hits_phase2 += hits
    total_draws_evaluated += 1

    print(f"  Draw {last_train_idx + i + 1}: Pred {combo} | Actual {sorted(actual_drawn)} | Hits {hits}")

avg_hit_rate_p2 = total_hits_phase2 / total_draws_evaluated
print(f"\n{'='*60}")
print(f"Phase 2 - Average Hit Rate (Top-6): {avg_hit_rate_p2:.4f} (out of 6)")
print(f"Total matches: {total_hits_phase2} across {total_draws_evaluated} test draws")

# ── 8. Phase 1 Comparison ────────────────────────────────────────────────
# Re-run Phase 1 logic for comparison (top-6 by raw probability)
total_hits_p1 = 0
for i, actual_drawn in enumerate(test_draws):
    feats = compute_features_for_draw(last_train_idx + i, draws)
    probs = rf.predict_proba(feats)[:, 1]
    top6_idx = np.argsort(probs)[-6:][::-1]
    top6_numbers = sorted([int(x + 1) for x in top6_idx])
    actual_set = set(actual_drawn)
    hits = sum(1 for n in top6_numbers if n in actual_set)
    total_hits_p1 += hits

avg_hit_rate_p1 = total_hits_p1 / len(test_draws)

print(f"\n{'='*60}")
print("COMPARISON: Phase 1 vs Phase 2")
print(f"{'='*60}")
print(f"  Phase 1 (Raw Top-6):          {avg_hit_rate_p1:.4f} hits/draw ({total_hits_p1} total)")
print(f"  Phase 2 (Macro-Aware Combo):  {avg_hit_rate_p2:.4f} hits/draw ({total_hits_phase2} total)")
improvement = avg_hit_rate_p2 - avg_hit_rate_p1
print(f"  Improvement:                  {improvement:+.4f} hits/draw")
if avg_hit_rate_p2 > avg_hit_rate_p1:
    print(f"  ✅ Phase 2 BEATS Phase 1 baseline!")
else:
    print(f"  ⚠️  Phase 2 does not beat Phase 1 yet.")

# ── 9. Predict Next Draw ─────────────────────────────────────────────────
next_probs = rf.predict_proba(test_features_for_next)[:, 1]
next_combo = select_best_combo(next_probs)
next_raw_top6 = sorted([int(x + 1) for x in np.argsort(next_probs)[-6:][::-1]])

print(f"\n{'='*60}")
print(f"Prediction for next draw (Draw #{n_draws + 1}):")
print(f"  Phase 1 (Raw Top-6):         {next_raw_top6}")
print(f"  Phase 2 (Macro-Aware Combo): {next_combo}")
print(f"\n  Phase 2 recommended numbers: {next_combo}")
print(f"    Sum: {sum(next_combo)} (target range: {sum_mean - 2*sum_std:.0f}~{sum_mean + 2*sum_std:.0f})")
odd_c = sum(1 for n in next_combo if n % 2 == 1)
print(f"    Odd/Even: {odd_c} odd / {6 - odd_c} even")

# ── 10. Log to experiment_log.tsv ────────────────────────────────────────
LOG_FILE = "experiment_log.tsv"
log_entry = {
    "phase": 2,
    "description": "Phase 2 - Random Forest + Macro Patterns (Sum, Odd/Even filtering)",
    "train_draws": len(train_draws),
    "test_draws": len(test_draws),
    "features": "Freq10, Freq30, Gap + Macro sum/odd-even combo filter",
    "model": "RandomForestClassifier(n_estimators=200, max_depth=10) + macro selection",
    "avg_hit_rate": f"{avg_hit_rate_p2:.4f}",
    "total_matches": total_hits_phase2,
    "next_prediction": ",".join(map(str, next_combo)),
}

header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

print("\nPhase 2 complete.")