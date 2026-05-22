#!/usr/bin/env python3
"""
Lotto 6/45 Phase 3 (Interactions & Math)
- Phase 1 features: Frequency (last 10/30), Gap
- Phase 3 feature engineering:
  1. AC Value (Arithmetic Complexity) - per-number encoding
  2. Hot Pairs / Association - per-number pair strength
  3. Consecutive Numbers - per-number consecutive tendency
- Advanced model: XGBoost
- Combination-level scoring with AC, consecutive, hot-pair validation
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter, defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

# ── 0. Install XGBoost if needed ──────────────────────────────────────────
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not found. Installing...")
    os.system("pip install xgboost")
    from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

# ── 1. Load Data ──────────────────────────────────────────────────────────
DATA_PATH = "data/lotto.csv"
df = pd.read_csv(DATA_PATH)
df = df.sort_values("round").reset_index(drop=True)

draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values.tolist()
n_draws = len(draws)
print(f"Total draws loaded: {n_draws}")

# ── 2. Compute Historical Distributions (for combination scoring) ────────

# 2a. AC Value for each historical draw
def compute_ac_value(drawn_numbers):
    """AC Value = number of unique positive differences between all pairs - 5"""
    diffs = set()
    for i in range(len(drawn_numbers)):
        for j in range(i + 1, len(drawn_numbers)):
            diffs.add(abs(drawn_numbers[i] - drawn_numbers[j]))
    return len(diffs) - 5

all_ac_values = [compute_ac_value(d) for d in draws]
ac_mean = np.mean(all_ac_values)
ac_std = np.std(all_ac_values)
print(f"\n[Phase 3] AC Value distribution: mean={ac_mean:.2f}, std={ac_std:.2f}")

# 2b. Consecutive pairs in each historical draw
def count_consecutive_pairs(drawn_numbers):
    """Count how many consecutive pairs (e.g., 14,15) exist in sorted numbers"""
    s = sorted(drawn_numbers)
    cnt = 0
    for i in range(len(s) - 1):
        if s[i + 1] - s[i] == 1:
            cnt += 1
    return cnt

all_consec_counts = [count_consecutive_pairs(d) for d in draws]
consec_mean = np.mean(all_consec_counts)
consec_std = np.std(all_consec_counts)
print(f"[Phase 3] Consecutive pairs distribution: mean={consec_mean:.2f}, std={consec_std:.2f}")

# 2c. Hot Pairs: co-occurrence matrix (45 x 45)
pair_counts = defaultdict(int)  # (i,j) -> count where i<j
num_appearances = Counter()     # number -> count of draws it appeared in

for d in draws:
    for i in range(6):
        num_appearances[d[i]] += 1
        for j in range(i + 1, 6):
            a, b = min(d[i], d[j]), max(d[i], d[j])
            pair_counts[(a, b)] += 1

# Pair strength: for each pair, P(B|A) = co-occurrence / appearance(A)
pair_strength = {}
for (a, b), cnt in pair_counts.items():
    p_strength = cnt / num_appearances[a]  # P(b|a)
    pair_strength[(a, b)] = p_strength

print(f"[Phase 3] Hot Pairs computed: {len(pair_counts)} unique pairs")

# ── 3. Per-Number Feature Engineering ──────────────────────────────────
def compute_features_for_draw_phase3(draw_idx, draws):
    """
    For each number 1..45, compute features based on draws BEFORE draw_idx.
    Returns feature matrix X (45 x 8+).
    """
    history = draws[:draw_idx]
    n_history = len(history)

    # Last appearance index
    last_seen = {}
    for i, drawn in enumerate(history):
        for num in drawn:
            last_seen[num] = i

    def freq_last_n(num, n):
        return sum(1 for drawn in history[-n:] if num in drawn)

    # Per-number AC: average AC value of draws containing this number
    num_ac_sum = defaultdict(float)
    num_ac_count = defaultdict(int)
    for i, drawn in enumerate(history):
        ac = compute_ac_value(drawn)
        for num in drawn:
            num_ac_sum[num] += ac
            num_ac_count[num] += 1

    # Per-number consecutive tendency: how often does this number appear with a consecutive neighbor?
    num_consec_sum = defaultdict(int)
    num_consec_count = defaultdict(int)
    for i, drawn in enumerate(history):
        s = sorted(drawn)
        for idx, num in enumerate(s):
            has_consec = 0
            if idx > 0 and s[idx] - s[idx - 1] == 1:
                has_consec = 1
            if idx < len(s) - 1 and s[idx + 1] - s[idx] == 1:
                has_consec = 1
            num_consec_sum[num] += has_consec
            num_consec_count[num] += 1

    # Per-number hot pair strength: average max pair strength with any other number
    def max_pair_strength_for_num(num):
        max_strength = 0.0
        for other in range(1, 46):
            if other == num:
                continue
            a, b = min(num, other), max(num, other)
            if (a, b) in pair_strength:
                max_strength = max(max_strength, pair_strength[(a, b)])
        return max_strength

    # Per-number: average pair strength with all co-occurring numbers in history
    num_pair_avg = defaultdict(float)
    num_pair_cnt = defaultdict(int)
    for i, drawn in enumerate(history):
        for idx, num in enumerate(drawn):
            # Average pair strength with other numbers in this draw
            total_ps = 0.0
            other_count = 0
            for jdx, other in enumerate(drawn):
                if jdx == idx:
                    continue
                a, b = min(num, other), max(num, other)
                if (a, b) in pair_strength:
                    total_ps += pair_strength[(a, b)]
                    other_count += 1
            if other_count > 0:
                num_pair_avg[num] += total_ps / other_count
                num_pair_cnt[num] += 1

    features = []
    for num in range(1, 46):
        # Phase 1 features
        f10 = freq_last_n(num, 10)
        f30 = freq_last_n(num, 30)
        if num in last_seen:
            gap = (n_history - 1) - last_seen[num]
        else:
            gap = n_history

        # Phase 3 per-number features
        avg_ac = num_ac_sum.get(num, 0) / max(num_ac_count.get(num, 1), 1)
        consec_rate = num_consec_sum.get(num, 0) / max(num_consec_count.get(num, 1), 1)
        max_ps = max_pair_strength_for_num(num)
        avg_pair = num_pair_avg.get(num, 0) / max(num_pair_cnt.get(num, 1), 1)

        features.append([f10, f30, gap, avg_ac, consec_rate, max_ps, avg_pair])

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
    feats = compute_features_for_draw_phase3(idx, draws[:len(train_draws)])
    target = target_vector(train_draws[idx])
    X_train_list.append(feats)
    y_train_list.append(target)

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)

print(f"Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
print(f"Feature columns: [Freq10, Freq30, Gap, avg_AC, consec_rate, max_pair_strength, avg_pair_strength]")

# ── 5. Train XGBoost ────────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    use_label_encoder=False,
)
xgb.fit(X_train, y_train)
print("XGBoost trained.")

# ── 6. Combination-Level Scoring Functions ──────────────────────────────

def score_combo_phase3(combo):
    """
    Score a 6-number combination based on AC value proximity, consecutive count,
    and hot pair strength. Returns a multiplier (higher = better).
    """
    s = sorted(combo)

    # AC value score (Gaussian proximity)
    ac = compute_ac_value(s)
    ac_z = abs(ac - ac_mean) / max(ac_std, 0.001)
    ac_score = max(0, 1.0 - ac_z / 3.0)

    # Consecutive count score
    consec = count_consecutive_pairs(s)
    consec_z = abs(consec - consec_mean) / max(consec_std, 0.001)
    consec_score = max(0, 1.0 - consec_z / 3.0)

    # Hot pair score: average pair strength among all pairs in this combo
    pair_scores = []
    for i in range(6):
        for j in range(i + 1, 6):
            a, b = min(s[i], s[j]), max(s[i], s[j])
            if (a, b) in pair_strength:
                pair_scores.append(pair_strength[(a, b)])
    avg_pair_score = np.mean(pair_scores) if pair_scores else 0.0
    # Normalize: each pair has theoretical max ~1.0 (always together)
    pair_norm = min(avg_pair_score * 3, 1.0)  # scale up since most are 0.1-0.3

    # Combined combo score
    combo_score = 0.4 * ac_score + 0.3 * consec_score + 0.3 * pair_norm
    return combo_score


def select_best_combo_phase3(probs, top_n_min=10, top_n_max=20):
    """
    1. From top probability numbers, generate all 6-number combos
    2. Score each combo: probability_weighted + combo_level_score
    3. Return best combo
    """
    candidates = set()
    for top_n in range(top_n_min, top_n_max + 1, 2):
        top_indices = np.argsort(probs)[-top_n:][::-1]
        top_numbers = [int(i + 1) for i in top_indices]
        combos = list(combinations(top_numbers, 6))
        for c in combos:
            candidates.add(c)

    best_total = -1
    best_combo = None
    for combo in candidates:
        prob_sum = sum(probs[n - 1] for n in combo)
        combo_quality = score_combo_phase3(combo)
        # Blend: combination must have decent probability AND good combo qualities
        total = prob_sum * (0.5 + 0.5 * combo_quality)
        if total > best_total:
            best_total = total
            best_combo = combo

    return sorted(best_combo)


# ── 7. Backtest on Test Set ──────────────────────────────────────────────
total_hits_p3 = 0
total_draws_evaluated = 0
last_train_idx = len(train_draws)
test_features_for_next = compute_features_for_draw_phase3(last_train_idx, draws)

print(f"\n{'='*60}")
print("Phase 3 Backtest Results (Interaction Features + XGBoost)")
print(f"{'='*60}")

for i, actual_drawn in enumerate(test_draws):
    feats = compute_features_for_draw_phase3(last_train_idx + i, draws)
    probs = xgb.predict_proba(feats)[:, 1]

    # Phase 3 combo selection
    combo = select_best_combo_phase3(probs)
    actual_set = set(actual_drawn)
    hits = sum(1 for n in combo if n in actual_set)
    total_hits_p3 += hits
    total_draws_evaluated += 1

    print(f"  Draw {last_train_idx + i + 1}: Pred {combo} | Actual {sorted(actual_drawn)} | Hits {hits}")

avg_hit_rate_p3 = total_hits_p3 / total_draws_evaluated
print(f"\n{'='*60}")
print(f"Phase 3 - Average Hit Rate (Top-6): {avg_hit_rate_p3:.4f} (out of 6)")
print(f"Total matches: {total_hits_p3} across {total_draws_evaluated} test draws")

# ── 8. Phase 1 (retrained with XGBoost on Phase 1 features) & Phase 3 ──
# Phase 1 comparison using same model on same test set (raw top-6 probs)
total_hits_p1 = 0
for i, actual_drawn in enumerate(test_draws):
    feats = compute_features_for_draw_phase3(last_train_idx + i, draws)
    probs = xgb.predict_proba(feats)[:, 1]
    top6_idx = np.argsort(probs)[-6:][::-1]
    top6_numbers = sorted([int(x + 1) for x in top6_idx])
    actual_set = set(actual_drawn)
    hits = sum(1 for n in top6_numbers if n in actual_set)
    total_hits_p1 += hits

avg_hit_rate_p1 = total_hits_p1 / len(test_draws)

print(f"\n{'='*60}")
print("COMPARISON: Phase 1 (XGB Raw) vs Phase 3 (XGB + Interaction Combo)")
print(f"{'='*60}")
print(f"  Phase 1 (RF Raw Top-6 from earlier run): 0.8000 hits/draw")
print(f"  Phase 1 (XGB Raw Top-6):                 {avg_hit_rate_p1:.4f} hits/draw ({total_hits_p1} total)")
print(f"  Phase 3 (XGB + Interaction Combo):       {avg_hit_rate_p3:.4f} hits/draw ({total_hits_p3} total)")

# Use Phase 1 RF result as the canonical baseline
baseline = 0.8000
improvement_vs_baseline = avg_hit_rate_p3 - baseline
print(f"\n  Improvement vs Phase 1 baseline (0.8000): {improvement_vs_baseline:+.4f} hits/draw")
if avg_hit_rate_p3 > baseline:
    print(f"  ✅ Phase 3 BEATS Phase 1 baseline!")
else:
    print(f"  ⚠️  Phase 3 does not beat Phase 1 baseline yet.")

# ── 9. Predict Next Draw ─────────────────────────────────────────────────
next_probs = xgb.predict_proba(test_features_for_next)[:, 1]
next_combo = select_best_combo_phase3(next_probs)
next_raw_top6 = sorted([int(x + 1) for x in np.argsort(next_probs)[-6:][::-1]])

print(f"\n{'='*60}")
print(f"Prediction for next draw (Draw #{n_draws + 1}):")
print(f"  Phase 1 (Raw Top-6):          {next_raw_top6}")
print(f"  Phase 3 (Interaction Combo):  {next_combo}")
print(f"\n  Phase 3 recommended numbers: {next_combo}")
print(f"    AC Value: {compute_ac_value(next_combo)} (target: {ac_mean:.1f})")
print(f"    Consecutive pairs: {count_consecutive_pairs(next_combo)} (target: {consec_mean:.1f})")

# ── 10. Log to experiment_log.tsv ────────────────────────────────────────
LOG_FILE = "experiment_log.tsv"
log_entry = {
    "phase": 3,
    "description": "Phase 3 - XGBoost + Interaction Features (AC, Hot Pairs, Consecutive)",
    "train_draws": len(train_draws),
    "test_draws": len(test_draws),
    "features": "Freq10, Freq30, Gap, avg_AC, consec_rate, max_pair_strength, avg_pair_strength + combo AC/pair/consec scoring",
    "model": "XGBClassifier(n_estimators=300, max_depth=6) + combo selection",
    "avg_hit_rate": f"{avg_hit_rate_p3:.4f}",
    "total_matches": total_hits_p3,
    "next_prediction": ",".join(map(str, next_combo)),
}

header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

print("\nPhase 3 complete.")