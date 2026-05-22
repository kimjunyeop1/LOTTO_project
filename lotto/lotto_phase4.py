#!/usr/bin/env python3
"""
Lotto 6/45 Phase 4 (Rich Time-Series & Momentum)
- Rich per-number features with NO data leakage, NO post-processing
- Features: Rolling Frequencies (5,10,15,30,50), Gap Statistics, Momentum
- XGBoost with hyperparameter tuning
- Raw Top-6 selection only
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# ── 1. Load Data ──────────────────────────────────────────────────────────
DATA_PATH = "data/lotto.csv"
df = pd.read_csv(DATA_PATH)
df = df.sort_values("round").reset_index(drop=True)

draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values.tolist()
n_draws = len(draws)
print(f"Total draws loaded: {n_draws}")

# ── 2. Feature Engineering ───────────────────────────────────────────────
def compute_features_for_draw_phase4(draw_idx, draws):
    """
    Per-number features based ONLY on draws BEFORE draw_idx (strict temporal ordering).
    Returns 45 x N_features matrix.
    """
    history = draws[:draw_idx]
    n_history = len(history)

    # Build number appearance timeline: for each number, list of draw indices (within history) it appeared
    num_appearances = defaultdict(list)  # num -> [draw_idx_relative]
    for i, drawn in enumerate(history):
        for num in drawn:
            num_appearances[num].append(i)

    # Precompute appearance flags grid: numbers x draws (bool)
    # For efficiency, build a dict for fast lookups
    appeared_at = defaultdict(set)
    for i, drawn in enumerate(history):
        for num in drawn:
            appeared_at[num].add(i)

    features = []
    for num in range(1, 46):
        # ── Rolling Frequencies: last N draws ──
        def rolling_freq(N):
            if n_history == 0:
                return 0
            start = max(0, n_history - N)
            end = n_history
            cnt = 0
            appeared = appeared_at.get(num, set())
            for idx in range(start, end):
                if idx in appeared:
                    cnt += 1
            return cnt

        f5 = rolling_freq(5)
        f10 = rolling_freq(10)
        f15 = rolling_freq(15)
        f30 = rolling_freq(30)
        f50 = rolling_freq(50)

        # ── Gap Statistics ──
        appear_list = sorted(num_appearances.get(num, []))

        if len(appear_list) > 0:
            current_gap = (n_history - 1) - appear_list[-1]  # draws since last appearance
        else:
            current_gap = n_history  # never appeared

        if len(appear_list) >= 2:
            previous_gap = appear_list[-1] - appear_list[-2]  # gap between 2nd-last and last
        elif len(appear_list) == 1:
            # Only appeared once: previous gap = time from start to that appearance
            previous_gap = appear_list[0]
        else:
            previous_gap = n_history  # never appeared

        if len(appear_list) >= 2:
            historical_avg_gap = np.mean([appear_list[i+1] - appear_list[i] for i in range(len(appear_list)-1)])
        elif len(appear_list) == 1:
            historical_avg_gap = appear_list[0]  # first appearance at this index
        else:
            historical_avg_gap = n_history  # never appeared

        # ── Momentum: short-term freq / long-term freq ──
        # Ratio of freq in last 10 to freq in last 50 (avoid div by zero)
        if f50 > 0:
            momentum_10_50 = f10 / f50
        else:
            momentum_10_50 = 0.0

        if f30 > 0:
            momentum_10_30 = f10 / f30
        else:
            momentum_10_30 = 0.0

        if f50 > 0:
            momentum_15_50 = f15 / f50
        else:
            momentum_15_50 = 0.0

        # ── Recent appearance recency score ──
        # Higher weight for more recent appearances
        recency_score = 0.0
        if len(appear_list) > 0:
            for appear_idx in appear_list:
                # Weight: exponential decay - recent appearances get higher weight
                recency_score += np.exp(-(n_history - 1 - appear_idx) / 10.0)
        # Normalize by expected value if appeared in every draw (very rare)
        recency_score = recency_score / max(n_history, 1)

        features.append([
            f5, f10, f15, f30, f50,
            current_gap, previous_gap, historical_avg_gap,
            momentum_10_50, momentum_10_30, momentum_15_50,
            recency_score,
        ])

    return np.array(features)

def target_vector(drawn_numbers):
    vec = np.zeros(45, dtype=int)
    for num in drawn_numbers:
        vec[num - 1] = 1
    return vec

# ── 3. Train / Test Split ────────────────────────────────────────────────
TEST_SIZE = 50
train_draws = draws[:-TEST_SIZE]
test_draws = draws[-TEST_SIZE:]

print(f"\nTrain draws: {len(train_draws)}, Test draws: {len(test_draws)}")

MIN_HISTORY = 50  # Need at least 50 draws for rolling features
X_train_list, y_train_list = [], []

for idx in range(MIN_HISTORY, len(train_draws)):
    feats = compute_features_for_draw_phase4(idx, draws[:len(train_draws)])
    target = target_vector(train_draws[idx])
    X_train_list.append(feats)
    y_train_list.append(target)

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)

print(f"Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
feature_names = [
    "Freq5", "Freq10", "Freq15", "Freq30", "Freq50",
    "CurrentGap", "PrevGap", "HistAvgGap",
    "Mom_10_50", "Mom_10_30", "Mom_15_50",
    "RecencyScore"
]
print(f"Feature columns: {feature_names}")

# ── 4. Hyperparameter Tuning (light grid) ────────────────────────────────
print("\n--- Hyperparameter Tuning ---")
best_score = -1
best_params = None
best_model = None

# Define a small grid
param_grid = [
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
    {"n_estimators": 400, "max_depth": 7, "learning_rate": 0.08, "subsample": 0.8},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.15, "subsample": 0.7},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
    {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.9},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.2,  "subsample": 0.75},
]

# Use last 10% of training for validation
val_cut = int(len(X_train_list) * 0.9)
# X_train_list is list of 45x12 arrays; val_cut refers to number of draws
# Reconstruct with proper split
X_train_draws_list = X_train_list
y_train_draws_list = y_train_list

# Validate on last 10% of training draws (chronological validation)
val_start = int(len(X_train_draws_list) * 0.9)

X_val_list = X_train_draws_list[val_start:]
y_val_list = y_train_draws_list[val_start:]

X_train_tune_list = X_train_draws_list[:val_start]
y_train_tune_list = y_train_draws_list[:val_start]

if len(X_val_list) > 0:
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)
    X_train_tune = np.vstack(X_train_tune_list)
    y_train_tune = np.concatenate(y_train_tune_list)

    for params in param_grid:
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
        )
        model.fit(X_train_tune, y_train_tune)

        # Validate: per-draw hit rate
        val_hits = 0
        for v_idx in range(len(X_val_list)):
            probs = model.predict_proba(X_val_list[v_idx])[:, 1]
            top6_idx = np.argsort(probs)[-6:][::-1]
            top6 = set(int(x + 1) for x in top6_idx)
            actual = set(train_draws[val_start + v_idx])
            val_hits += sum(1 for n in top6 if n in actual)

        val_rate = val_hits / len(X_val_list)
        print(f"  params={params}: val_hit_rate={val_rate:.4f}")

        if val_rate > best_score:
            best_score = val_rate
            best_params = params
            best_model = model

    print(f"\nBest validation params: {best_params} (hit rate: {best_score:.4f})")
else:
    # Fallback: use default params
    best_params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8}
    print("Validation set too small, using default params.")

# ── 5. Train Final Model on Full Training Set ────────────────────────────
if best_model is None:
    # If tuning didn't happen, train fresh
    best_model = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0,
    )
    best_model.fit(X_train, y_train)

print("Phase 4 XGBoost trained.")

# ── 6. Backtest on Test Set (Phase 4) ─────────────────────────────────────
total_hits_p4 = 0
total_draws_evaluated = 0
last_train_idx = len(train_draws)

# Pre-compute features for next draw
test_features_for_next = compute_features_for_draw_phase4(last_train_idx, draws)

print(f"\n{'='*60}")
print("Phase 4 Backtest Results (Rich Time-Series + XGBoost, Raw Top-6)")
print(f"{'='*60}")

for i, actual_drawn in enumerate(test_draws):
    feats = compute_features_for_draw_phase4(last_train_idx + i, draws)
    probs = best_model.predict_proba(feats)[:, 1]

    # Phase 4: RAW TOP-6 selection (NO post-processing constraints!)
    top6_idx = np.argsort(probs)[-6:][::-1]
    top6_numbers = sorted([int(x + 1) for x in top6_idx])

    actual_set = set(actual_drawn)
    hits = sum(1 for n in top6_numbers if n in actual_set)
    total_hits_p4 += hits
    total_draws_evaluated += 1

    print(f"  Draw {last_train_idx + i + 1}: Pred {top6_numbers} | Actual {sorted(actual_drawn)} | Hits {hits}")

avg_hit_rate_p4 = total_hits_p4 / total_draws_evaluated
print(f"\n{'='*60}")
print(f"Phase 4 - Average Hit Rate (Raw Top-6): {avg_hit_rate_p4:.4f} (out of 6)")
print(f"Total matches: {total_hits_p4} across {total_draws_evaluated} test draws")

# ── 7. Phase 1 XGB comparison (same model but with Phase 1 features) ────
# Phase 1 features: Freq10, Freq30, Gap only
def compute_features_phase1_simple(draw_idx, draws):
    """Pure Phase 1 features: Freq10, Freq30, Gap"""
    history = draws[:draw_idx]
    n_history = len(history)
    last_seen = {}
    for i, drawn in enumerate(history):
        for num in drawn:
            last_seen[num] = i
    def freq_last_n(num, n):
        return sum(1 for drawn in history[-n:] if num in drawn)
    features = []
    for num in range(1, 46):
        f10 = freq_last_n(num, 10)
        f30 = freq_last_n(num, 30)
        if num in last_seen:
            gap = (n_history - 1) - last_seen[num]
        else:
            gap = n_history
        features.append([f10, f30, gap])
    return np.array(features)

# Train XGBoost on Phase 1 features only (for fair comparison)
X_train_p1_list, y_train_p1_list = [], []
for idx in range(MIN_HISTORY, len(train_draws)):
    feats = compute_features_phase1_simple(idx, draws[:len(train_draws)])
    target = target_vector(train_draws[idx])
    X_train_p1_list.append(feats)
    y_train_p1_list.append(target)

X_train_p1 = np.vstack(X_train_p1_list)
y_train_p1 = np.concatenate(y_train_p1_list)

xgb_p1 = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, eval_metric='logloss',
    use_label_encoder=False, verbosity=0,
)
xgb_p1.fit(X_train_p1, y_train_p1)

total_hits_p1_xgb = 0
for i, actual_drawn in enumerate(test_draws):
    feats = compute_features_phase1_simple(last_train_idx + i, draws)
    probs = xgb_p1.predict_proba(feats)[:, 1]
    top6_idx = np.argsort(probs)[-6:][::-1]
    top6_numbers = sorted([int(x + 1) for x in top6_idx])
    actual_set = set(actual_drawn)
    hits = sum(1 for n in top6_numbers if n in actual_set)
    total_hits_p1_xgb += hits

avg_hit_rate_p1_xgb = total_hits_p1_xgb / len(test_draws)

print(f"\n{'='*60}")
print("COMPARISON: Phase 1 XGB Simple vs Phase 4 XGB Rich Features")
print(f"{'='*60}")
print(f"  Phase 1 (XGB + 3 features):           {avg_hit_rate_p1_xgb:.4f} hits/draw ({total_hits_p1_xgb} total)")
print(f"  Phase 4 (XGB + 12 rich features):     {avg_hit_rate_p4:.4f} hits/draw ({total_hits_p4} total)")
improvement = avg_hit_rate_p4 - avg_hit_rate_p1_xgb
print(f"  Improvement:                          {improvement:+.4f} hits/draw")
if avg_hit_rate_p4 > avg_hit_rate_p1_xgb:
    print(f"  ✅ Phase 4 BEATS Phase 1 XGB baseline!")
elif avg_hit_rate_p4 == avg_hit_rate_p1_xgb:
    print(f"  ⚖️  Phase 4 ties Phase 1 XGB baseline.")
else:
    print(f"  ⚠️  Phase 4 does not beat Phase 1 XGB baseline yet.")

# Phase 1 RF reference from Phase 1 run
print(f"\n  Ref: Phase 1 (RF + 3 features earlier): 0.8000 hits/draw")

# ── 8. Predict Next Draw ─────────────────────────────────────────────────
next_probs = best_model.predict_proba(test_features_for_next)[:, 1]
next_top6 = sorted([int(x + 1) for x in np.argsort(next_probs)[-6:][::-1]])

# Also get Phase 1 XGB prediction (need Phase 1 features, not Phase 4 features)
next_feats_p1 = compute_features_phase1_simple(last_train_idx, draws)
next_probs_p1 = xgb_p1.predict_proba(next_feats_p1)[:, 1]
next_top6_p1 = sorted([int(x + 1) for x in np.argsort(next_probs_p1)[-6:][::-1]])

print(f"\n{'='*60}")
print(f"Prediction for next draw (Draw #{n_draws + 1}):")
print(f"  Phase 1 (XGB Simple):   {next_top6_p1}")
print(f"  Phase 4 (XGB Rich):     {next_top6}")
print(f"\n  >>> Phase 4 recommended numbers: {next_top6}")

# Show feature info for next draw prediction
print(f"\n  Per-number probabilities for Phase 4:")
next_feat_df = pd.DataFrame(test_features_for_next, columns=feature_names)
for rank, idx in enumerate(np.argsort(next_probs)[-6:][::-1]):
    num = idx + 1
    prob = next_probs[idx]
    feat_row = next_feat_df.iloc[idx]
    print(f"    #{rank+1}: Num {num:2d} | prob={prob:.4f} | Freq5={feat_row['Freq5']:.0f} Freq10={feat_row['Freq10']:.0f} Freq30={feat_row['Freq30']:.0f} Gap={feat_row['CurrentGap']:.0f} Mom={feat_row['Mom_10_50']:.2f}")

# ── 9. Log to experiment_log.tsv ────────────────────────────────────────
LOG_FILE = "experiment_log.tsv"
log_entry = {
    "phase": 4,
    "description": "Phase 4 - XGBoost + Rich Time-Series (Rolling Freq 5/10/15/30/50, Gap stats, Momentum)",
    "train_draws": len(train_draws),
    "test_draws": len(test_draws),
    "features": "Freq5,Freq10,Freq15,Freq30,Freq50,CurrentGap,PrevGap,HistAvgGap,Mom_10_50,Mom_10_30,Mom_15_50,RecencyScore",
    "model": f"XGBoost(n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, lr={best_params['learning_rate']})",
    "avg_hit_rate": f"{avg_hit_rate_p4:.4f}",
    "total_matches": total_hits_p4,
    "next_prediction": ",".join(map(str, next_top6)),
}

header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

print("\nPhase 4 complete.")