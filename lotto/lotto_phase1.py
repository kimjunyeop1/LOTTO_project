#!/usr/bin/env python3
"""
Lotto 6/45 Phase 1 (Baseline)
- Basic features: Frequency (last 10/30 draws), Gap
- Random Forest classifier per number (1-45)
- Hold out last 50 draws as test set
- Top-6 selection per draw
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import os

# ── 1. Load Data ──────────────────────────────────────────────────────────
DATA_PATH = "data/lotto.csv"
df = pd.read_csv(DATA_PATH)

# Ensure sorted by round ascending
df = df.sort_values("round").reset_index(drop=True)

# Extract drawn numbers (6 main numbers per draw)
draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values.tolist()  # list of list[int]
n_draws = len(draws)
print(f"Total draws loaded: {n_draws}")

# ── 2. Feature Engineering ───────────────────────────────────────────────
def compute_features_for_draw(draw_idx, draws, window_10=10, window_30=30):
    """
    For a given draw index, compute features for each number 1..45
    based on draws BEFORE draw_idx (i.e. historical data only, no look-ahead).
    Returns feature matrix X (45 x 3).
    """
    # History up to (but not including) draw_idx
    history = draws[:draw_idx]

    # Last appearance index for each number
    last_seen = {}  # number -> last draw index (relative to history)
    for i, drawn in enumerate(history):
        for num in drawn:
            last_seen[num] = i

    # Count appearances in last N draws
    def freq_last_n(history, num, n):
        return sum(1 for drawn in history[-n:] if num in drawn)

    features = []
    for num in range(1, 46):
        f10 = freq_last_n(history, num, window_10)
        f30 = freq_last_n(history, num, window_30)
        # Gap: if never seen, set to large value (e.g., len(history) + 1)
        if num in last_seen:
            gap = (len(history) - 1) - last_seen[num]
        else:
            gap = len(history)  # never appeared
        features.append([f10, f30, gap])

    return np.array(features)


def target_vector(drawn_numbers):
    """Return binary vector length 45: 1 if number was drawn, else 0."""
    vec = np.zeros(45, dtype=int)
    for num in drawn_numbers:
        vec[num - 1] = 1
    return vec


# ── 3. Train / Test Split ────────────────────────────────────────────────
TEST_SIZE = 50
train_draws = draws[:-TEST_SIZE]
test_draws = draws[-TEST_SIZE:]

print(f"Train draws: {len(train_draws)}, Test draws: {len(test_draws)}")

# Build training samples: for each historical draw index (starting from min_history),
# compute features and target.
MIN_HISTORY = 30  # need at least 30 draws of history for features

X_train_list = []
y_train_list = []

for idx in range(MIN_HISTORY, len(train_draws)):
    feats = compute_features_for_draw(idx, draws[: len(train_draws)])
    target = target_vector(train_draws[idx])
    X_train_list.append(feats)   # (45, 3)
    y_train_list.append(target)  # (45,)

# Stack into (samples * 45, 3) and (samples * 45,)
X_train = np.vstack(X_train_list)      # shape: (N_train*45, 3)
y_train = np.concatenate(y_train_list) # shape: (N_train*45,)

print(f"Training samples (per-number): {X_train.shape[0]}, features: {X_train.shape[1]}")

# ── 4. Train Random Forest ───────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Random Forest trained.")

# ── 5. Backtest on Test Set ──────────────────────────────────────────────
total_hits = 0
total_draws_evaluated = 0

# For next-draw prediction (after last available draw)
last_train_idx = len(train_draws)
test_features_for_next = compute_features_for_draw(last_train_idx, draws)

all_predictions = []

for i, actual_drawn in enumerate(test_draws):
    # Feature vector for this test draw (based on all prior draws)
    feats = compute_features_for_draw(last_train_idx + i, draws)
    probs = rf.predict_proba(feats)[:, 1]  # probability of class 1 for each number
    # Top-6 selection
    top6_idx = np.argsort(probs)[-6:][::-1]  # indices 0..44
    top6_numbers = sorted([int(x + 1) for x in top6_idx])

    # Count matches
    actual_set = set(actual_drawn)
    hits = sum(1 for n in top6_numbers if n in actual_set)
    total_hits += hits
    total_draws_evaluated += 1

    all_predictions.append((last_train_idx + i + 1, top6_numbers, actual_drawn, hits))
    print(f"  Draw {last_train_idx + i + 1}: Pred {top6_numbers} | Actual {sorted(actual_drawn)} | Hits {hits}")

avg_hit_rate = total_hits / total_draws_evaluated
print(f"\n{'='*60}")
print(f"Phase 1 - Average Hit Rate (Top-6): {avg_hit_rate:.4f} (out of 6)")
print(f"Total matches: {total_hits} across {total_draws_evaluated} test draws")

# ── 6. Predict Next Draw ─────────────────────────────────────────────────
next_probs = rf.predict_proba(test_features_for_next)[:, 1]
next_top6_idx = np.argsort(next_probs)[-6:][::-1]
next_top6_numbers = sorted([int(x + 1) for x in next_top6_idx])
next_probs_sorted = [next_probs[idx] for idx in next_top6_idx]

print(f"\n{'='*60}")
print(f"Prediction for next draw (Draw #{n_draws + 1}):")
for num, prob in zip(next_top6_numbers, next_probs_sorted):
    print(f"  Number {num:2d}: probability {prob:.4f}")
print(f"Recommended numbers: {next_top6_numbers}")

# ── 7. Log to experiment_log.tsv ────────────────────────────────────────
LOG_FILE = "experiment_log.tsv"
log_entry = {
    "phase": 1,
    "description": "Phase 1 Baseline - Random Forest (Freq10, Freq30, Gap)",
    "train_draws": len(train_draws),
    "test_draws": len(test_draws),
    "features": "Frequency_last_10, Frequency_last_30, Gap",
    "model": "RandomForestClassifier(n_estimators=200, max_depth=10)",
    "avg_hit_rate": f"{avg_hit_rate:.4f}",
    "total_matches": total_hits,
    "next_prediction": ",".join(map(str, next_top6_numbers)),
}

header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

print("\nPhase 1 complete.")