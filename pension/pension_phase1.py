#!/usr/bin/env python3
"""
PENSION Phase 1 (Multi-Output Classification)
- 7 independent XGBoost classifiers (1 for Class, 6 for Digits)
- Time-series features: Rolling Frequencies, Gap stats, Momentum
- 5-fold TimeSeriesSplit validation
- Strict zero data leakage
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

# ── 1. Load Data ──────────────────────────────────────────────────────────
DATA_PATH = "data/pension.csv"
df = pd.read_csv(DATA_PATH)
df = df.sort_values("round").reset_index(drop=True)

n_draws = len(df)
print(f"Total pension draws loaded: {n_draws}")

# Extract targets
targets_info = {
    "group": {"values": df["group"].values - 1, "n_classes": 5, "range": (0, 4)},  # remap 1-5 to 0-4
    "n1": {"values": df["n1"].values, "n_classes": 10, "range": (0, 9)},
    "n2": {"values": df["n2"].values, "n_classes": 10, "range": (0, 9)},
    "n3": {"values": df["n3"].values, "n_classes": 10, "range": (0, 9)},
    "n4": {"values": df["n4"].values, "n_classes": 10, "range": (0, 9)},
    "n5": {"values": df["n5"].values, "n_classes": 10, "range": (0, 9)},
    "n6": {"values": df["n6"].values, "n_classes": 10, "range": (0, 9)},
}

# ── 2. Feature Engineering Function ──────────────────────────────────────
def compute_features_for_target(draw_idx, target_values, target_min, target_max):
    """
    For a specific target (group or one digit), compute per-class features
    based ONLY on draws BEFORE draw_idx.
    
    target_values: list of target values for this target across all draws
    target_min, target_max: range of values
    
    Returns: (n_classes, n_features) feature matrix
    """
    history = target_values[:draw_idx]
    n_history = len(history)
    n_classes = target_max - target_min + 1
    
    # Track appearances for each class
    class_appearances = defaultdict(list)  # class_value -> [draw_indices]
    class_appeared_at = defaultdict(set)
    for i, val in enumerate(history):
        class_appearances[val].append(i)
        class_appeared_at[val].add(i)
    
    features = []
    for cls in range(target_min, target_max + 1):
        # ── Rolling Frequencies ──
        def rolling_freq(N):
            if n_history == 0:
                return 0
            start = max(0, n_history - N)
            cnt = 0
            appeared = class_appeared_at.get(cls, set())
            for idx in range(start, n_history):
                if idx in appeared:
                    cnt += 1
            return cnt
        
        f5 = rolling_freq(5)
        f10 = rolling_freq(10)
        f15 = rolling_freq(15)
        f20 = rolling_freq(20)
        f30 = rolling_freq(30)
        
        # ── Gap Statistics ──
        appear_list = sorted(class_appearances.get(cls, []))
        
        current_gap = (n_history - 1) - appear_list[-1] if appear_list else n_history
        
        if len(appear_list) >= 2:
            previous_gap = appear_list[-1] - appear_list[-2]
        elif len(appear_list) == 1:
            previous_gap = appear_list[0]
        else:
            previous_gap = n_history
        
        if len(appear_list) >= 2:
            hist_avg_gap = np.mean([appear_list[i+1] - appear_list[i] for i in range(len(appear_list)-1)])
        elif len(appear_list) == 1:
            hist_avg_gap = appear_list[0]
        else:
            hist_avg_gap = n_history
        
        # ── Momentum ──
        mom_10_30 = f10 / f30 if f30 > 0 else 0.0
        mom_10_20 = f10 / f20 if f20 > 0 else 0.0
        mom_15_30 = f15 / f30 if f30 > 0 else 0.0
        
        # ── Recency Score ──
        recency_score = 0.0
        if appear_list:
            for appear_idx in appear_list:
                recency_score += np.exp(-(n_history - 1 - appear_idx) / 10.0)
        recency_score /= max(n_history, 1)
        
        features.append([f5, f10, f15, f20, f30, current_gap, previous_gap,
                         hist_avg_gap, mom_10_30, mom_10_20, mom_15_30, recency_score])
    
    return np.array(features)

N_FEATURES = 12

# ── 3. Pre-compute features for all targets at all draws ─────────────────
print("Pre-computing features for all targets...")
all_features_cache = {}  # target_name -> np.array (n_draws, n_classes, n_features)

for target_name, info in targets_info.items():
    print(f"  Computing features for '{target_name}'...")
    target_feats = []
    for draw_idx in range(n_draws):
        feats = compute_features_for_target(
            draw_idx, info["values"], info["range"][0], info["range"][1]
        )
        target_feats.append(feats)
    all_features_cache[target_name] = np.array(target_feats)
    print(f"    Shape: {all_features_cache[target_name].shape}")

# ── 4. TSCV Evaluation ───────────────────────────────────────────────────
MIN_HISTORY = 30
all_draw_indices = list(range(MIN_HISTORY, n_draws))
N_FOLDS = 5
tscv = TimeSeriesSplit(n_splits=N_FOLDS)

print(f"\n{'='*60}")
print("PENSION Phase 1 - 5-fold TSCV Evaluation")
print(f"{'='*60}")

# Store results per target
target_cv_results = {name: [] for name in targets_info}

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(all_draw_indices)):
    fold_train_draws = [all_draw_indices[i] for i in train_idx]
    fold_val_draws = [all_draw_indices[i] for i in val_idx]
    
    print(f"\n  Fold {fold_idx + 1}: Train={len(fold_train_draws)}, Val={len(fold_val_draws)}")
    
    for target_name, info in targets_info.items():
        n_classes = info["n_classes"]
        target_min, target_max = info["range"]
        
    # Training data: flatten per-class features into a single vector per draw
        X_train_list = [all_features_cache[target_name][idx].ravel() for idx in fold_train_draws]
        y_train = [info["values"][idx] for idx in fold_train_draws]
        X_train = np.vstack(X_train_list)
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold_idx,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
            objective='multi:softprob',
            num_class=n_classes,
        )
        model.fit(X_train, y_train)
        
        # Validation
        val_correct = 0
        val_total = 0
        for val_draw_idx in fold_val_draws:
            X_val = all_features_cache[target_name][val_draw_idx].ravel().reshape(1, -1)
            pred = model.predict(X_val)[0]
            actual = info["values"][val_draw_idx]
            if pred == actual:
                val_correct += 1
            val_total += 1
        
        accuracy = val_correct / val_total
        target_cv_results[target_name].append(accuracy)
        print(f"    {target_name}: accuracy={accuracy:.4f} ({val_correct}/{val_total})")

# ── 5. CV Summary ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CROSS-VALIDATION RESULTS (5-fold TSCV)")
print(f"{'='*60}")
print(f"{'Target':<10} {'Mean Acc':<10} {'Std Acc':<10} {'Per-fold':<30}")
print("-" * 60)

all_targets_mean = []
for target_name in targets_info:
    mean_acc = np.mean(target_cv_results[target_name])
    std_acc = np.std(target_cv_results[target_name])
    per_fold = " ".join([f"{v:.3f}" for v in target_cv_results[target_name]])
    all_targets_mean.append(mean_acc)
    print(f"{target_name:<10} {mean_acc:<10.4f} {std_acc:<10.4f} {per_fold}")

overall_mean = np.mean(all_targets_mean)
print("-" * 60)
print(f"{'AVERAGE':<10} {overall_mean:<10.4f}")

# ── 6. Train Final Models on FULL Data ───────────────────────────────────
print(f"\n{'='*60}")
print("Training Final Models on Full Training Set")
print(f"{'='*60}")

final_models = {}
TEST_SIZE = 30
train_indices = all_draw_indices[:-TEST_SIZE]
test_indices = all_draw_indices[-TEST_SIZE:]

for target_name, info in targets_info.items():
    n_classes = info["n_classes"]
    X_train_list = [all_features_cache[target_name][idx].ravel() for idx in train_indices]
    y_train = [info["values"][idx] for idx in train_indices]
    X_train = np.vstack(X_train_list)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0,
        objective='multi:softprob',
        num_class=n_classes,
    )
    model.fit(X_train, y_train)
    final_models[target_name] = model
    print(f"  {target_name}: trained (C={n_classes})")

# ── 7. Evaluate on Held-Out Test Set ─────────────────────────────────────
print(f"\n{'='*60}")
print("Held-Out Test Set Evaluation (Last 30 draws)")
print(f"{'='*60}")

test_results = {}
for target_name in targets_info:
    correct = 0
    total = len(test_indices)
    for test_draw_idx in test_indices:
        X_test = all_features_cache[target_name][test_draw_idx].ravel().reshape(1, -1)
        pred = final_models[target_name].predict(X_test)[0]
        actual = info["values"][test_draw_idx]
        if pred == actual:
            correct += 1
    test_results[target_name] = correct / total
    # For group, remap prediction back to 1-5 for display
    if target_name == "group":
        display_correct = sum(1 for idx in test_indices if (final_models["group"].predict(
            all_features_cache["group"][idx].ravel().reshape(1, -1))[0] + 1) == df["group"].values[idx])
        display_acc = display_correct / total
        print(f"  {target_name}: test accuracy={display_acc:.4f} ({display_correct}/{total})")
    else:
        print(f"  {target_name}: test accuracy={correct/total:.4f} ({correct}/{total})")

# ── 8. Predict Next Draw ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Prediction for next draw (Draw #{n_draws + 1})")
print(f"{'='*60}")

next_predictions = {}
for target_name, info in targets_info.items():
    # Features for next draw use all available draws
    X_next = compute_features_for_target(
        n_draws, info["values"], info["range"][0], info["range"][1]
    ).ravel().reshape(1, -1)
    pred = final_models[target_name].predict(X_next)[0]
    if target_name == "group":
        next_predictions[target_name] = int(pred) + 1  # remap back to 1-5
    else:
        next_predictions[target_name] = int(pred)

print(f"\n  Predicted Class:  {next_predictions['group']}")
print(f"  Predicted Digits: [{next_predictions['n1']}, {next_predictions['n2']}, "
      f"{next_predictions['n3']}, {next_predictions['n4']}, {next_predictions['n5']}, "
      f"{next_predictions['n6']}]")

# Show probabilities for next draw
print(f"\n  Per-target top-3 probabilities:")
for target_name in targets_info:
    info = targets_info[target_name]
    X_next = compute_features_for_target(
        n_draws, info["values"], info["range"][0], info["range"][1]
    ).ravel().reshape(1, -1)
    probs = final_models[target_name].predict_proba(X_next)[0]
    top3 = np.argsort(probs)[-3:][::-1]
    offset = info["range"][0]
    probs_str = " | ".join([f"val={int(offset + i)}: prob={probs[i]:.4f}" for i in top3])
    print(f"    {target_name}: {probs_str}")

# ── 9. Log to pension_log.tsv ────────────────────────────────────────────
LOG_FILE = "pension/pension_log.tsv"
log_entry = {
    "phase": 1,
    "description": "Pension Phase 1 - 7 independent XGBoost classifiers",
    "targets": ",".join(targets_info.keys()),
    "features": "Freq5,Freq10,Freq15,Freq20,Freq30,Gap,PrevGap,HistAvgGap,Mom_10_30,Mom_10_20,Mom_15_30,Recency",
    "model": "XGBClassifier(n_estimators=200, max_depth=6, lr=0.1) per target",
    "cv_mean_accuracy": f"{overall_mean:.4f}",
    "test_mean_accuracy": f"{np.mean(list(test_results.values())):.4f}",
    "next_prediction": f"{next_predictions['group']},{next_predictions['n1']},{next_predictions['n2']},{next_predictions['n3']},{next_predictions['n4']},{next_predictions['n5']},{next_predictions['n6']}",
}

header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

# ── 10. Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PENSION PHASE 1 COMPLETE - SUMMARY")
print(f"{'='*60}")
print(f"  Targets: 7 independent (1 Class + 6 Digits)")
print(f"  Model:   XGBoost per target (n_estimators=200, max_depth=6)")
print(f"  CV:      5-fold TimeSeriesSplit")
print(f"\n  Per-target CV Accuracy:")
for target_name in targets_info:
    mean_acc = np.mean(target_cv_results[target_name])
    print(f"    {target_name}: {mean_acc:.4f}")
print(f"  Average CV: {overall_mean:.4f}")
print(f"\n  Per-target Test Accuracy:")
for target_name in targets_info:
    print(f"    {target_name}: {test_results[target_name]:.4f}")
print(f"  Average Test: {np.mean(list(test_results.values())):.4f}")
print(f"\n  Next draw prediction:")
print(f"    Class: {next_predictions['group']}")
print(f"    Numbers: [{next_predictions['n1']}, {next_predictions['n2']}, "
      f"{next_predictions['n3']}, {next_predictions['n4']}, {next_predictions['n5']}, "
      f"{next_predictions['n6']}]")
print("\nPension Phase 1 complete.")