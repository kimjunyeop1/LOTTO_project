#!/usr/bin/env python3
"""
PENSION Phase 2 (Macro Properties)
- Predict Sum_Digits, Odd_Count, High_Count of the 6 digits (n1-n6)
- XGBRegressor for Sum, XGBClassifier for Odd/High counts
- Rolling historical features for macro properties
- 5-fold TimeSeriesSplit validation
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# ── 1. Load Data & Engineer Targets ──────────────────────────────────────
DATA_PATH = "data/pension.csv"
df = pd.read_csv(DATA_PATH)
df = df.sort_values("round").reset_index(drop=True)

n_draws = len(df)
print(f"Total pension draws loaded: {n_draws}")

# Extract raw digits
digits = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values  # (n_draws, 6)

# Create macro targets
sum_digits = digits.sum(axis=1)  # range 0-54
odd_count = np.array([sum(1 for d in row if d % 2 == 1) for row in digits])  # range 0-6
high_count = np.array([sum(1 for d in row if d >= 5) for row in digits])  # range 0-6

print(f"\nTarget distributions:")
print(f"  Sum_Digits: mean={sum_digits.mean():.2f}, std={sum_digits.std():.2f}, range=[{sum_digits.min()}, {sum_digits.max()}]")
from collections import Counter
odd_dist = Counter(odd_count)
print(f"  Odd_Count: {dict(sorted(odd_dist.items()))}")
high_dist = Counter(high_count)
print(f"  High_Count: {dict(sorted(high_dist.items()))}")

# ── 2. Feature Engineering ───────────────────────────────────────────────
def compute_macro_features(draw_idx, sum_values, odd_values, high_values):
    """
    For a given draw index, compute rolling features for each macro target
    based ONLY on draws BEFORE draw_idx.
    
    Returns: feature vector for this draw
    """
    history_end = draw_idx
    n_history = history_end
    
    if n_history == 0:
        return np.zeros(36)  # 12 features × 3 targets
    
    sum_hist = sum_values[:history_end]
    odd_hist = odd_values[:history_end]
    high_hist = high_values[:history_end]
    
    features = []
    
    for hist_values in [sum_hist, odd_hist, high_hist]:
        # Rolling means
        def rolling_mean(N):
            if n_history == 0 or N == 0:
                return 0.0
            window = hist_values[-min(N, len(hist_values)):]
            return np.mean(window)
        
        # Rolling std
        def rolling_std(N):
            if n_history < 2 or N == 0:
                return 0.0
            window = hist_values[-min(N, len(hist_values)):]
            return np.std(window) if len(window) >= 2 else 0.0
        
        # Rolling sum
        def rolling_sum(N):
            if n_history == 0 or N == 0:
                return 0.0
            window = hist_values[-min(N, len(hist_values)):]
            return np.sum(window)
        
        # Rolling min/max
        def rolling_min(N):
            if n_history == 0 or N == 0:
                return 0.0
            window = hist_values[-min(N, len(hist_values)):]
            return np.min(window)
        
        def rolling_max(N):
            if n_history == 0 or N == 0:
                return 0.0
            window = hist_values[-min(N, len(hist_values)):]
            return np.max(window)
        
        # Rate of change (momentum)
        def rate_of_change(N):
            if n_history < N or N == 0:
                return 0.0
            recent = hist_values[-min(N, len(hist_values)):]
            older = hist_values[-min(2*N, len(hist_values)):-N] if n_history >= 2*N else hist_values[:max(1, len(hist_values)-N)]
            if len(older) == 0 or np.mean(older) == 0:
                return 0.0
            return (np.mean(recent) - np.mean(older)) / max(np.mean(older), 0.001)
        
        features.extend([
            rolling_mean(5), rolling_mean(10), rolling_mean(20),
            rolling_std(5), rolling_std(10), rolling_std(20),
            rolling_sum(5), rolling_sum(10),
            rolling_min(10), rolling_max(10),
            rate_of_change(10), rate_of_change(20),
        ])
    
    return np.array(features)

N_FEATURES = 36  # 12 features × 3 macro targets

# ── 3. Pre-compute features ──────────────────────────────────────────────
print("\nPre-computing macro features for all draws...")
all_features = []
for idx in range(n_draws):
    feats = compute_macro_features(idx, sum_digits, odd_count, high_count)
    all_features.append(feats)
all_features = np.array(all_features)
print(f"All features shape: {all_features.shape}")

# ── 4. TSCV Evaluation ───────────────────────────────────────────────────
MIN_HISTORY = 30
all_draw_indices = list(range(MIN_HISTORY, n_draws))
N_FOLDS = 5
tscv = TimeSeriesSplit(n_splits=N_FOLDS)

print(f"\n{'='*60}")
print("PENSION Phase 2 - 5-fold TSCV Evaluation")
print(f"{'='*60}")

target_cv_results = {
    "Sum_Digits": {"mae": [], "n_draws": []},
    "Odd_Count": {"accuracy": [], "n_draws": []},
    "High_Count": {"accuracy": [], "n_draws": []},
}

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(all_draw_indices)):
    fold_train_draws = [all_draw_indices[i] for i in train_idx]
    fold_val_draws = [all_draw_indices[i] for i in val_idx]
    
    print(f"\n  Fold {fold_idx + 1}: Train={len(fold_train_draws)}, Val={len(fold_val_draws)}")
    
    # ── Sum_Digits (Regression) ──
    X_train_sum = np.vstack([all_features[idx] for idx in fold_train_draws])
    y_train_sum = sum_digits[fold_train_draws]
    
    reg = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42 + fold_idx, n_jobs=-1,
        verbosity=0,
    )
    reg.fit(X_train_sum, y_train_sum)
    
    val_preds_sum = []
    val_actuals_sum = []
    for val_idx in fold_val_draws:
        X_val = all_features[val_idx].reshape(1, -1)
        pred = reg.predict(X_val)[0]
        val_preds_sum.append(pred)
        val_actuals_sum.append(sum_digits[val_idx])
    
    mae = mean_absolute_error(val_actuals_sum, val_preds_sum)
    target_cv_results["Sum_Digits"]["mae"].append(mae)
    target_cv_results["Sum_Digits"]["n_draws"].append(len(fold_val_draws))
    print(f"    Sum_Digits: MAE={mae:.4f}")
    
    # ── Odd_Count (Classification, 7 classes 0-6) ──
    X_train_odd = np.vstack([all_features[idx] for idx in fold_train_draws])
    y_train_odd = odd_count[fold_train_draws]
    
    # Ensure all 7 classes (0-6) are present in training data
    # by adding small dummy sample(s) for any missing class
    missing_classes = set(range(7)) - set(y_train_odd)
    if missing_classes:
        for mc in missing_classes:
            dummy_idx = np.random.randint(0, len(y_train_odd))
            x_dummy = X_train_odd[dummy_idx:dummy_idx+1].copy()
            y_dummy = np.array([mc])
            X_train_odd = np.vstack([X_train_odd, x_dummy])
            y_train_odd = np.concatenate([y_train_odd, y_dummy])
    
    clf_odd = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42 + fold_idx, n_jobs=-1,
        eval_metric='mlogloss', use_label_encoder=False,
        verbosity=0,
    )
    clf_odd.fit(X_train_odd, y_train_odd)
    
    odd_correct = 0
    for val_idx in fold_val_draws:
        X_val = all_features[val_idx].reshape(1, -1)
        pred = clf_odd.predict(X_val)[0]
        if pred == odd_count[val_idx]:
            odd_correct += 1
    
    odd_acc = odd_correct / len(fold_val_draws)
    target_cv_results["Odd_Count"]["accuracy"].append(odd_acc)
    print(f"    Odd_Count: accuracy={odd_acc:.4f} ({odd_correct}/{len(fold_val_draws)})")
    
    # ── High_Count (Classification, 7 classes 0-6) ──
    X_train_high = np.vstack([all_features[idx] for idx in fold_train_draws])
    y_train_high = high_count[fold_train_draws]
    
    missing_classes_high = set(range(7)) - set(y_train_high)
    if missing_classes_high:
        for mc in missing_classes_high:
            dummy_idx = np.random.randint(0, len(y_train_high))
            x_dummy = X_train_high[dummy_idx:dummy_idx+1].copy()
            y_dummy = np.array([mc])
            X_train_high = np.vstack([X_train_high, x_dummy])
            y_train_high = np.concatenate([y_train_high, y_dummy])
    
    clf_high = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42 + fold_idx, n_jobs=-1,
        eval_metric='mlogloss', use_label_encoder=False,
        verbosity=0,
    )
    clf_high.fit(X_train_high, y_train_high)
    
    high_correct = 0
    for val_idx in fold_val_draws:
        X_val = all_features[val_idx].reshape(1, -1)
        pred = clf_high.predict(X_val)[0]
        if pred == high_count[val_idx]:
            high_correct += 1
    
    high_acc = high_correct / len(fold_val_draws)
    target_cv_results["High_Count"]["accuracy"].append(high_acc)
    print(f"    High_Count: accuracy={high_acc:.4f} ({high_correct}/{len(fold_val_draws)})")

# ── 5. CV Summary ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CROSS-VALIDATION RESULTS (5-fold TSCV)")
print(f"{'='*60}")

mean_mae = np.mean(target_cv_results["Sum_Digits"]["mae"])
std_mae = np.std(target_cv_results["Sum_Digits"]["mae"])
print(f"  Sum_Digits: CV MAE = {mean_mae:.4f} ± {std_mae:.4f}")
print(f"    Per-fold MAEs: {[f'{v:.3f}' for v in target_cv_results['Sum_Digits']['mae']]}")

mean_odd = np.mean(target_cv_results["Odd_Count"]["accuracy"])
std_odd = np.std(target_cv_results["Odd_Count"]["accuracy"])
print(f"  Odd_Count: CV Accuracy = {mean_odd:.4f} ± {std_odd:.4f}")
print(f"    Per-fold: {[f'{v:.3f}' for v in target_cv_results['Odd_Count']['accuracy']]}")

mean_high = np.mean(target_cv_results["High_Count"]["accuracy"])
std_high = np.std(target_cv_results["High_Count"]["accuracy"])
print(f"  High_Count: CV Accuracy = {mean_high:.4f} ± {std_high:.4f}")
print(f"    Per-fold: {[f'{v:.3f}' for v in target_cv_results['High_Count']['accuracy']]}")

# Random baselines
sum_std_baseline = np.std(sum_digits[MIN_HISTORY:])  # guess mean → MAE ≈ std
odd_baseline = 1.0 / 7  # random guess: 1/7 ≈ 0.1429
high_baseline = 1.0 / 7
print(f"\n  Random baselines:")
print(f"    Sum_Digits MAE (predict mean): {sum_std_baseline:.4f}")
print(f"    Odd_Count (1/7): {odd_baseline:.4f}")
print(f"    High_Count (1/7): {high_baseline:.4f}")

# ── 6. Train Final Models on FULL Data ───────────────────────────────────
print(f"\n{'='*60}")
print("Training Final Models on Full Training Set")
print(f"{'='*60}")

TEST_SIZE = 30
train_indices = all_draw_indices[:-TEST_SIZE]
test_indices = all_draw_indices[-TEST_SIZE:]

X_train_full = np.vstack([all_features[idx] for idx in train_indices])

# Final Sum model
final_sum = XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0,
)
final_sum.fit(X_train_full, sum_digits[train_indices])
print("  Sum_Digits: trained")

# Final Odd model
final_odd = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    eval_metric='mlogloss', use_label_encoder=False,
    verbosity=0,
)
final_odd.fit(X_train_full, odd_count[train_indices])
print("  Odd_Count: trained")

# Final High model
final_high = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    eval_metric='mlogloss', use_label_encoder=False,
    verbosity=0,
)
final_high.fit(X_train_full, high_count[train_indices])
print("  High_Count: trained")

# ── 7. Evaluate on Held-Out Test Set ─────────────────────────────────────
print(f"\n{'='*60}")
print("Held-Out Test Set Evaluation (Last 30 draws)")
print(f"{'='*60}")

# Sum
test_preds_sum = []
test_actuals_sum = []
for idx in test_indices:
    X_test = all_features[idx].reshape(1, -1)
    pred = final_sum.predict(X_test)[0]
    test_preds_sum.append(pred)
    test_actuals_sum.append(sum_digits[idx])
test_mae = mean_absolute_error(test_actuals_sum, test_preds_sum)
print(f"  Sum_Digits: Test MAE = {test_mae:.4f}")

# Odd
odd_test_correct = 0
for idx in test_indices:
    X_test = all_features[idx].reshape(1, -1)
    pred = final_odd.predict(X_test)[0]
    if pred == odd_count[idx]:
        odd_test_correct += 1
odd_test_acc = odd_test_correct / len(test_indices)
print(f"  Odd_Count: Test Accuracy = {odd_test_acc:.4f} ({odd_test_correct}/{len(test_indices)})")

# High
high_test_correct = 0
for idx in test_indices:
    X_test = all_features[idx].reshape(1, -1)
    pred = final_high.predict(X_test)[0]
    if pred == high_count[idx]:
        high_test_correct += 1
high_test_acc = high_test_correct / len(test_indices)
print(f"  High_Count: Test Accuracy = {high_test_acc:.4f} ({high_test_correct}/{len(test_indices)})")

# ── 8. Predict Next Draw ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Prediction for next draw (Draw #{n_draws + 1})")
print(f"{'='*60}")

next_feats = compute_macro_features(n_draws, sum_digits, odd_count, high_count).reshape(1, -1)

pred_sum = final_sum.predict(next_feats)[0]
pred_odd = int(final_odd.predict(next_feats)[0])
pred_high = int(final_high.predict(next_feats)[0])

print(f"\n  Predicted Sum:        {pred_sum:.2f} (actual range: 0-54)")
print(f"  Predicted Odd Count:  {pred_odd} (out of 6)")
print(f"  Predicted High Count: {pred_high} (out of 6)")

# Show prediction probabilities for classifiers
print(f"\n  Odd_Count probabilities:")
odd_probs = final_odd.predict_proba(next_feats)[0]
for cls in np.argsort(odd_probs)[-3:][::-1]:
    print(f"    count={cls}: prob={odd_probs[cls]:.4f}")

print(f"\n  High_Count probabilities:")
high_probs = final_high.predict_proba(next_feats)[0]
for cls in np.argsort(high_probs)[-3:][::-1]:
    print(f"    count={cls}: prob={high_probs[cls]:.4f}")

# ── 9. Log to pension_log.tsv ────────────────────────────────────────────
LOG_FILE = "pension/pension_log.tsv"
log_entry = {
    "phase": 2,
    "description": "Pension Phase 2 - Macro Properties (Sum, Odd_Count, High_Count)",
    "targets": "Sum_Digits,Odd_Count,High_Count",
    "features": "Rolling mean/std/sum/min/max/roc of macro targets (last 5/10/20 draws)",
    "model": "XGBRegressor(Sum) + XGBClassifier(Odd/High), n_estimators=200, max_depth=5",
    "sum_mae": f"{mean_mae:.4f}",
    "odd_accuracy": f"{mean_odd:.4f}",
    "high_accuracy": f"{mean_high:.4f}",
    "next_prediction": f"sum={pred_sum:.1f},odd={pred_odd},high={pred_high}",
}
header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

# ── 10. Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PENSION PHASE 2 COMPLETE - SUMMARY")
print(f"{'='*60}")
print(f"  Targets: Sum_Digits (reg), Odd_Count (cls), High_Count (cls)")
print(f"  Features: 36 (12 rolling features × 3 targets)")
print(f"  Model: XGBoost (Regressor + 2 Classifiers)")
print(f"  CV: 5-fold TimeSeriesSplit")
print(f"\n  CV Results:")
print(f"    Sum_Digits: MAE = {mean_mae:.4f} ± {std_mae:.4f} (baseline: {sum_std_baseline:.4f})")
print(f"    Odd_Count:  Accuracy = {mean_odd:.4f} ± {std_odd:.4f} (baseline: {odd_baseline:.4f})")
print(f"    High_Count: Accuracy = {mean_high:.4f} ± {std_high:.4f} (baseline: {high_baseline:.4f})")
print(f"\n  Test Results:")
print(f"    Sum_Digits: MAE = {test_mae:.4f}")
print(f"    Odd_Count:  Accuracy = {odd_test_acc:.4f}")
print(f"    High_Count: Accuracy = {high_test_acc:.4f}")
print(f"\n  Next draw (#{n_draws + 1}) prediction:")
print(f"    Sum: {pred_sum:.1f}, Odd Count: {pred_odd}, High Count: {pred_high}")
print("\nPension Phase 2 complete.")