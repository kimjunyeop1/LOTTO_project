#!/usr/bin/env python3
"""
Lotto 6/45 Phase 7 (Hybrid Ensemble: Linear + XGBoost)
- Phase 5 features (12 time-series/momentum) with strict zero leakage
- TimeSeriesSplit (5-fold) walk-forward validation
- Hybrid ensemble: LogisticRegression (L2) + XGBoost via VotingClassifier
- Comparison: Phase 5 XGBoost alone vs Phase 7 Ensemble
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# ── 1. Load Data ──────────────────────────────────────────────────────────
DATA_PATH = "data/lotto.csv"
df = pd.read_csv(DATA_PATH)
df = df.sort_values("round").reset_index(drop=True)

draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values.tolist()
n_draws = len(draws)
print(f"Total draws loaded: {n_draws}")

# ── 2. Feature Engineering (Phase 5 features, strict zero leakage) ──────
def compute_features_for_draw(draw_idx, draws):
    """Per-number features based ONLY on draws BEFORE draw_idx. Returns 45 x 12."""
    history = draws[:draw_idx]
    n_history = len(history)
    num_appearances = defaultdict(list)
    appeared_at = defaultdict(set)
    for i, drawn in enumerate(history):
        for num in drawn:
            num_appearances[num].append(i)
            appeared_at[num].add(i)
    features = []
    for num in range(1, 46):
        def rolling_freq(N):
            if n_history == 0:
                return 0
            start = max(0, n_history - N)
            cnt = 0
            appeared = appeared_at.get(num, set())
            for idx in range(start, n_history):
                if idx in appeared:
                    cnt += 1
            return cnt
        f5 = rolling_freq(5); f10 = rolling_freq(10)
        f15 = rolling_freq(15); f30 = rolling_freq(30); f50 = rolling_freq(50)
        appear_list = sorted(num_appearances.get(num, []))
        current_gap = (n_history - 1) - appear_list[-1] if appear_list else n_history
        if len(appear_list) >= 2:
            previous_gap = appear_list[-1] - appear_list[-2]
        elif len(appear_list) == 1:
            previous_gap = appear_list[0]
        else:
            previous_gap = n_history
        if len(appear_list) >= 2:
            historical_avg_gap = np.mean([appear_list[i+1] - appear_list[i] for i in range(len(appear_list)-1)])
        elif len(appear_list) == 1:
            historical_avg_gap = appear_list[0]
        else:
            historical_avg_gap = n_history
        mom_10_50 = f10 / f50 if f50 > 0 else 0.0
        mom_10_30 = f10 / f30 if f30 > 0 else 0.0
        mom_15_50 = f15 / f50 if f50 > 0 else 0.0
        recency_score = 0.0
        if appear_list:
            for appear_idx in appear_list:
                recency_score += np.exp(-(n_history - 1 - appear_idx) / 10.0)
        recency_score /= max(n_history, 1)
        features.append([f5, f10, f15, f30, f50, current_gap, previous_gap,
                         historical_avg_gap, mom_10_50, mom_10_30, mom_15_50, recency_score])
    return np.array(features)

feature_names = ["Freq5","Freq10","Freq15","Freq30","Freq50","CurrentGap",
                 "PrevGap","HistAvgGap","Mom_10_50","Mom_10_30","Mom_15_50","RecencyScore"]
N_FEATURES = 12

def target_vector(drawn_numbers):
    vec = np.zeros(45, dtype=int)
    for num in drawn_numbers:
        vec[num - 1] = 1
    return vec

# ── 3. Pre-compute Features ──────────────────────────────────────────────
print("Pre-computing features for all draws...")
all_features = []
for idx in range(n_draws):
    all_features.append(compute_features_for_draw(idx, draws))
all_features = np.array(all_features)
print(f"All features shape: {all_features.shape}")

# ── 4. Prepare dataset ──────────────────────────────────────────────────
MIN_HISTORY = 50
all_draw_indices = list(range(MIN_HISTORY, n_draws))

def prepare_draw_samples(draw_indices):
    X_list, y_list = [], []
    for idx in draw_indices:
        X_list.append(all_features[idx])
        y_list.append(target_vector(draws[idx]))
    return np.vstack(X_list), np.concatenate(y_list)

# ── 5. TSCV Hit Rate Evaluation Function ─────────────────────────────────
def evaluate_hit_rate(model, X_list_by_draw, y_draw_indices):
    """
    Evaluate hit rate: for each draw, get per-number probs, pick top-6.
    X_list_by_draw: list of (45, n_features) per draw
    y_draw_indices: list of draw indices (for retrieving actual numbers from draws[])
    Returns average hit rate.
    """
    total_hits = 0
    n_draws_eval = len(X_list_by_draw)
    for i in range(n_draws_eval):
        if hasattr(model, "predict_proba"):
            if len(model.classes_) == 2:
                probs = model.predict_proba(X_list_by_draw[i])[:, 1]
            else:
                # Fallback
                probs = model.predict_proba(X_list_by_draw[i])
                if probs.ndim == 2 and probs.shape[1] == 2:
                    probs = probs[:, 1]
                else:
                    probs = probs.ravel()
        else:
            probs = model.predict(X_list_by_draw[i])
        top6_idx = np.argsort(probs)[-6:][::-1]
        top6 = set(int(x + 1) for x in top6_idx)
        actual = set(draws[y_draw_indices[i]])
        hits = sum(1 for n in top6 if n in actual)
        total_hits += hits
    return total_hits / max(n_draws_eval, 1)

# ── 6. TSCV Evaluation ──────────────────────────────────────────────────
N_FOLDS = 5
tscv = TimeSeriesSplit(n_splits=N_FOLDS)

print(f"\n{'='*60}")
print("Phase 7 - Hybrid Ensemble (Linear + XGBoost) - 5-fold TSCV")
print(f"{'='*60}")

cv_results_xgb = []
cv_results_ensemble = []
all_fold_train_indices = []
all_fold_val_indices = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(all_draw_indices)):
    fold_train_draws = [all_draw_indices[i] for i in train_idx]
    fold_val_draws = [all_draw_indices[i] for i in val_idx]
    all_fold_train_indices.append(fold_train_draws)
    all_fold_val_indices.append(fold_val_draws)

    # Prepare training data
    X_train_fold, y_train_fold = prepare_draw_samples(fold_train_draws)
    X_val_list = [all_features[idx] for idx in fold_val_draws]

    # ── Model 1: XGBoost (Phase 5 best params) ──
    xgb = XGBClassifier(
        n_estimators=357, max_depth=7, learning_rate=0.0117,
        subsample=0.8, colsample_bytree=0.585, gamma=0.325,
        min_child_weight=10, reg_alpha=4.828, reg_lambda=4.042,
        random_state=42, n_jobs=-1, eval_metric='logloss',
        use_label_encoder=False, verbosity=0,
    )
    xgb.fit(X_train_fold, y_train_fold)

    # ── Model 2: Logistic Regression (L2 penalty) ──
    lr = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs', max_iter=1000,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    lr.fit(X_train_fold, y_train_fold)

    # ── Model 3: Hybrid Ensemble (VotingClassifier, soft voting) ──
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('xgb', xgb),
        ],
        voting='soft',  # Average predicted probabilities
        weights=[0.3, 0.7],  # Weighted: XGBoost gets 70%, LR gets 30%
    )
    # No need to refit, already fitted individually
    # We need to fit the ensemble fresh
    ensemble.fit(X_train_fold, y_train_fold)

    # ── Evaluate XGBoost alone ──
    xgb_hit_rate = evaluate_hit_rate(xgb, X_val_list, fold_val_draws)
    cv_results_xgb.append(xgb_hit_rate)

    # ── Evaluate Ensemble ──
    ensemble_hit_rate = evaluate_hit_rate(ensemble, X_val_list, fold_val_draws)
    cv_results_ensemble.append(ensemble_hit_rate)

    print(f"\n  Fold {fold_idx + 1}:")
    print(f"    Train draws: {len(fold_train_draws)}, Val draws: {len(fold_val_draws)}")
    print(f"    XGBoost alone:   {xgb_hit_rate:.4f}")
    print(f"    Ensemble (LR+XGB): {ensemble_hit_rate:.4f}")

mean_xgb = np.mean(cv_results_xgb)
mean_ensemble = np.mean(cv_results_ensemble)
std_xgb = np.std(cv_results_xgb)
std_ensemble = np.std(cv_results_ensemble)

print(f"\n{'='*60}")
print("CROSS-VALIDATION RESULTS (5-fold TSCV)")
print(f"{'='*60}")
print(f"  XGBoost alone:          {mean_xgb:.4f} ± {std_xgb:.4f}")
print(f"  Hybrid Ensemble (LR+XGB): {mean_ensemble:.4f} ± {std_ensemble:.4f}")
improvement = mean_ensemble - mean_xgb
print(f"  Improvement:            {improvement:+.4f}")
if mean_ensemble > mean_xgb:
    print(f"  ✅ Hybrid Ensemble BEATS XGBoost alone!")
else:
    print(f"  ⚠️  Hybrid Ensemble does not beat XGBoost alone.")

# Per-fold detail
print(f"\n  Per-fold details:")
for i in range(N_FOLDS):
    marker = "✅" if cv_results_ensemble[i] > cv_results_xgb[i] else "⚠️"
    print(f"    Fold {i+1}: XGB={cv_results_xgb[i]:.4f} Ensemble={cv_results_ensemble[i]:.4f} {marker}")

# ── 7. Train Final Model on FULL Training Set ────────────────────────────
print(f"\n{'='*60}")
print("Training Final Models on Full Training Set")
print(f"{'='*60}")

TEST_SIZE = 50
train_indices = all_draw_indices[:-TEST_SIZE]
test_indices = all_draw_indices[-TEST_SIZE:]

X_train_full, y_train_full = prepare_draw_samples(train_indices)
X_test_list = [all_features[idx] for idx in test_indices]

# Final XGBoost
final_xgb = XGBClassifier(
    n_estimators=357, max_depth=7, learning_rate=0.0117,
    subsample=0.8, colsample_bytree=0.585, gamma=0.325,
    min_child_weight=10, reg_alpha=4.828, reg_lambda=4.042,
    random_state=42, n_jobs=-1, eval_metric='logloss',
    use_label_encoder=False, verbosity=0,
)
final_xgb.fit(X_train_full, y_train_full)

# Final LR
final_lr = LogisticRegression(
    penalty='l2', C=1.0, solver='lbfgs', max_iter=1000,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
final_lr.fit(X_train_full, y_train_full)

# Final Ensemble
final_ensemble = VotingClassifier(
    estimators=[('lr', final_lr), ('xgb', final_xgb)],
    voting='soft', weights=[0.3, 0.7],
)
final_ensemble.fit(X_train_full, y_train_full)

print("Final models trained.")

# ── 8. Evaluate on Held-Out Test Set ─────────────────────────────────────
print(f"\n{'='*60}")
print("Held-Out Test Set Evaluation (Last 50 draws)")
print(f"{'='*60}")

xgb_test_hits = 0
ensemble_test_hits = 0

for i, actual_drawn_idx in enumerate(test_indices):
    X_draw = X_test_list[i]

    # XGBoost prediction
    xgb_probs = final_xgb.predict_proba(X_draw)[:, 1]
    xgb_top6 = sorted([int(x + 1) for x in np.argsort(xgb_probs)[-6:][::-1]])

    # Ensemble prediction
    ens_probs = final_ensemble.predict_proba(X_draw)
    if ens_probs.ndim == 2 and ens_probs.shape[1] == 2:
        ens_probs = ens_probs[:, 1]
    else:
        ens_probs = ens_probs.ravel()
    ens_top6 = sorted([int(x + 1) for x in np.argsort(ens_probs)[-6:][::-1]])

    actual = draws[actual_drawn_idx]
    actual_set = set(actual)

    xgb_hits = sum(1 for n in xgb_top6 if n in actual_set)
    ens_hits = sum(1 for n in ens_top6 if n in actual_set)
    xgb_test_hits += xgb_hits
    ensemble_test_hits += ens_hits

    marker = "✅" if ens_hits >= xgb_hits else ""
    print(f"  Draw {actual_drawn_idx + 1}: XGB={xgb_top6} hits={xgb_hits} | Ensemble={ens_top6} hits={ens_hits} {marker}")

xgb_test_rate = xgb_test_hits / len(test_indices)
ensemble_test_rate = ensemble_test_hits / len(test_indices)

print(f"\n{'='*60}")
print(f"XGBoost Test Hit Rate:      {xgb_test_rate:.4f} ({xgb_test_hits}/{len(test_indices)})")
print(f"Hybrid Ensemble Test Rate:  {ensemble_test_rate:.4f} ({ensemble_test_hits}/{len(test_indices)})")
print(f"Improvement (test):         {ensemble_test_rate - xgb_test_rate:+.4f}")

# ── 9. Predict Next Draw ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Prediction for next draw (Draw #{n_draws + 1})")
print(f"{'='*60}")

next_features = all_features[n_draws - 1]  # features for next draw based on last draw's history
# Actually we need features computed with ALL draws as history
next_features = compute_features_for_draw(n_draws, draws)  # strict: use all available draws

xgb_next_probs = final_xgb.predict_proba(next_features)[:, 1]
ens_next_probs = final_ensemble.predict_proba(next_features)
if ens_next_probs.ndim == 2 and ens_next_probs.shape[1] == 2:
    ens_next_probs = ens_next_probs[:, 1]
else:
    ens_next_probs = ens_next_probs.ravel()

lr_next_probs = final_lr.predict_proba(next_features)[:, 1]

xgb_top6 = sorted([int(x + 1) for x in np.argsort(xgb_next_probs)[-6:][::-1]])
ens_top6 = sorted([int(x + 1) for x in np.argsort(ens_next_probs)[-6:][::-1]])
lr_top6 = sorted([int(x + 1) for x in np.argsort(lr_next_probs)[-6:][::-1]])

print(f"  Logistic Regression: {lr_top6}")
print(f"  XGBoost:             {xgb_top6}")
print(f"  Hybrid Ensemble:     {ens_top6}")
print(f"\n  >>> Phase 7 Recommended Numbers: {ens_top6}")

# Show Top-12 probabilities from ensemble
print(f"\n  Top-12 numbers by Ensemble probability:")
top12_idx = np.argsort(ens_next_probs)[-12:][::-1]
for rank, idx in enumerate(top12_idx, 1):
    num = idx + 1
    prob = ens_next_probs[idx]
    xgb_prob = xgb_next_probs[idx]
    lr_prob = lr_next_probs[idx]
    print(f"    #{rank:2d}: Num {num:2d} | Ensemble={prob:.4f} (XGB={xgb_prob:.4f}, LR={lr_prob:.4f})")

# ── 10. Log to experiment_log.tsv ────────────────────────────────────────
LOG_FILE = "experiment_log.tsv"
log_entry = {
    "phase": 7,
    "description": "Phase 7 - Hybrid Ensemble (LogisticRegression L2 + XGBoost, soft voting, 70/30 weight)",
    "train_draws": len(train_indices),
    "test_draws": len(test_indices),
    "features": "12 Phase5 features (Freq5-50, Gap stats, Momentum, Recency)",
    "model": "VotingClassifier(LR(C=1.0,L2) + XGBoost(Phase5 best)), voting=soft, weights=[0.3,0.7]",
    "avg_hit_rate": f"{ensemble_test_rate:.4f}",
    "total_matches": ensemble_test_hits,
    "next_prediction": ",".join(map(str, ens_top6)),
}
header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

# ── 11. Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 7 COMPLETE - SUMMARY")
print(f"{'='*60}")
print(f"  Model 1: LogisticRegression (L2, C=1.0, balanced)")
print(f"  Model 2: XGBoost (Phase 5 optimized)")
print(f"  Fusion:  VotingClassifier (soft, 30% LR + 70% XGB)")
print(f"  CV (5-fold TSCV):")
print(f"    XGBoost alone:        {mean_xgb:.4f} ± {std_xgb:.4f}")
print(f"    Hybrid Ensemble:      {mean_ensemble:.4f} ± {std_ensemble:.4f}")
print(f"  Held-out Test:")
print(f"    XGBoost alone:        {xgb_test_rate:.4f}")
print(f"    Hybrid Ensemble:      {ensemble_test_rate:.4f}")
print(f"  Next draw:              {ens_top6}")
print("\nPhase 7 complete.")