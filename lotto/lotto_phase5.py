#!/usr/bin/env python3
"""
Lotto 6/45 Phase 5 (Robust CV + Optuna Tuning + SHAP)
- Phase 4 features (12 rich time-series/momentum) with STRICT zero leakage
- TimeSeriesSplit (5-fold) walk-forward validation
- Optuna hyperparameter optimization (50+ trials)
- SHAP feature importance analysis
- Final prediction with uncertainty estimate
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
import optuna
import shap

# ── 1. Load Data ──────────────────────────────────────────────────────────
DATA_PATH = "data/lotto.csv"
df = pd.read_csv(DATA_PATH)
df = df.sort_values("round").reset_index(drop=True)

draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values.tolist()
n_draws = len(draws)
print(f"Total draws loaded: {n_draws}")

# ── 2. Feature Engineering (Phase 4 features, strict zero leakage) ──────
def compute_features_for_draw(draw_idx, draws):
    """
    Per-number features based ONLY on draws BEFORE draw_idx.
    Returns 45 x 12 feature matrix.
    """
    history = draws[:draw_idx]
    n_history = len(history)

    # Build number appearance timeline
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

        f5 = rolling_freq(5)
        f10 = rolling_freq(10)
        f15 = rolling_freq(15)
        f30 = rolling_freq(30)
        f50 = rolling_freq(50)

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

        # Momentum ratios
        momentum_10_50 = f10 / f50 if f50 > 0 else 0.0
        momentum_10_30 = f10 / f30 if f30 > 0 else 0.0
        momentum_15_50 = f15 / f50 if f50 > 0 else 0.0

        # Recency score with exponential decay
        recency_score = 0.0
        if appear_list:
            for appear_idx in appear_list:
                recency_score += np.exp(-(n_history - 1 - appear_idx) / 10.0)
        recency_score /= max(n_history, 1)

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

feature_names = [
    "Freq5", "Freq10", "Freq15", "Freq30", "Freq50",
    "CurrentGap", "PrevGap", "HistAvgGap",
    "Mom_10_50", "Mom_10_30", "Mom_15_50",
    "RecencyScore"
]
N_FEATURES = 12

# ── 3. Prepare dataset as list of (features, targets) per draw ───────────
MIN_HISTORY = 50  # Need at least 50 draws for rolling features

def prepare_draw_samples(draw_indices, full_draws):
    """For a list of draw indices, compute features & targets."""
    X_list, y_list = [], []
    for idx in draw_indices:
        feats = compute_features_for_draw(idx, full_draws)
        target = target_vector(full_draws[idx])
        X_list.append(feats)
        y_list.append(target)
    return X_list, y_list

# All draws from MIN_HISTORY onwards
all_draw_indices = list(range(MIN_HISTORY, n_draws))

# Split: keep last 50 for final held-out test set
TEST_SIZE = 50
train_indices = all_draw_indices[:-TEST_SIZE]
test_indices = all_draw_indices[-TEST_SIZE:]

print(f"\nTraining draws: {len(train_indices)}, Test draws: {len(test_indices)}")

# ── 4. Optuna Objective Function with TimeSeriesSplit ───────────────────
N_FOLDS = 5
tscv = TimeSeriesSplit(n_splits=N_FOLDS)

def compute_hit_rate(y_true_draws_list, y_pred_probs_list, full_draws_ref, draw_indices):
    """
    Compute per-draw hit rate: for each draw, top-6 hit count.
    y_true_draws_list: list of target vectors (binary 45-length arrays)
    """
    total_hits = 0
    n_draws_eval = len(y_true_draws_list)
    for i in range(n_draws_eval):
        probs = y_pred_probs_list[i][:, 1]
        top6_idx = np.argsort(probs)[-6:][::-1]
        top6 = set(int(x + 1) for x in top6_idx)
        actual_draw = full_draws_ref[draw_indices[i]]
        actual_set = set(actual_draw)
        hits = sum(1 for n in top6 if n in actual_set)
        total_hits += hits
    return total_hits / max(n_draws_eval, 1)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }

    cv_hit_rates = []

    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(tscv.split(train_indices)):
        fold_train_draws_idx = [train_indices[i] for i in train_fold_idx]
        fold_val_draws_idx = [train_indices[i] for i in val_fold_idx]

        # Build features for this fold
        X_fold_train_list, y_fold_train_list = prepare_draw_samples(fold_train_draws_idx, draws)
        X_fold_val_list, y_fold_val_list = prepare_draw_samples(fold_val_draws_idx, draws)

        X_ft = np.vstack(X_fold_train_list)
        y_ft = np.concatenate(y_fold_train_list)

        model = XGBClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
        )
        model.fit(X_ft, y_ft)

        # Predict on validation
        val_probs = [model.predict_proba(X_fold_val_list[i]) for i in range(len(X_fold_val_list))]
        val_hit_rate = compute_hit_rate(y_fold_val_list, val_probs, draws, fold_val_draws_idx)
        cv_hit_rates.append(val_hit_rate)

    mean_hit_rate = np.mean(cv_hit_rates)
    return mean_hit_rate


print("\n--- Optuna Hyperparameter Tuning (50+ trials, 5-fold TSCV) ---")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_cv_hit_rate = study.best_value
print(f"\nBest CV Hit Rate: {best_cv_hit_rate:.4f}")
print(f"Best params: {best_params}")

# ── 5. Train Final Model on Full Training Set ────────────────────────────
print("\n--- Training Final Model on Full Training Set ---")
X_train_list, y_train_list = prepare_draw_samples(train_indices, draws)
X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)

final_model = XGBClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0,
)
final_model.fit(X_train, y_train)
print("Final model trained.")

# ── 6. Evaluate on Held-Out Test Set ─────────────────────────────────────
print(f"\n{'='*60}")
print("Phase 5 - Held-Out Test Set Evaluation (Last 50 draws)")
print(f"{'='*60}")

X_test_list, y_test_list = prepare_draw_samples(test_indices, draws)
test_probs = [final_model.predict_proba(X_test_list[i]) for i in range(len(X_test_list))]
test_hit_rate = compute_hit_rate(y_test_list, test_probs, draws, test_indices)

total_hits_test = 0
for i, actual_drawn_idx in enumerate(test_indices):
    probs = test_probs[i][:, 1]
    top6_idx = np.argsort(probs)[-6:][::-1]
    top6_numbers = sorted([int(x + 1) for x in top6_idx])
    actual = draws[actual_drawn_idx]
    hits = sum(1 for n in top6_numbers if n in actual)
    total_hits_test += hits
    print(f"  Draw {actual_drawn_idx + 1}: Pred {top6_numbers} | Actual {sorted(actual)} | Hits {hits}")

print(f"\n{'='*60}")
print(f"Held-Out Test Hit Rate: {test_hit_rate:.4f} (Total matches: {total_hits_test}/{len(test_indices)})")
print(f"Best CV Hit Rate (from Optuna): {best_cv_hit_rate:.4f}")

# ── 7. SHAP Analysis ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SHAP Feature Importance Analysis")
print(f"{'='*60}")

# Use a representative sample for SHAP (100 random draws from training)
np.random.seed(42)
sample_indices = np.random.choice(len(X_train), min(5000, len(X_train)), replace=False)
X_shap_sample = X_train[sample_indices]

# SHAP KernelExplainer would be too slow, use TreeExplainer
print("Computing SHAP values with TreeExplainer...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_shap_sample)

# If binary classification, shap_values might be list or array
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # class 1 (number appears)
else:
    shap_vals = shap_values

# Mean absolute SHAP values per feature
mean_shap = np.abs(shap_vals).mean(axis=0)
feature_importance = list(zip(feature_names, mean_shap))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print(f"\nFeature Importance (mean |SHAP value|):")
print(f"{'Rank':<5} {'Feature':<18} {'Importance':<12} {'Pct':<8}")
print("-" * 43)
for rank, (name, imp) in enumerate(feature_importance, 1):
    pct = imp / max(mean_shap) * 100
    print(f"{rank:<5} {name:<18} {imp:<12.6f} {pct:<8.1f}%")

# ── 8. Predict Next Draw (with uncertainty sampling) ─────────────────────
print(f"\n{'='*60}")
print(f"Prediction for next draw (Draw #{n_draws + 1})")
print(f"{'='*60}")

next_features = compute_features_for_draw(n_draws, draws)
next_probs = final_model.predict_proba(next_features)[:, 1]

# Top-12 numbers for context
top12_idx = np.argsort(next_probs)[-12:][::-1]
print(f"\nTop-12 numbers by probability:")
for rank, idx in enumerate(top12_idx, 1):
    num = idx + 1
    prob = next_probs[idx]
    print(f"  #{rank:2d}: Num {num:2d} | prob={prob:.4f}")

# Final Top-6
next_top6_idx = np.argsort(next_probs)[-6:][::-1]
next_top6 = sorted([int(x + 1) for x in next_top6_idx])
print(f"\n>>> Next Draw Recommended Numbers: {next_top6}")

# Show feature context for top-6
print(f"\n  Feature context for Top-6 prediction:")
next_feat_df = pd.DataFrame(next_features, columns=feature_names)
for rank, idx in enumerate(next_top6_idx, 1):
    num = idx + 1
    prob = next_probs[idx]
    row = next_feat_df.iloc[idx]
    print(f"    #{rank}: Num {num:2d} prob={prob:.4f} | "
          f"Freq5={row['Freq5']:.0f} Freq10={row['Freq10']:.0f} Freq30={row['Freq30']:.0f} "
          f"Gap={row['CurrentGap']:.0f} Mom_10_50={row['Mom_10_50']:.2f} Recency={row['RecencyScore']:.4f}")

# ── 9. Log to experiment_log.tsv ────────────────────────────────────────
LOG_FILE = "experiment_log.tsv"
best_params_str = str({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()})
log_entry = {
    "phase": 5,
    "description": "Phase 5 - XGBoost + Optuna (50 trials) + 5-fold TSCV + SHAP",
    "train_draws": len(train_indices),
    "test_draws": len(test_indices),
    "features": "Freq5,Freq10,Freq15,Freq30,Freq50,CurrentGap,PrevGap,HistAvgGap,Mom_10_50,Mom_10_30,Mom_15_50,RecencyScore",
    "model": f"XGBoost Optuna best: {best_params_str}",
    "avg_hit_rate": f"{test_hit_rate:.4f}",
    "total_matches": total_hits_test,
    "next_prediction": ",".join(map(str, next_top6)),
}

header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

# ── 10. Summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 5 COMPLETE - SUMMARY")
print(f"{'='*60}")
print(f"  Validation method:     TimeSeriesSplit (5-fold)")
print(f"  Optuna trials:         50")
print(f"  Best CV Hit Rate:      {best_cv_hit_rate:.4f}")
print(f"  Held-out Test Hit Rate: {test_hit_rate:.4f}")
print(f"  Best params:           {best_params}")
print(f"\n  Top 3 SHAP features:")
for rank, (name, imp) in enumerate(feature_importance[:3], 1):
    print(f"    {rank}. {name}: {imp:.6f}")
print(f"\n  Next draw prediction:  {next_top6}")
print("\nPhase 5 complete.")