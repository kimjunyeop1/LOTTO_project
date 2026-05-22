#!/usr/bin/env python3
"""
Lotto 6/45 Phase 6 (Deep Learning LSTM)
- 3D sequence data from Phase 5 features (Gap, Recency, Rolling Freq)
- PyTorch LSTM with strong Dropout + L2 regularization + Early Stopping
- Per-number shared LSTM: each number treated as independent time series
- Pre-computed features for efficiency
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 1. Load Data ──────────────────────────────────────────────────────────
DATA_PATH = "data/lotto.csv"
df = pd.read_csv(DATA_PATH)
df = df.sort_values("round").reset_index(drop=True)

draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values.tolist()
n_draws = len(draws)
print(f"Total draws loaded: {n_draws}")

# ── 2. Feature Engineering ───────────────────────────────────────────────
def compute_features_for_draw(draw_idx, draws):
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

# ── 3. Pre-compute Features for ALL draws (once) ─────────────────────────
print("Pre-computing features for all draws...")
all_features = []
for idx in range(n_draws):
    all_features.append(compute_features_for_draw(idx, draws))
all_features = np.array(all_features)  # (n_draws, 45, 12)
print(f"All features shape: {all_features.shape}")

# ── 4. Build 3D Sequences ────────────────────────────────────────────────
SEQ_LEN = 10

def build_sequences(draw_indices, full_draws, feature_cache, seq_len=SEQ_LEN):
    X_list, y_list, sample_map = [], [], []
    for draw_idx in draw_indices:
        history_start = max(0, draw_idx - seq_len)
        for num_idx in range(45):
            seq_features = feature_cache[history_start:draw_idx, num_idx, :].copy()
            if len(seq_features) < seq_len:
                pad_len = seq_len - len(seq_features)
                pad = np.zeros((pad_len, N_FEATURES))
                seq_features = np.vstack([pad, seq_features])
            target = 1.0 if (num_idx + 1) in full_draws[draw_idx] else 0.0
            X_list.append(seq_features)
            y_list.append(target)
            sample_map.append((draw_idx, num_idx + 1))
    return np.array(X_list), np.array(y_list), sample_map

# ── 5. Split Data ────────────────────────────────────────────────────────
MIN_HISTORY = 50
all_draw_indices = list(range(MIN_HISTORY, n_draws))
TEST_SIZE = 50
test_indices = all_draw_indices[-TEST_SIZE:]
train_val_indices = all_draw_indices[:-TEST_SIZE]
VAL_SIZE = 100
val_indices = train_val_indices[-VAL_SIZE:]
train_indices = train_val_indices[:-VAL_SIZE]

print(f"\nTrain draws: {len(train_indices)}, Val draws: {len(val_indices)}, Test draws: {len(test_indices)}")

print("Building training sequences...")
X_train, y_train, _ = build_sequences(train_indices, draws, all_features)
print(f"  Train samples: {X_train.shape}")

print("Building validation sequences...")
X_val, y_val, _ = build_sequences(val_indices, draws, all_features)
print(f"  Val samples: {X_val.shape}")

print("Building test sequences...")
X_test, y_test, test_map = build_sequences(test_indices, draws, all_features)
print(f"  Test samples: {X_test.shape}")

# ── 6. Normalize features per-channel ────────────────────────────────────
X_train_flat = X_train.reshape(-1, N_FEATURES)
feat_mean = X_train_flat.mean(axis=0)
feat_std = X_train_flat.std(axis=0) + 1e-8
def normalize(X):
    return (X - feat_mean) / feat_std

X_train_norm = normalize(X_train)
X_val_norm = normalize(X_val)
X_test_norm = normalize(X_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

X_train_t = torch.FloatTensor(X_train_norm).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val_norm).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test_norm).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# ── 7. LSTM Model Definition ─────────────────────────────────────────────
class LotteryLSTM(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=64, num_layers=2,
                 dropout=0.5, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        dropped = self.dropout(last_out)
        out = self.fc(dropped)
        return self.sigmoid(out).squeeze()

model = LotteryLSTM(input_size=N_FEATURES, hidden_size=64, num_layers=2,
                    dropout=0.5, output_size=1).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nLSTM Model parameters: {total_params:,}")
print(model)

# ── 8. Training Setup ────────────────────────────────────────────────────
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

N_EPOCHS = 200
patience = 15
best_val_loss = float('inf')
best_epoch = -1
best_model_state = None
patience_counter = 0
train_losses, val_losses = [], []

print(f"\n{'='*60}")
print("Training LSTM with Early Stopping")
print(f"{'='*60}")

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_train_loss = 0.0
    n_train_batches = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item()
        n_train_batches += 1
    avg_train_loss = epoch_train_loss / n_train_batches

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{N_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val_loss: {best_val_loss:.6f})")
        break

model.load_state_dict(best_model_state)
print(f"Best model from epoch {best_epoch} restored.")

# ── 9. Plot Training History ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.axvline(x=best_epoch - 1, color='green', linestyle='--', alpha=0.5, label=f'Best epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.title('LSTM Training History')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('phase6_training_history.png', dpi=150)
print("Training history plot saved to phase6_training_history.png")

# ── 10. Test Set Evaluation ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("Phase 6 - Test Set Evaluation (Last 50 draws)")
print(f"{'='*60}")

model.eval()
with torch.no_grad():
    test_probs_all = model(X_test_t).cpu().numpy()

total_hits_test = 0
n_test_draws = len(test_indices)
draw_predictions = {}
for i in range(len(test_map)):
    draw_idx, num = test_map[i]
    prob = test_probs_all[i]
    if draw_idx not in draw_predictions:
        draw_predictions[draw_idx] = []
    draw_predictions[draw_idx].append((num, prob))

for i, actual_drawn_idx in enumerate(test_indices):
    probs_list = draw_predictions[actual_drawn_idx]
    probs_list.sort(key=lambda x: x[1], reverse=True)
    top6_numbers = sorted([num for num, prob in probs_list[:6]])
    actual = draws[actual_drawn_idx]
    hits = sum(1 for n in top6_numbers if n in actual)
    total_hits_test += hits
    print(f"  Draw {actual_drawn_idx + 1}: Pred {top6_numbers} | Actual {sorted(actual)} | Hits {hits}")

test_hit_rate = total_hits_test / n_test_draws
print(f"\n{'='*60}")
print(f"LSTM Test Hit Rate: {test_hit_rate:.4f} ({total_hits_test}/{n_test_draws})")

# ── 11. Predict Next Draw ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Prediction for next draw (Draw #{n_draws + 1})")
print(f"{'='*60}")

next_draw_idx = n_draws
history_start = max(0, next_draw_idx - SEQ_LEN)
next_X = all_features[history_start:next_draw_idx, :, :]  # (seq_len, 45, 12)
# Transpose to (45, seq_len, 12)
next_X = next_X.transpose(1, 0, 2)
if next_X.shape[1] < SEQ_LEN:
    pad_len = SEQ_LEN - next_X.shape[1]
    pad = np.zeros((45, pad_len, N_FEATURES))
    next_X = np.concatenate([pad, next_X], axis=1)

next_X_norm = normalize(next_X)
next_X_t = torch.FloatTensor(next_X_norm).to(device)

model.eval()
with torch.no_grad():
    next_probs = model(next_X_t).cpu().numpy()

top12_idx = np.argsort(next_probs)[-12:][::-1]
print(f"\nTop-12 numbers by probability:")
for rank, idx in enumerate(top12_idx, 1):
    num = idx + 1
    prob = next_probs[idx]
    print(f"  #{rank:2d}: Num {num:2d} | prob={prob:.4f}")

next_top6_idx = np.argsort(next_probs)[-6:][::-1]
next_top6 = sorted([int(x + 1) for x in next_top6_idx])
print(f"\n>>> Next Draw Recommended Numbers: {next_top6}")

# ── 12. Log to experiment_log.tsv ────────────────────────────────────────
LOG_FILE = "experiment_log.tsv"
log_entry = {
    "phase": 6,
    "description": "Phase 6 - LSTM (2 layers, hidden=64, dropout=0.5, weight_decay=1e-4, early_stop)",
    "train_draws": len(train_indices),
    "test_draws": len(test_indices),
    "features": "12 Phase5 features per number, seq_len=10, shared LSTM over 45 numbers",
    "model": "LotteryLSTM(input=12, hidden=64, layers=2, dropout=0.5) + EarlyStopping",
    "avg_hit_rate": f"{test_hit_rate:.4f}",
    "total_matches": total_hits_test,
    "next_prediction": ",".join(map(str, next_top6)),
}
header = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, sep="\t", mode="a", header=header, index=False)
print(f"\nExperiment log saved to {LOG_FILE}")

# ── 13. Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 6 COMPLETE - SUMMARY")
print(f"{'='*60}")
print(f"  Architecture:          LotteryLSTM (input=12, hidden=64, layers=2)")
print(f"  Regularization:       Dropout=0.5, L2 weight_decay=1e-4, gradient clipping")
print(f"  Early stopping:       patience={patience}, best epoch={best_epoch}")
print(f"  Best Val Loss:        {best_val_loss:.6f}")
print(f"  Test Hit Rate:        {test_hit_rate:.4f} ({total_hits_test}/{n_test_draws})")
print(f"  Total params:         {total_params:,}")
print(f"  Training samples:     {X_train.shape[0]:,}")
print(f"  Next draw prediction: {next_top6}")
print("\nPhase 6 complete.")