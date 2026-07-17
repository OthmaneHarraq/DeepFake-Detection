#!/usr/bin/env python3
"""
run_experiments.py — All Phase 4 training and evaluation experiments.

Runs in sequence:
    --experiment technique    Run A (z-score) + Run B (augment) isolation
    --experiment hp           Run C one-at-a-time HP tuning
    --experiment test         Evaluate all configs on locked test set
    --experiment diagnostics  Per-identity AUC, rebalance check, multi-seed
    --experiment eval18       18-identity evaluation (val + test merged)
    --experiment permethod    Per-method AUC breakdown across 7 TF generators
    --experiment all          Run everything in sequence (default)

Results are saved to --data-root/results/ as JSON files.

Usage:
    python3 run_experiments.py --data-root ~/data
    python3 run_experiments.py --data-root ~/data --experiment permethod
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 experiments")
    p.add_argument("--data-root",   default=str(Path.home() / "data"))
    p.add_argument("--experiment",  default="all",
                   choices=["technique", "hp", "test", "diagnostics",
                             "eval18", "permethod", "all"])
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch",       type=int, default=64)
    p.add_argument("--n-folds",     type=int, default=5)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model architectures
# ─────────────────────────────────────────────────────────────────────────────
class Waveform1DCNN(nn.Module):
    def __init__(self, dropout=0.3, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Conv1d(1, c1, 9, padding=4); self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, 7, padding=3); self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, 5, padding=2); self.bn3 = nn.BatchNorm1d(c3)
        self.pool1 = nn.MaxPool1d(2); self.pool2 = nn.MaxPool1d(2)
        self.gap  = nn.AdaptiveAvgPool1d(1); self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(c3, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(self.drop(self.gap(x).squeeze(-1))).squeeze(-1)


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )
    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class Waveform1DResNet(nn.Module):
    def __init__(self, dropout=0.3, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        self.stem   = nn.Sequential(
            nn.Conv1d(1, c1, 7, padding=3, bias=False),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.stage1 = nn.Sequential(BasicBlock1D(c1, c1, dropout=dropout),
                                    BasicBlock1D(c1, c1, dropout=dropout))
        self.stage2 = nn.Sequential(BasicBlock1D(c1, c2, stride=2, dropout=dropout),
                                    BasicBlock1D(c2, c2, dropout=dropout))
        self.stage3 = nn.Sequential(BasicBlock1D(c2, c3, stride=2, dropout=dropout),
                                    BasicBlock1D(c3, c3, dropout=dropout))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(c3, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.gap(self.stage3(self.stage2(self.stage1(self.stem(x))))).squeeze(-1)).squeeze(-1)


class WaveformTransformer(nn.Module):
    def __init__(self, patch_size=8, d_model=64, nhead=4, num_layers=2,
                 mlp_dim=128, dropout=0.3):
        super().__init__()
        assert 160 % patch_size == 0
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, 160 // patch_size + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer    = nn.TransformerEncoderLayer(d_model, nhead, mlp_dim, dropout,
                                                  activation="gelu", batch_first=True,
                                                  norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, 1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.patch_embed(x).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1) + self.pos_embed
        return self.fc(self.dropout(self.norm(self.encoder(x)[:, 0]))).squeeze(-1)


class ToeplitzViT(nn.Module):
    def __init__(self, image_size=160, patch_size=16, d_model=64, nhead=4,
                 num_layers=2, mlp_dim=128, dropout=0.3):
        super().__init__()
        n = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(1, d_model, patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer    = nn.TransformerEncoderLayer(d_model, nhead, mlp_dim, dropout,
                                                  activation="gelu", batch_first=True,
                                                  norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, 1)
    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1) + self.pos_embed
        return self.fc(self.dropout(self.norm(self.encoder(x)[:, 0]))).squeeze(-1)


ARCH_CLASSES = {
    "1D CNN":       (Waveform1DCNN,       "X"),
    "1D ResNet":    (Waveform1DResNet,    "X"),
    "Transformer":  (WaveformTransformer, "X"),
    "Toeplitz ViT": (ToeplitzViT,         "T"),
}

BEST_CONFIGS = {
    "1D CNN":       {"lr": 5e-4, "wd": 1e-3,  "dropout": 0.3},
    "1D ResNet":    {"lr": 1e-3, "wd": 5e-4,  "dropout": 0.5},
    "Transformer":  {"lr": 1e-3, "wd": 1e-4,  "dropout": 0.3},
    "Toeplitz ViT": {"lr": 1e-3, "wd": 5e-4,  "dropout": 0.5},
}

HP_SWEEP = {
    "lr":      [5e-4, 1e-3, 2e-3],
    "wd":      [1e-4, 5e-4, 1e-3],
    "dropout": [0.1,  0.3,  0.5],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def augment_waveform(x, noise_std=0.05):
    x = x + np.random.normal(0, noise_std, x.shape).astype(np.float32)
    mask_len = np.random.randint(10, 25)
    start    = np.random.randint(0, len(x) - mask_len)
    x        = x.copy(); x[start:start + mask_len] = 0.0
    return x

def build_toeplitz(X):
    L = X.shape[1]
    idx = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    return X[:, idx].astype(np.float32)

def load_waveforms(wave_root, df, label_col="class"):
    """Load all waveforms from a dataframe. Returns (X, y)."""
    waves, labels = [], []
    for _, row in df.iterrows():
        path = wave_root / row[label_col] / f"{row['video_id']}.npy"
        waves.append(np.load(path).astype(np.float32))
        labels.append(float(row["class"] == "fake"))
    return np.stack(waves), np.array(labels, dtype=np.float32)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i   = np.argmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_X(X_plain, X_zscore, T_plain, T_zscore, arch_name, use_zscore):
    _, inp = ARCH_CLASSES[arch_name]
    if inp == "X":
        return X_zscore if use_zscore else X_plain
    return T_zscore if use_zscore else T_plain


def run_cv(model_class, dropout, X_data, y_data, groups,
           lr, wd, augment=False, epochs=30, batch=64, n_folds=5):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    all_probs, all_labels, fold_aucs = [], [], []
    for tr_idx, va_idx in sgkf.split(X_data, y_data, groups=groups):
        X_tr, X_va = X_data[tr_idx], X_data[va_idx]
        y_tr, y_va = y_data[tr_idx], y_data[va_idx]
        if augment:
            X_tr = np.stack([augment_waveform(w) for w in X_tr])
        pw    = torch.tensor([(y_tr==0).sum() / max((y_tr==1).sum(), 1)]).to(DEVICE)
        mdl   = model_class(dropout=dropout).to(DEVICE)
        opt   = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
        crit  = nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt    = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
        yt    = torch.from_numpy(y_tr).to(DEVICE)
        Xv    = torch.from_numpy(X_va).unsqueeze(1).to(DEVICE)
        best_auc, best_probs = -1, None
        for _ in range(epochs):
            mdl.train()
            perm = torch.randperm(len(Xt), device=DEVICE)
            for s in range(0, len(Xt), batch):
                ix = perm[s:s+batch]
                opt.zero_grad()
                crit(mdl(Xt[ix]).reshape(-1), yt[ix]).backward()
                opt.step()
            mdl.eval()
            with torch.no_grad():
                probs = torch.sigmoid(mdl(Xv).reshape(-1)).cpu().numpy()
            auc = roc_auc_score(y_va, probs)
            if auc > best_auc:
                best_auc, best_probs = auc, probs.copy()
        all_probs.extend(best_probs)
        all_labels.extend(y_va)
        fold_aucs.append(best_auc)
    return np.array(fold_aucs), np.array(all_probs), np.array(all_labels)


def train_full(model_class, dropout, X_tr, y_tr, lr, wd,
               augment=False, epochs=30, batch=64, seed=42):
    torch.manual_seed(seed)
    pw    = torch.tensor([(y_tr==0).sum() / max((y_tr==1).sum(), 1)]).to(DEVICE)
    mdl   = model_class(dropout=dropout).to(DEVICE)
    opt   = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)
    X_use = np.stack([augment_waveform(w) for w in X_tr]) if augment else X_tr
    Xt    = torch.from_numpy(X_use).unsqueeze(1).to(DEVICE)
    yt    = torch.from_numpy(y_tr).to(DEVICE)
    for _ in range(epochs):
        mdl.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for s in range(0, len(Xt), batch):
            ix = perm[s:s+batch]
            opt.zero_grad()
            crit(mdl(Xt[ix]).reshape(-1), yt[ix]).backward()
            opt.step()
    return mdl


def get_probs(mdl, X, batch=512):
    mdl.eval()
    all_p = []
    Xt = torch.from_numpy(X).unsqueeze(1)
    with torch.no_grad():
        for s in range(0, len(Xt), batch):
            p = torch.sigmoid(mdl(Xt[s:s+batch].to(DEVICE)).reshape(-1)).cpu().numpy()
            all_p.extend(p)
    return np.array(all_p)


def eval_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(y_true, probs)
    eer   = compute_eer(y_true, probs)
    rpt   = classification_report(y_true, preds,
                                  target_names=["real", "fake"], output_dict=True)
    return {"auc": auc, "eer": eer,
            "p_real": rpt["real"]["precision"],
            "r_real": rpt["real"]["recall"],
            "f1_real": rpt["real"]["f1-score"],
            "p_fake": rpt["fake"]["precision"],
            "f1_fake": rpt["fake"]["f1-score"]}


def save_results(results_dir, name, data):
    path = results_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Data setup (shared across experiments)
# ─────────────────────────────────────────────────────────────────────────────
def setup_data(args):
    data_root    = Path(args.data_root)
    wave_root    = data_root / "waveforms"
    split_df     = pd.read_csv(data_root / "dataset_split.csv")
    results_dir  = data_root / "results"
    results_dir.mkdir(exist_ok=True)

    train_real = split_df[(split_df["split"] == "train") & (split_df["class"] == "real")]
    train_fake = split_df[(split_df["split"] == "train") & (split_df["class"] == "fake")]

    print("Loading training waveforms...")
    X_real_t = np.stack([np.load(wave_root/"real"/f"{r['video_id']}.npy").astype(np.float32)
                         for _, r in train_real.iterrows()])
    X_fake_t = np.stack([np.load(wave_root/"fake"/f"{r['video_id']}.npy").astype(np.float32)
                         for _, r in train_fake.iterrows()])

    X_plain  = np.concatenate([X_real_t, X_fake_t])
    y        = np.array([0.0]*len(X_real_t) + [1.0]*len(X_fake_t), dtype=np.float32)
    groups   = np.concatenate([
        train_real["video_id"].str.extract(r"(id\d+)")[0].values,
        train_fake["video_id"].str.extract(r"(id\d+)")[0].values,
    ])

    X_zscore = np.stack([zscore(x) for x in X_plain])
    T_plain  = build_toeplitz(X_plain)
    T_zscore = build_toeplitz(X_zscore)

    print(f"Training: {int((y==0).sum())} real + {int((y==1).sum())} fake "
          f"| {len(np.unique(groups))} identity groups")

    return dict(
        data_root=data_root, wave_root=wave_root, split_df=split_df,
        results_dir=results_dir,
        X_plain=X_plain, X_zscore=X_zscore, T_plain=T_plain, T_zscore=T_zscore,
        y=y, groups=groups
    )


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A/B: Technique isolation
# ─────────────────────────────────────────────────────────────────────────────
def run_technique_isolation(ctx, args):
    print(f"\n{'='*70}")
    print("  EXPERIMENT A/B — Technique Isolation (z-score only, augment only)")
    print(f"{'='*70}")
    results = {"A_zscore_only": {}, "B_augment_only": {}}
    total   = len(ARCH_CLASSES) * 2
    done    = 0

    for label, use_zscore, augment in [
        ("A_zscore_only", True, False),
        ("B_augment_only", False, True),
    ]:
        print(f"\n  RUN {label}:")
        for arch_name, (cls, _) in ARCH_CLASSES.items():
            cfg  = BEST_CONFIGS[arch_name]
            X_in = get_X(ctx["X_plain"], ctx["X_zscore"],
                         ctx["T_plain"], ctx["T_zscore"],
                         arch_name, use_zscore)
            t0 = time.time()
            fold_aucs, probs, labels = run_cv(
                cls, cfg["dropout"], X_in, ctx["y"], ctx["groups"],
                lr=cfg["lr"], wd=cfg["wd"], augment=augment,
                epochs=args.epochs, batch=args.batch, n_folds=args.n_folds
            )
            done += 1
            preds = (probs >= 0.5).astype(int)
            rpt   = classification_report(labels, preds,
                                          target_names=["real","fake"], output_dict=True)
            results[label][arch_name] = {
                "mean_auc": float(fold_aucs.mean()),
                "std_auc":  float(fold_aucs.std()),
                "eer":      float(compute_eer(labels, probs)),
                "p_real":   float(rpt["real"]["precision"]),
                "f1_real":  float(rpt["real"]["f1-score"]),
            }
            print(f"    {arch_name:<16} AUC: {fold_aucs.mean():.4f}±{fold_aucs.std():.4f}  "
                  f"P-Real: {rpt['real']['precision']:.4f}  "
                  f"EER: {compute_eer(labels,probs)*100:.1f}%  "
                  f"({done}/{total})  [{time.time()-t0:.0f}s]")

    save_results(ctx["results_dir"], "technique_isolation", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C: HP tuning
# ─────────────────────────────────────────────────────────────────────────────
def run_hp_tuning(ctx, args):
    print(f"\n{'='*70}")
    print("  EXPERIMENT C — One-at-a-Time HP Tuning")
    print(f"{'='*70}")
    results  = {}
    total    = len(ARCH_CLASSES) * sum(len(v) for v in HP_SWEEP.values())
    done     = 0

    for arch_name, (cls, _) in ARCH_CLASSES.items():
        base = BEST_CONFIGS[arch_name]
        X_in = get_X(ctx["X_plain"], ctx["X_zscore"],
                     ctx["T_plain"], ctx["T_zscore"],
                     arch_name, False)
        results[arch_name] = {}
        print(f"\n  {arch_name}  "
              f"[base: lr={base['lr']}, wd={base['wd']}, dropout={base['dropout']}]")

        for hp_name, values in HP_SWEEP.items():
            results[arch_name][hp_name] = {}
            for hp_val in values:
                cfg = base.copy(); cfg[hp_name] = hp_val
                is_base = (hp_val == base[hp_name])
                t0 = time.time()
                fold_aucs, probs, labels_ = run_cv(
                    cls, cfg["dropout"], X_in, ctx["y"], ctx["groups"],
                    lr=cfg["lr"], wd=cfg["wd"], augment=False,
                    epochs=args.epochs, batch=args.batch, n_folds=args.n_folds
                )
                done += 1
                preds = (probs >= 0.5).astype(int)
                rpt   = classification_report(labels_, preds,
                                              target_names=["real","fake"], output_dict=True)
                results[arch_name][hp_name][str(hp_val)] = {
                    "mean_auc": float(fold_aucs.mean()),
                    "std_auc":  float(fold_aucs.std()),
                    "eer":      float(compute_eer(labels_, probs)),
                    "p_real":   float(rpt["real"]["precision"]),
                    "f1_real":  float(rpt["real"]["f1-score"]),
                }
                marker = " [BASE]" if is_base else ""
                print(f"    {hp_name}={hp_val}{marker:<8}  "
                      f"AUC: {fold_aucs.mean():.4f}±{fold_aucs.std():.4f}  "
                      f"P-Real: {rpt['real']['precision']:.4f}  "
                      f"({done}/{total})  [{time.time()-t0:.0f}s]")

    save_results(ctx["results_dir"], "hp_tuning", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test set evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_test_evaluation(ctx, args):
    print(f"\n{'='*70}")
    print("  TEST SET EVALUATION — All configs on 9-identity locked test set")
    print(f"{'='*70}")

    wave_root = ctx["wave_root"]
    split_df  = ctx["split_df"]
    test_df   = split_df[split_df["split"] == "test"]

    print("Loading test waveforms...")
    X_test_plain = np.stack([
        np.load(wave_root / r["class"] / f"{r['video_id']}.npy").astype(np.float32)
        for _, r in test_df.iterrows()
    ])
    y_test        = test_df["label"].values.astype(np.float32)
    X_test_zscore = np.stack([zscore(x) for x in X_test_plain])
    T_test_plain  = build_toeplitz(X_test_plain)
    T_test_zscore = build_toeplitz(X_test_zscore)

    print(f"Test set: {int((y_test==0).sum())} real + {int((y_test==1).sum())} fake")

    def get_test_X(arch_name, use_zscore):
        _, inp = ARCH_CLASSES[arch_name]
        if inp == "X":
            return X_test_zscore if use_zscore else X_test_plain
        return T_test_zscore if use_zscore else T_test_plain

    results = {"A_zscore_only": {}, "B_augment_only": {}, "C_hp_tuning": {}}
    total   = len(ARCH_CLASSES) * 2 + len(ARCH_CLASSES) * sum(len(v) for v in HP_SWEEP.values())
    done    = 0

    for label, use_zscore, augment in [
        ("A_zscore_only", True, False),
        ("B_augment_only", False, True),
    ]:
        print(f"\n  TEST {label}:")
        for arch_name, (cls, _) in ARCH_CLASSES.items():
            cfg  = BEST_CONFIGS[arch_name]
            X_in = get_X(ctx["X_plain"], ctx["X_zscore"],
                         ctx["T_plain"], ctx["T_zscore"], arch_name, use_zscore)
            Xte  = get_test_X(arch_name, use_zscore)
            mdl  = train_full(cls, cfg["dropout"], X_in, ctx["y"],
                               lr=cfg["lr"], wd=cfg["wd"], augment=augment,
                               epochs=args.epochs, batch=args.batch)
            probs = get_probs(mdl, Xte)
            m     = eval_metrics(y_test, probs)
            results[label][arch_name] = m
            done += 1
            print(f"    {arch_name:<16} AUC: {m['auc']:.4f}  EER: {m['eer']*100:.1f}%  "
                  f"P-Real: {m['p_real']:.4f}  ({done}/{total})")

    print(f"\n  TEST C — HP Tuning:")
    for arch_name, (cls, _) in ARCH_CLASSES.items():
        base = BEST_CONFIGS[arch_name]
        X_in = get_X(ctx["X_plain"], ctx["X_zscore"],
                     ctx["T_plain"], ctx["T_zscore"], arch_name, False)
        Xte  = get_test_X(arch_name, False)
        results["C_hp_tuning"][arch_name] = {}
        print(f"\n  {arch_name}")
        for hp_name, values in HP_SWEEP.items():
            results["C_hp_tuning"][arch_name][hp_name] = {}
            for hp_val in values:
                cfg = base.copy(); cfg[hp_name] = hp_val
                mdl = train_full(cls, cfg["dropout"], X_in, ctx["y"],
                                  lr=cfg["lr"], wd=cfg["wd"], augment=False,
                                  epochs=args.epochs, batch=args.batch)
                probs = get_probs(mdl, Xte)
                m     = eval_metrics(y_test, probs)
                results["C_hp_tuning"][arch_name][hp_name][str(hp_val)] = m
                done += 1
                is_base = (hp_val == base[hp_name])
                marker  = " [BASE]" if is_base else ""
                print(f"    {hp_name}={hp_val}{marker:<8}  AUC: {m['auc']:.4f}  "
                      f"EER: {m['eer']*100:.1f}%  P-Real: {m['p_real']:.4f}  ({done}/{total})")

    save_results(ctx["results_dir"], "test_evaluation", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostics(ctx, args):
    print(f"\n{'='*70}")
    print("  DIAGNOSTICS — Per-identity, rebalance, multi-seed (1D ResNet)")
    print(f"{'='*70}")

    wave_root = ctx["wave_root"]
    split_df  = ctx["split_df"]
    test_df   = split_df[split_df["split"] == "test"].copy()
    test_df["identity"] = test_df["video_id"].str.extract(r"(id\d+)")[0]

    X_test = np.stack([
        np.load(wave_root / r["class"] / f"{r['video_id']}.npy").astype(np.float32)
        for _, r in test_df.iterrows()
    ])
    y_test     = test_df["label"].values.astype(np.float32)
    identities = test_df["identity"].values
    test_ids   = sorted(test_df["identity"].unique())

    print(f"Test: {int((y_test==0).sum())} real + {int((y_test==1).sum())} fake "
          f"| {len(test_ids)} identities")

    # Train base model
    print("\nTraining base model (seed=42, dropout=0.3)...")
    mdl = train_full(Waveform1DResNet, 0.3, ctx["X_plain"], ctx["y"],
                      lr=1e-3, wd=5e-4, epochs=args.epochs, batch=args.batch, seed=42)
    probs_full = get_probs(mdl, X_test)
    print(f"Full test AUC: {roc_auc_score(y_test, probs_full):.4f}")

    # Per-identity breakdown
    print(f"\n  {'Identity':<12} {'Real':>6} {'Fake':>6} {'AUC':>8} {'EER':>7}")
    per_id = {}
    for identity in test_ids:
        mask = identities == identity
        y_id, p_id = y_test[mask], probs_full[mask]
        if (y_id == 0).sum() == 0 or (y_id == 1).sum() == 0:
            continue
        auc = roc_auc_score(y_id, p_id)
        eer = compute_eer(y_id, p_id)
        per_id[identity] = {"auc": auc, "eer": eer,
                             "n_real": int((y_id==0).sum()),
                             "n_fake": int((y_id==1).sum())}
        flag = " ← outlier" if auc < 0.55 else ""
        print(f"  {identity:<12} {int((y_id==0).sum()):>6} {int((y_id==1).sum()):>6} "
              f"{auc:>8.4f} {eer*100:>6.1f}%{flag}")

    auc_vals = [v["auc"] for v in per_id.values()]
    print(f"\n  AUC range: {min(auc_vals):.4f} – {max(auc_vals):.4f}")
    print(f"  Std:       {np.std(auc_vals):.4f}  (>0.08 = identity-specific difficulty)")

    # Rebalance check
    np.random.seed(42)
    real_idx    = np.where(y_test == 0)[0]
    fake_sample = np.random.choice(np.where(y_test == 1)[0],
                                   size=len(real_idx), replace=False)
    bal_idx  = np.concatenate([real_idx, fake_sample])
    auc_bal  = roc_auc_score(y_test[bal_idx], probs_full[bal_idx])
    auc_full = roc_auc_score(y_test, probs_full)
    delta    = auc_bal - auc_full
    print(f"\n  Rebalance delta: {delta:+.4f}  (>+0.02 = imbalance is a factor)")

    # Multi-seed
    print("\n  Multi-seed stability (1D ResNet, dropout=0.3, 5 seeds):")
    seed_aucs = []
    for seed in [42, 7, 123, 999, 2024]:
        m_s   = train_full(Waveform1DResNet, 0.3, ctx["X_plain"], ctx["y"],
                            lr=1e-3, wd=5e-4, epochs=args.epochs,
                            batch=args.batch, seed=seed)
        p_s   = get_probs(m_s, X_test)
        auc_s = roc_auc_score(y_test, p_s)
        eer_s = compute_eer(y_test, p_s)
        seed_aucs.append(auc_s)
        print(f"    seed={seed:<6}  AUC: {auc_s:.4f}  EER: {eer_s*100:.1f}%")

    mean_auc, std_auc = float(np.mean(seed_aucs)), float(np.std(seed_aucs))
    print(f"\n  Mean ± std: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  → Paper-reported (9-identity, 5 seeds): {mean_auc:.4f} ± {std_auc:.4f}")

    results = {
        "per_identity": per_id,
        "rebalance_delta": float(delta),
        "seed_aucs": {str(s): float(a) for s, a in zip([42,7,123,999,2024], seed_aucs)},
        "mean_auc": mean_auc, "std_auc": std_auc,
        "id_auc_std": float(np.std(auc_vals)),
    }
    save_results(ctx["results_dir"], "diagnostics", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 18-identity evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_eval18(ctx, args):
    print(f"\n{'='*70}")
    print("  18-IDENTITY EVALUATION — Val + Test merged (1D ResNet)")
    print(f"{'='*70}")

    wave_root = ctx["wave_root"]
    split_df  = ctx["split_df"]
    eval_df   = split_df[split_df["split"].isin(["val", "test"])].copy()
    eval_df["identity"] = eval_df["video_id"].str.extract(r"(id\d+)")[0]

    print("Loading val + test waveforms...")
    X_eval = np.stack([
        np.load(wave_root / r["class"] / f"{r['video_id']}.npy").astype(np.float32)
        for _, r in eval_df.iterrows()
    ])
    y_eval       = eval_df["label"].values.astype(np.float32)
    eval_ids_arr = eval_df["identity"].values
    eval_ids     = sorted(eval_df["identity"].unique())
    print(f"Combined: {int((y_eval==0).sum())} real + {int((y_eval==1).sum())} fake "
          f"| {len(eval_ids)} identities")

    # Multi-seed
    print("\n  1D ResNet dropout=0.3, 5 seeds:")
    seed_results = []
    for seed in [42, 7, 123, 999, 2024]:
        mdl   = train_full(Waveform1DResNet, 0.3, ctx["X_plain"], ctx["y"],
                            lr=1e-3, wd=5e-4, epochs=args.epochs,
                            batch=args.batch, seed=seed)
        probs = get_probs(mdl, X_eval)
        auc   = roc_auc_score(y_eval, probs)
        eer   = compute_eer(y_eval, probs)
        seed_results.append({"seed": seed, "auc": float(auc), "eer": float(eer), "probs": probs})
        print(f"  seed={seed:<6}  AUC: {auc:.4f}  EER: {eer*100:.1f}%")

    mean18 = float(np.mean([r["auc"] for r in seed_results]))
    std18  = float(np.std([r["auc"] for r in seed_results]))
    print(f"\n  Mean ± std:  {mean18:.4f} ± {std18:.4f}")
    print(f"  → Paper-reported AUC: {mean18:.4f} ± {std18:.4f}")

    # Per-identity (seed=42)
    best_probs = seed_results[0]["probs"]
    print("\n  Per-identity AUC (seed=42):")
    per_id18 = {}
    for identity in eval_ids:
        mask   = eval_ids_arr == identity
        y_id   = y_eval[mask]; p_id = best_probs[mask]
        split  = eval_df[eval_df["identity"] == identity]["split"].iloc[0]
        if (y_id==0).sum() == 0 or (y_id==1).sum() == 0:
            continue
        auc = roc_auc_score(y_id, p_id)
        per_id18[identity] = {"auc": float(auc), "split": split,
                               "n_real": int((y_id==0).sum()),
                               "n_fake": int((y_id==1).sum())}
        flag = " ← outlier" if auc < 0.55 else ""
        print(f"  {identity:<10} {split:>6} {int((y_id==0).sum()):>4} real "
              f"{int((y_id==1).sum()):>4} fake  AUC {auc:.4f}{flag}")

    auc_vals = [v["auc"] for v in per_id18.values()]
    print(f"\n  AUC std across 18 identities: {np.std(auc_vals):.4f}")

    results = {
        "mean_auc": mean18, "std_auc": std18,
        "per_identity": per_id18,
        "id_auc_std": float(np.std(auc_vals)),
        "seed_aucs": {str(r["seed"]): r["auc"] for r in seed_results},
    }
    save_results(ctx["results_dir"], "eval18", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-method evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_permethod(ctx, args):
    print(f"\n{'='*70}")
    print("  PER-METHOD EVALUATION — 7 TF generation methods")
    print(f"{'='*70}")

    METHODS      = ["AniTalker", "EchoMimic", "EDTalk", "FLOAT",
                    "IP_LAP", "Real3DPortrait", "SadTalker"]
    METHOD_SIZES = [357, 357, 357, 357, 357, 357, 358]

    wave_root = ctx["wave_root"]
    split_df  = ctx["split_df"]

    # All real waveforms
    all_real = split_df[split_df["class"] == "real"].drop_duplicates("video_id")
    print(f"Loading {len(all_real)} real waveforms...")
    X_real = np.stack([np.load(wave_root/"real"/f"{r['video_id']}.npy").astype(np.float32)
                       for _, r in all_real.iterrows()])
    y_real = np.zeros(len(X_real), dtype=np.float32)

    # All fake waveforms sorted
    all_fakes = split_df[split_df["class"] == "fake"].sort_values("video_id").reset_index(drop=True)
    print(f"Loading {len(all_fakes)} fake waveforms...")
    X_fake = np.stack([np.load(wave_root/"fake"/f"{r['video_id']}.npy").astype(np.float32)
                       for _, r in all_fakes.iterrows()])

    # Extract method directly from video_id prefix (e.g. "AniTalker__id0_0000_...")
    all_fakes["method"] = all_fakes["video_id"].apply(
        lambda x: x.split("__")[0] if "__" in x else "unknown"
    )

    print("\nMethod assignment:")
    for method, size in zip(METHODS, METHOD_SIZES):
        count = (all_fakes["method"] == method).sum()
        print(f"  {method:<20} {count:>5} fakes")

    # Train model
    print("\nTraining 1D ResNet (dropout=0.3, seed=42)...")
    X_train_pm = ctx["X_plain"]
    y_train_pm = ctx["y"]

    mdl = train_full(Waveform1DResNet, 0.3, X_train_pm, y_train_pm,
                      lr=1e-3, wd=5e-4, epochs=args.epochs, batch=args.batch, seed=42)

    # Inference on all waveforms
    probs_real = get_probs(mdl, X_real)
    probs_fake = get_probs(mdl, X_fake)
    print(f"Real scores: mean={probs_real.mean():.3f} | "
          f"Fake scores: mean={probs_fake.mean():.3f}")

    # Per-method metrics
    print(f"\n  {'Method':<22} {'N-Fake':>7} {'AUC':>8} {'EER':>7} "
          f"{'P-Real':>8} {'R-Real':>8}")
    print("  " + "─"*68)

    method_results = {}
    for method in METHODS:
        mask      = all_fakes["method"] == method
        p_method  = probs_fake[mask.values]
        n_fake    = mask.sum()
        y_comb    = np.concatenate([y_real, np.ones(n_fake, dtype=np.float32)])
        p_comb    = np.concatenate([probs_real, p_method])
        m         = eval_metrics(y_comb, p_comb)
        method_results[method] = {**m, "n_fake": int(n_fake)}
        print(f"  {method:<22} {n_fake:>7} {m['auc']:>8.4f} {m['eer']*100:>6.1f}% "
              f"{m['p_real']:>8.4f} {m['r_real']:>8.4f}")

    aucs = [method_results[m]["auc"] for m in METHODS]
    print("  " + "─"*68)
    print(f"  {'Mean':<22} {'':>7} {np.mean(aucs):>8.4f} "
          f"{np.mean([method_results[m]['eer'] for m in METHODS])*100:>6.1f}%")
    print(f"\n  AUC spread: {max(aucs)-min(aucs):.4f}  "
          f"({min(METHODS, key=lambda m: method_results[m]['auc'])} hardest, "
          f"{max(METHODS, key=lambda m: method_results[m]['auc'])} easiest)")

    save_results(ctx["results_dir"], "permethod", method_results)
    return method_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print(f"Device: {DEVICE}")
    print(f"Data root: {args.data_root}")
    print(f"Experiment: {args.experiment}")

    ctx = setup_data(args)
    exp = args.experiment

    if exp in ("technique", "all"):
        run_technique_isolation(ctx, args)

    if exp in ("hp", "all"):
        run_hp_tuning(ctx, args)

    if exp in ("test", "all"):
        run_test_evaluation(ctx, args)

    if exp in ("diagnostics", "all"):
        run_diagnostics(ctx, args)

    if exp in ("eval18", "all"):
        run_eval18(ctx, args)

    if exp in ("permethod", "all"):
        run_permethod(ctx, args)

    print(f"\n=== All done. Results in {Path(args.data_root) / 'results'} ===")


if __name__ == "__main__":
    main()