#!/usr/bin/env python3
"""
per_method_isolated.py — Train and evaluate independently on each TalkingFace method.

For each of 7 methods:
  Train : real (train IDs) + that method's fakes (train IDs), z-scored
  Eval : real (val+test IDs) + that method's fakes (val+test IDs), 18 identities
  5 seeds, 30 epochs, batch=64, weighted BCE, z-score per waveform

Best HPs from hp_tuning.json:
  1D ResNet: lr=0.001,  wd=0.0005, dropout=0.5
  Transformer: lr=0.0005, wd=0.001,  dropout=0.1

Results → data/results/{M_DD_YYYY}/per_method_isolated.json
Log → data/results/{M_DD_YYYY}/per_method_isolated_log.txt

Resumable: re-run to continue from last completed method/architecture.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS = ["AniTalker", "EchoMimic", "EDTalk", "FLOAT", "IP_LAP", "Real3DPortrait", "SadTalker"]
SEEDS = [42, 7, 123, 999, 2024]
BASELINE = 0.8064

BEST_HPS = {
    "1D ResNet":   {"lr": 0.001,  "wd": 0.0005, "dropout": 0.5},
    "Transformer": {"lr": 0.0005, "wd": 0.001,  "dropout": 0.1},
}


# ── Models ────────────────────────────────────────────────────────────────────

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class Waveform1DResNet(nn.Module):
    def __init__(self, dropout=0.3, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, 7, padding=3, bias=False),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.stage1 = nn.Sequential(BasicBlock1D(c1, c1, dropout=dropout), BasicBlock1D(c1, c1, dropout=dropout))
        self.stage2 = nn.Sequential(BasicBlock1D(c1, c2, stride=2, dropout=dropout), BasicBlock1D(c2, c2, dropout=dropout))
        self.stage3 = nn.Sequential(BasicBlock1D(c2, c3, stride=2, dropout=dropout), BasicBlock1D(c3, c3, dropout=dropout))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c3, 1)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.gap(
            self.stage3(self.stage2(self.stage1(self.stem(x))))
        ).squeeze(-1)).squeeze(-1)


class WaveformTransformer(nn.Module):
    def __init__(self, patch_size=8, d_model=64, nhead=4, num_layers=2, mlp_dim=128, dropout=0.3):
        super().__init__()
        assert 160 % patch_size == 0
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 160 // patch_size + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, mlp_dim, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.patch_embed(x).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1) + self.pos_embed
        return self.fc(self.dropout(self.norm(self.encoder(x)[:, 0]))).squeeze(-1)


ARCH_CLASSES = {
    "1D ResNet":   Waveform1DResNet,
    "Transformer": WaveformTransformer,
}


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def load_waveforms_z(wave_root, df):
    """Load waveforms and apply per-waveform z-score. Returns (X, y)."""
    waves, labels = [], []
    for _, row in df.iterrows():
        path = wave_root / row["class"] / f"{row['video_id']}.npy"
        w = np.load(path).astype(np.float32)
        waves.append(zscore(w))
        labels.append(float(row["class"] == "fake"))
    return np.stack(waves), np.array(labels, dtype=np.float32)


def train_and_eval(model_class, dropout, X_tr, y_tr, X_ev, y_ev, lr, wd, seed=42, epochs=30, batch=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pw = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
    mdl = model_class(dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.from_numpy(X_tr).unsqueeze(1).to(DEVICE)
    yt = torch.from_numpy(y_tr).to(DEVICE)
    Xe = torch.from_numpy(X_ev).unsqueeze(1)

    for _ in range(epochs):
        mdl.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for s in range(0, len(Xt), batch):
            ix = perm[s:s+batch]
            opt.zero_grad()
            crit(mdl(Xt[ix]).reshape(-1), yt[ix]).backward()
            opt.step()

    mdl.eval()
    all_p = []
    with torch.no_grad():
        for s in range(0, len(Xe), 512):
            p = torch.sigmoid(mdl(Xe[s:s+512].to(DEVICE)).reshape(-1)).cpu().numpy()
            all_p.extend(p)

    probs = np.array(all_p)
    return float(roc_auc_score(y_ev, probs)), float(compute_eer(y_ev, probs))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(Path.home() / "rPPG_Detection" / "data"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    wave_root = data_root / "waveforms"
    split_df = pd.read_csv(data_root / "dataset_split.csv")

    today = datetime.now().strftime("%-m_%d_%Y")
    results_dir = data_root / "results" / today
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "per_method_isolated.json"
    log_path = results_dir / "per_method_isolated_log.txt"

    print(f"Device: {DEVICE}")
    print(f"Results dir: {results_dir}")

    # Resume: load existing results
    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = [m for m in results if all(a in results[m] for a in ARCH_CLASSES)]
        print(f"Resuming — methods fully done: {done}")

    # Extract method from video_id prefix
    split_df["method"] = split_df["video_id"].apply(
        lambda x: x.split("__")[0] if "__" in x else None
    )

    # Verify all 7 methods are present
    found = set(split_df[split_df["class"] == "fake"]["method"].unique())
    missing = set(METHODS) - found
    if missing:
        print(f"WARNING: methods not found in split: {missing}")

    # Load real waveforms once (shared across all methods)
    train_real = split_df[(split_df["split"] == "train") & (split_df["class"] == "real")]
    eval_real = split_df[split_df["split"].isin(["val", "test"]) & (split_df["class"] == "real")]
    print(f"\nLoading real waveforms (train={len(train_real)}, eval={len(eval_real)})...")
    X_real_tr, y_real_tr = load_waveforms_z(wave_root, train_real)
    X_real_ev, y_real_ev = load_waveforms_z(wave_root, eval_real)

    with open(log_path, "a") as log:
        log.write(f"\n=== Started {datetime.now()} ===\n")
        log.write(f"Real train: {len(train_real)}, Real eval: {len(eval_real)}\n")

    for method in METHODS:
        already_done = results.get(method, {})
        archs_needed = [a for a in ARCH_CLASSES if a not in already_done]
        if not archs_needed:
            print(f"\n[SKIP] {method} — all architectures done")
            continue

        t_method = time.time()
        print(f"\n{'='*60}")
        print(f"METHOD: {method}")
        print(f"{'='*60}")

        # Method-specific fakes
        m_train = split_df[(split_df["split"] == "train") & (split_df["class"] == "fake") & (split_df["method"] == method)]
        m_eval  = split_df[split_df["split"].isin(["val", "test"]) & (split_df["class"] == "fake") & (split_df["method"] == method)]

        print(f"Fake waveforms — train: {len(m_train)}  eval: {len(m_eval)}")
        X_fake_tr, y_fake_tr = load_waveforms_z(wave_root, m_train)
        X_fake_ev, y_fake_ev = load_waveforms_z(wave_root, m_eval)

        X_tr = np.concatenate([X_real_tr, X_fake_tr])
        y_tr = np.concatenate([y_real_tr, y_fake_tr])
        X_ev = np.concatenate([X_real_ev, X_fake_ev])
        y_ev = np.concatenate([y_real_ev, y_fake_ev])

        n_real_tr = int((y_tr == 0).sum())
        n_fake_tr = int((y_tr == 1).sum())
        print(f"Train: {n_real_tr} real + {n_fake_tr} fake  "
              f"(ratio {n_fake_tr/max(n_real_tr,1):.2f})")
        print(f"Eval:  {int((y_ev==0).sum())} real + {int((y_ev==1).sum())} fake")

        if method not in results:
            results[method] = {}

        for arch_name in archs_needed:
            cls = ARCH_CLASSES[arch_name]
            hp = BEST_HPS[arch_name]
            t0 = time.time()
            print(f"\n  [{arch_name}]  lr={hp['lr']} wd={hp['wd']} dropout={hp['dropout']}")

            seed_aucs, seed_eers = [], []
            for seed in SEEDS:
                auc, eer = train_and_eval(
                    cls, hp["dropout"], X_tr, y_tr, X_ev, y_ev,
                    lr=hp["lr"], wd=hp["wd"], seed=seed,
                    epochs=args.epochs, batch=args.batch
                )
                seed_aucs.append(auc)
                seed_eers.append(eer)
                print(f"    seed={seed}  AUC={auc:.4f}  EER={eer*100:.1f}%")

            mean_auc = float(np.mean(seed_aucs))
            std_auc = float(np.std(seed_aucs))
            delta = mean_auc - BASELINE
            elapsed = time.time() - t0
            print(f"  → {mean_auc:.4f} ± {std_auc:.4f}  "
                  f"delta={delta:+.4f}  [{elapsed:.0f}s]")

            results[method][arch_name] = {
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_eer": float(np.mean(seed_eers)),
                "seed_aucs": {str(s): float(a) for s, a in zip(SEEDS, seed_aucs)},
                "n_train_real": n_real_tr,
                "n_train_fake": n_fake_tr,
            }

            # Save after every architecture (resumability)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        with open(log_path, "a") as log:
            log.write(f"  {method}: done in {time.time()-t_method:.0f}s\n")

        print(f"\n  [{method} complete — {time.time()-t_method:.0f}s]")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  PER-METHOD ISOLATED EVALUATION  (18-identity, 5 seeds, vs baseline={BASELINE})")
    print(f"{'='*78}")
    print(f"  {'Method':<16} {'1D ResNet AUC':>18} {'Transformer AUC':>18} {'vs baseline':>12}")
    print("  " + "-"*66)

    resnet_aucs, transformer_aucs = [], []
    for method in METHODS:
        r = results[method]["1D ResNet"]
        t = results[method]["Transformer"]
        avg = (r["mean_auc"] + t["mean_auc"]) / 2
        delta = avg - BASELINE
        print(f"  {method:<16} "
              f"{r['mean_auc']:.4f} ± {r['std_auc']:.4f}  "
              f"{t['mean_auc']:.4f} ± {t['std_auc']:.4f}  "
              f"{delta:>+.4f}")
        resnet_aucs.append(r["mean_auc"])
        transformer_aucs.append(t["mean_auc"])

    print("  " + "-"*66)
    print(f"  {'Mean':<16} "
          f"{np.mean(resnet_aucs):.4f}              "
          f"{np.mean(transformer_aucs):.4f}")

    with open(log_path, "a") as log:
        log.write(f"=== Completed {datetime.now()} ===\n")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
