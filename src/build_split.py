#!/usr/bin/env python3
"""
build_split.py — Build subject-independent dataset splits and temporal crop manifest.

Reads manifest.csv (written by extract_waveforms.py), assigns celebrity
identities by parsing the id[0-9]+ prefix from video_id, and partitions the 59
identities into 41 train / 9 val / 9 test with zero overlap.

Also generates temporal_crops_manifest.csv: for each training real video,
extracts multiple 160-frame window indices at stride s=60 for class balancing.

Usage:
    python3 build_split.py --data-root ~/data
    python3 build_split.py --data-root ~/data --seed 42 --stride 60
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed test and val identity sets — same as Colab experiments for comparability
# These 9 test identities are locked and must never change between experiments
TEST_IDS = {"id0", "id4", "id6", "id11", "id13", "id16", "id23", "id27", "id54"}
VAL_IDS  = {"id8", "id12", "id19", "id22", "id34", "id42", "id44", "id52", "id56"}


def parse_args():
    p = argparse.ArgumentParser(description="Build subject-independent splits")
    p.add_argument("--data-root", default=str(Path.home() / "data"))
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--stride", type=int, default=60,
                   help="Stride for temporal crop windows (default: 60)")
    p.add_argument("--n-frames", type=int, default=160,
                   help="Window length (must match extraction)")
    return p.parse_args()


def assign_split(identity):
    if identity in TEST_IDS:
        return "test"
    if identity in VAL_IDS:
        return "val"
    return "train"


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_path   = data_root / "manifest.csv"
    split_path      = data_root / "dataset_split.csv"
    crops_path      = data_root / "temporal_crops_manifest.csv"
    wave_root       = data_root / "waveforms"

    assert manifest_path.exists(), \
        f"manifest.csv not found at {manifest_path}. Run extract_waveforms.py first."

    df = pd.read_csv(manifest_path)
    ok = df[df["status"] == "ok"].copy()

    # Filter to only files that actually exist on disk
    ok = ok[ok.apply(
        lambda r: (wave_root / r["class"] / f"{r['video_id']}.npy").exists(), axis=1
    )].copy()
    print(f"Manifest OK entries: {len(df[df['status']=='ok'])} | "
          f"Files on disk:       {len(ok)}")

    # Extract identity prefix (id0, id1, ..., id58) from video_id
    ok["identity"] = ok["video_id"].str.extract(r"(id\d+)")[0]
    missing_id = ok["identity"].isna().sum()
    if missing_id > 0:
        print(f"WARNING: {missing_id} videos have no parseable identity — dropping")
        ok = ok.dropna(subset=["identity"])

    all_ids = sorted(ok["identity"].unique())
    print(f"Unique identities: {len(all_ids)}")

    # Verify fixed split identities exist
    for name, fixed in [("TEST_IDS", TEST_IDS), ("VAL_IDS", VAL_IDS)]:
        missing = fixed - set(all_ids)
        if missing:
            print(f"WARNING: {name} has identities not in data: {missing}")

    # Assign splits
    ok["split"] = ok["identity"].map(assign_split)
    ok["label"] = (ok["class"] == "fake").astype(int)

    # Print split stats
    print("\n=== Dataset Split Statistics ===")
    for split in ["train", "val", "test"]:
        sub = ok[ok["split"] == split]
        n_real = (sub["class"] == "real").sum()
        n_fake = (sub["class"] == "fake").sum()
        n_ids  = sub["identity"].nunique()
        print(f"  {split:<6} | {n_ids:>3} identities | "
              f"{n_real:>6} real | {n_fake:>6} fake | {len(sub):>6} total")

    # Save split CSV
    ok.to_csv(split_path, index=False)
    print(f"\nSplit saved → {split_path}")

    # ── Training balance check ─────────────────────────────────────────────────
    # Windowed real waveforms (id_w0, id_w1, ...) are already in the manifest
    # as separate entries — no separate crop manifest needed.
    train_df  = ok[ok["split"] == "train"]
    n_real_tr = (train_df["class"] == "real").sum()
    n_fake_tr = (train_df["class"] == "fake").sum()
    ratio     = n_fake_tr / max(n_real_tr, 1)
    print(f"\nTraining balance (windowed real waveforms baked in at extraction):")
    print(f"  Real: {n_real_tr}")
    print(f"  Fake: {n_fake_tr}")
    print(f"  Ratio fake/real: {ratio:.2f}  (target: near 1.0)")

    print(f"\nNext step: python3 run_experiments.py --data-root {args.data_root}")


if __name__ == "__main__":
    main()