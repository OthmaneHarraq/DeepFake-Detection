# Physiological Signals as a Forensic Modality for Talking-Face Deepfake Detection

**Othmane Harraq, Tamer Aldwairi — Temple University**

---

## Overview

This repository contains the code and results for our rPPG-based talking-face deepfake detection framework. We show that remote photoplethysmography (rPPG) is a uniquely motivated detection modality for talking-face (TF) synthesis, where, unlike face-swap, no real video substrate exists from which physiological characteristics can be inherited.

Our 1D ResNet achieves **AUC 0.806 ± 0.003** on the TF subset of Celeb-DF++ under a strict subject-independent protocol, placing it within 2.4 points of the best published general-purpose detector (Effort, ICML 2025) while operating exclusively on the physiological channel.

---

## Repository Structure

```
├── src/
│   ├── extract_waveforms.py          
│   ├── build_split.py                
│   ├── run_experiments.py            
│   ├── per_method_isolated.py       
│   ├── generate_roc_curves.py       
│   └── generate_waveform_comparison.py
├── figures/
│   ├── roc_curves.pdf
│   └── waveform_comparison.pdf
├── results/
│   └── per_method_isolated.json 
└── requirements.txt
```

---

## Requirements

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

> **GPU note:** The `requirements.txt` lists the CUDA 11.8 build of PyTorch. If you need a different CUDA version or CPU-only, install PyTorch separately first:
> ```bash
> # CUDA 11.8
> pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
> # CPU only
> pip install torch torchvision
> ```

> **OpenCV note:** `opencv-python-headless` is used (no GUI dependency). Do not install `opencv-python` or `opencv-contrib-python` alongside it as they conflict.

---

## Dataset

We use the **TalkingFace subset of Celeb-DF++** (Li et al., 2025):
- 590 real videos from 59 celebrity identities
- 17,500 TF-forged videos across 7 synthesis methods (2,500 per method): AniTalker, EchoMimic, EDTalk, FLOAT, IP-LAP, Real3DPortrait, SadTalker

Dataset access: [https://github.com/OUC-VAS/Celeb-DF-PP](https://github.com/OUC-VAS/Celeb-DF-PP)

> **Note:** Celeb-DF++ requires signing a license agreement. We cannot redistribute the data.

---

## RhythmFormer

rPPG waveforms are extracted using **RhythmFormer** (Zou et al., 2024) with the `UBFC_cross` checkpoint.

Download the checkpoint from the official RhythmFormer repository and place it at:
```
checkpoints/UBFC_cross.pth
```

RhythmFormer repo: https://github.com/zizheng-guo/rhythmformer

> **Note:** The RhythmFormer checkpoint belongs to the original authors and cannot be redistributed here.

---

## Reproduction Steps

### Step 1 — Extract rPPG waveforms

```bash
python src/extract_waveforms.py \
  --video_dir /path/to/CelebDF/TalkingFace \
  --real_dir /path/to/CelebDF/Celeb-real \
  --output_dir data/waveforms \
  --checkpoint checkpoints/UBFC_cross.pth
```

This produces one `.npy` waveform file per fake video and stride-60 windowed waveforms for real videos.

### Step 2 — Build the subject-independent split

```bash
python src/build_split.py \
  --waveform_dir data/waveforms \
  --output data/dataset_split.csv
```

Fixed identity assignments:
- **Test:** id0, id4, id6, id11, id13, id16, id23, id27, id54
- **Val:** id8, id12, id19, id22, id34, id42, id44, id52, id56
- **Train:** remaining 41 identities

### Step 3 — Run main experiments

```bash
python src/run_experiments.py \
  --split data/dataset_split.csv \
  --waveform_dir data/waveforms \
  --output_dir results/
```

Reproduces Tables II, III, and IV from the paper (technique isolation, main results, architecture comparison).

### Step 4 — Per-method isolated training

```bash
python src/per_method_isolated.py \
  --split data/dataset_split.csv \
  --waveform_dir data/waveforms \
  --output results/per_method_isolated.json
```

Reproduces Table V (per-method AUC, 5 seeds, 1D ResNet vs Transformer).

### Step 5 — Generate figures

```bash
python src/generate_roc_curves.py --output figures/roc_curves.pdf
python src/generate_waveform_comparison.py \
  --waveform_dir data/waveforms \
  --output figures/waveform_comparison.pdf
```

---

## Main Results

### Overall (18-identity, 5 seeds)

| Method | AUC (mean ± std) | EER |
|---|---|---|
| DeepFakesON-Phys (reproduction) | 0.622 | — |
| Effort (ICML 2025, SOTA) | 0.830 | — |
| **Ours — 1D ResNet** | **0.806 ± 0.003** | **27.8%** |
| Ours — Transformer | 0.789 ± 0.005 | 29.1% |

### Per-method (1D ResNet, isolated training)

| Method | AUC (mean ± std) | EER |
|---|---|---|
| Real3DPortrait | 0.985 ± 0.001 | ~5.8% |
| EDTalk | 0.950 ± 0.002 | ~11.1% |
| SadTalker | 0.946 ± 0.002 | ~13.2% |
| AniTalker | 0.908 ± 0.003 | ~16.9% |
| EchoMimic | 0.865 ± 0.003 | ~20.7% |
| FLOAT | 0.794 ± 0.005 | ~26.2% |
| IP-LAP | 0.690 ± 0.010 | ~35.2% |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{harraq2027rppg,
  author={Harraq, Othmane and Aldwairi, Tamer},
  title={Physiological Signals as a Forensic Modality for Talking-Face Deepfake Detection},
  booktitle={WACV},
  year={2027}
}
```

---

## License

This code is released for research purposes only. See LICENSE for details.
