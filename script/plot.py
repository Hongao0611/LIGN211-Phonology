#!/usr/bin/env python3
"""
Make TWO figures from all logs matching MLP*.log:

Figure A (accuracy):
  rows = {SEGMENTAL, PROSODIC, COMBINED}
  cols = {MLP, LSTM}
  each subplot overlays trajectories from all representations found for that run
  (e.g., baseline / GloVe / GloVe24), plotting dev_acc over epoch.

Figure B (loss):
  same grid, overlays train_loss and dev_loss over epoch.

Assumptions:
- Log files are named like:
    MLP-baseline.log
    MLP-GloVe.log
    MLP-GloVe24.log
    LSTM-baseline.log
    LSTM-GloVe.log
    LSTM-GloVe24.log
- Lines contain model headers:
    "Training SEGMENTAL model" etc.
- Epoch lines contain:
    "Epoch 10: train_loss=..., dev_loss=..., dev_acc=..."
  or zero-padded epochs:
    "Epoch 005: train_loss=..., dev_loss=..., dev_acc=..."

Usage:
  python plot_MLP_grids.py --glob "MLP*.log" --outdir figures

Requires:
  pip install matplotlib pandas
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


MODEL_HEADER_RE = re.compile(r"Training\s+(SEGMENTAL|PROSODIC|COMBINED)\s+model", re.IGNORECASE)

EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+):\s+train_loss=(?P<train_loss>[-+]?(\d+(\.\d*)?|\.\d+))"
    r",\s+dev_loss=(?P<dev_loss>[-+]?(\d+(\.\d*)?|\.\d+))"
    r",\s+dev_acc=(?P<dev_acc>[-+]?(\d+(\.\d*)?|\.\d+))"
)

# Filename: LSTM-GloVe24.log -> run=LSTM, rep=GloVe24
FNAME_RE = re.compile(r"^(?P<run>LSTM?|MLP)\-(?P<rep>.+)\.log$", re.IGNORECASE)


@dataclass
class Curve:
    epoch: List[int]
    train_loss: List[float]
    dev_loss: List[float]
    dev_acc: List[float]

    def df(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "epoch": self.epoch,
                    "train_loss": self.train_loss,
                    "dev_loss": self.dev_loss,
                    "dev_acc": self.dev_acc,
                }
            )
            .sort_values("epoch")
            .drop_duplicates(subset=["epoch"], keep="last")
        )


def parse_MLP_log(path: Path) -> Dict[str, Curve]:
    """
    Returns: dict condition -> Curve
      condition in {"SEGMENTAL","PROSODIC","COMBINED"}
    """
    tmp = {
        "SEGMENTAL": {"epoch": [], "train_loss": [], "dev_loss": [], "dev_acc": []},
        "PROSODIC": {"epoch": [], "train_loss": [], "dev_loss": [], "dev_acc": []},
        "COMBINED": {"epoch": [], "train_loss": [], "dev_loss": [], "dev_acc": []},
    }
    current: Optional[str] = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mh = MODEL_HEADER_RE.search(line)
            if mh:
                current = mh.group(1).upper()
                continue
            if current is None:
                continue

            m = EPOCH_LINE_RE.search(line)
            if not m:
                continue

            tmp[current]["epoch"].append(int(m.group("epoch")))
            tmp[current]["train_loss"].append(float(m.group("train_loss")))
            tmp[current]["dev_loss"].append(float(m.group("dev_loss")))
            tmp[current]["dev_acc"].append(float(m.group("dev_acc")))

    out: Dict[str, Curve] = {}
    for cond, d in tmp.items():
        if d["epoch"]:
            out[cond] = Curve(
                epoch=d["epoch"],
                train_loss=d["train_loss"],
                dev_loss=d["dev_loss"],
                dev_acc=d["dev_acc"],
            )
    return out


def parse_run_and_rep(path: Path) -> Optional[Tuple[str, str]]:
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    run = m.group("run").upper()
    rep = m.group("rep")
    # normalize run name to exactly "MLP" or "LSTM"
    run = "LSTM" if run.startswith("LSTM") else "MLP"
    return run, rep


def load_all(glob_pat: str) -> Dict[str, Dict[str, Dict[str, Curve]]]:
    """
    Returns nested dict:
      data[run][rep][condition] = Curve
    """
    data: Dict[str, Dict[str, Dict[str, Curve]]] = {"MLP": {}, "LSTM": {}}
    for p in sorted(Path(".").glob(glob_pat)):
        rr = parse_run_and_rep(p)
        if rr is None:
            continue
        run, rep = rr
        curves = parse_MLP_log(p)
        if not curves:
            # likely not an epoch-based log
            continue
        data[run][rep] = curves
    return data


def plot_grid_accuracy(
    data: Dict[str, Dict[str, Dict[str, Curve]]],
    outpath: Path,
    title: str = "Training trajectories — dev accuracy",
):
    runs = ["MLP", "LSTM"]
    conds = ["SEGMENTAL", "PROSODIC", "COMBINED"]

    # Define specific y-axis limits for each condition (min, max)
    # Adjust these numbers based on your actual data!
    ylims = {
        "SEGMENTAL": (0.48, 0.52),
        "PROSODIC":  (0.60, 0.65),
        "COMBINED":  (0.62, 0.68),
    }

    fig, axes = plt.subplots(
        nrows=len(conds),
        ncols=len(runs),
        figsize=(13, 10),
        constrained_layout=True,
        sharex=False,
        sharey="row", # This ensures MLP and LSTM share the same scale for the same condition
    )

    for r_i, cond in enumerate(conds):
        for c_i, run in enumerate(runs):
            ax = axes[r_i][c_i]
            reps = sorted(data.get(run, {}).keys())

            any_plotted = False
            for rep in reps:
                curves = data[run][rep]
                if cond not in curves:
                    continue
                df = curves[cond].df()
                ax.plot(df["epoch"], df["dev_acc"], linewidth=2, label=rep)
                any_plotted = True

            ax.set_title(f"{cond} — {run}")
            ax.set_xlabel("Epoch")
            if c_i == 0:
                ax.set_ylabel("Dev accuracy")
            
            # Apply the specific y-limits for this condition
            ax.set_ylim(ylims[cond])
            
            ax.grid(True, alpha=0.3)
            if any_plotted:
                ax.legend(fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=14)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_grid_loss(
    data: Dict[str, Dict[str, Dict[str, Curve]]],
    outpath: Path,
    title: str = "Training trajectories — loss (train/dev)",
):
    runs = ["MLP", "LSTM"]
    conds = ["SEGMENTAL", "PROSODIC", "COMBINED"]

    fig, axes = plt.subplots(
        nrows=len(conds),
        ncols=len(runs),
        figsize=(13, 10),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )

    # For loss plots, line styles distinguish train/dev; color distinguishes rep.
    train_ls = "-"
    dev_ls = "--"

    for r_i, cond in enumerate(conds):
        for c_i, run in enumerate(runs):
            ax = axes[r_i][c_i]
            reps = sorted(data.get(run, {}).keys())

            any_plotted = False
            for rep in reps:
                curves = data[run][rep]
                if cond not in curves:
                    continue
                df = curves[cond].df()

                # consistent color per rep: matplotlib cycles colors by call order.
                (line_train,) = ax.plot(
                    df["epoch"], df["train_loss"], linewidth=2, linestyle=train_ls, label=f"{rep} (train)"
                )
                ax.plot(
                    df["epoch"],
                    df["dev_loss"],
                    linewidth=2,
                    linestyle=dev_ls,
                    color=line_train.get_color(),
                    label=f"{rep} (dev)",
                )
                any_plotted = True

            ax.set_title(f"{cond} — {run}")
            ax.set_xlabel("Epoch")
            if c_i == 0:
                ax.set_ylabel("Loss (cross-entropy / NLL)")
            ax.grid(True, alpha=0.3)
            if any_plotted:
                ax.legend(fontsize=7, ncol=2, frameon=True)

    fig.suptitle(title, fontsize=14)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="[LM]*.log", help="Glob for input logs.")
    ap.add_argument("--outdir", default="figures", help="Output directory.")
    ap.add_argument("--acc-name", default="grid_devacc.png", help="Filename for accuracy grid.")
    ap.add_argument("--loss-name", default="grid_loss.png", help="Filename for loss grid.")
    args = ap.parse_args()

    data = load_all(args.glob)

    # Basic sanity print
    for run in ["MLP", "LSTM"]:
        reps = sorted(data[run].keys())
        print(f"{run}: {len(reps)} reps -> {reps}")

    outdir = Path(args.outdir)
    plot_grid_accuracy(data, outdir / args.acc_name)
    print(f"[ok] wrote {outdir / args.acc_name}")

    plot_grid_loss(data, outdir / args.loss_name)
    print(f"[ok] wrote {outdir / args.loss_name}")


if __name__ == "__main__":
    main()