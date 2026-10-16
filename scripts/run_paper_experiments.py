from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))

from griddx.paper_experiments import run_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run the paper's controlled neural-network experiments.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paper_experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--tasks", nargs="+", choices=["device", "station"], default=["device", "station"])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--device-epochs", type=int, default=25)
    parser.add_argument("--station-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.device_epochs, args.station_epochs, args.batch_size, args.threads) < 1:
        parser.error("Epochs, batch size, and threads must be positive.")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique.")
    return args


if __name__ == "__main__":
    run_experiments(parse_args())
