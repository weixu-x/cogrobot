"""Continue the Stage 1 epoch ablation from 90 to 120 epochs."""

from __future__ import annotations

import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.experiments.corsi_memory_recall_v2 import train as train_mod


CONFIG = Path("corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json")
OUTPUT_ROOT = Path("corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12")
BASE_RUN = OUTPUT_ROOT / "stage1_seed0_e090_ablation_20260629"
RUN_DIR = OUTPUT_ROOT / "stage1_seed0_e120_ablation_20260629"
SUMMARY_PATH = OUTPUT_ROOT / "stage1_epoch_ablation_20260629_summary.json"
DEVICE = "cuda:0"
SEED = 0


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def safe_restore_rng_state(state: dict) -> None:
    if "torch" in state:
        torch_state = state["torch"]
        if torch.is_tensor(torch_state):
            torch_state = torch_state.detach().cpu()
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and state.get("cuda"):
        cuda_states = []
        for item in state["cuda"]:
            cuda_states.append(item.detach().cpu() if torch.is_tensor(item) else item)
        torch.cuda.set_rng_state_all(cuda_states)
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def update_ablation_summary(entry: dict) -> None:
    entries = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else []
    filtered = [
        item
        for item in entries
        if "stage1_seed0_e120_ablation_20260629" not in str(item.get("run_dir", ""))
    ]
    filtered.append(entry)
    SUMMARY_PATH.write_text(json.dumps(filtered, indent=2), encoding="utf-8")


def main() -> int:
    if not BASE_RUN.exists():
        raise RuntimeError(f"base run does not exist: {BASE_RUN}")
    if RUN_DIR.exists():
        raise RuntimeError(f"target run already exists: {RUN_DIR}")

    base_summary = load_summary(BASE_RUN / "summary.json")
    shutil.copytree(BASE_RUN, RUN_DIR)
    train_mod.restore_rng_state = safe_restore_rng_state

    cfg = train_mod.load_config(CONFIG)
    cfg["device"] = DEVICE
    emit({"event": "start_resume", "from": str(BASE_RUN), "run_dir": str(RUN_DIR), "max_epochs": 120})
    start = time.time()
    resumed_summary = train_mod.train_one_run(
        cfg,
        stage=1,
        seed=SEED,
        run_dir=RUN_DIR,
        max_epochs=120,
        overfit_episodes=0,
        resume=True,
        allow_full_training=True,
        dry_run=False,
    )
    elapsed = time.time() - start

    resumed_history = resumed_summary.get("history") or []
    merged_history = list(base_summary.get("history") or []) + resumed_history
    best_row = min(merged_history, key=lambda row: row.get("val", {}).get("loss", float("inf")))
    best_val_loss = float(best_row["val"]["loss"])
    best_epoch = int(best_row["epoch"])

    merged_summary = dict(resumed_summary)
    merged_summary["max_epochs"] = 120
    merged_summary["history"] = merged_history
    merged_summary["history_source"] = {
        "base_run": str(BASE_RUN),
        "base_epochs": [0, 89],
        "resumed_epochs": [90, 119],
    }
    merged_summary["best_metric"] = -best_val_loss
    merged_summary["best_epoch"] = best_epoch
    merged_summary["best_val_loss"] = best_val_loss
    merged_summary["best_checkpoint"] = str(RUN_DIR / "best.pt")
    merged_summary.setdefault("best_checkpoints", {})["primary"] = str(RUN_DIR / "best.pt")
    merged_summary["best_checkpoints"]["val_loss"] = str(RUN_DIR / "best_val_loss.pt")
    (RUN_DIR / "summary.json").write_text(json.dumps(merged_summary, indent=2), encoding="utf-8")

    compact = {
        "event": "done_train",
        "epochs": 120,
        "elapsed_sec": round(elapsed, 3),
        "run_dir": str(RUN_DIR),
        "history_len": len(merged_history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "last_epoch": int(merged_history[-1]["epoch"]),
        "last_val_loss": float(merged_history[-1]["val"]["loss"]),
        "best_checkpoint": str(RUN_DIR / "best.pt"),
        "full_summary_path": str(RUN_DIR / "summary.json"),
        "continued_from": str(BASE_RUN),
    }
    update_ablation_summary(compact)
    emit(compact)
    emit({"event": "updated_ablation_summary", "path": str(SUMMARY_PATH)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
