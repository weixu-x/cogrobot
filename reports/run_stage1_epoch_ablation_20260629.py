"""Run Stage 1 epoch ablations for Corsi memory-recall V2."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.experiments.corsi_memory_recall_v2.train import (
    _build_model,
    canonical_manifest_path,
    load_config,
    save_checkpoint,
    set_seed,
    train_one_run,
)


CONFIG = Path("corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json")
OUTPUT_ROOT = Path("corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12")
SUMMARY_PATH = OUTPUT_ROOT / "stage1_epoch_ablation_20260629_summary.json"
DEVICE = "cuda:0"
SEED = 0


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def make_zero_epoch_control(cfg: dict, manifest: dict) -> dict:
    run_name = "stage1_seed0_e000_ablation_20260629"
    run_dir = OUTPUT_ROOT / run_name
    if run_dir.exists():
        raise RuntimeError(f"run_dir already exists: {run_dir}")
    set_seed(SEED)
    model = _build_model(cfg, stage=1, manifest=manifest)
    run_dir.mkdir(parents=True, exist_ok=False)
    extra = {
        "zero_epoch_control": True,
        "note": "Untrained Stage 1 model checkpoint for 0-epoch warm-start/no-pretrain ablation control.",
        "selection_key": "zero_epoch",
        "selection_metric": "not_trained",
    }
    for name in ("best.pt", "latest.pt"):
        save_checkpoint(
            run_dir / name,
            model=model,
            optimizer=None,
            scaler=None,
            epoch=-1,
            stage=1,
            best_metric=0.0,
            config=cfg,
            manifest=manifest,
            seed=SEED,
            extra=extra,
        )
    summary = {
        "stage": 1,
        "seed": SEED,
        "run_dir": str(run_dir),
        "device": "cpu_checkpoint_created_from_config_device_cuda0",
        "max_epochs": 0,
        "history": [],
        "best_metric": None,
        "best_checkpoint": str(run_dir / "best.pt"),
        "primary_selection": "zero_epoch_control",
        "best_checkpoints": {
            "primary": str(run_dir / "best.pt"),
            "latest": str(run_dir / "latest.pt"),
        },
        "zero_epoch_control": True,
        "config": str(CONFIG),
        "canonical_fingerprint": manifest.get("canonical_fingerprint"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stage1(cfg: dict, epochs: int) -> dict:
    run_name = f"stage1_seed0_e{epochs:03d}_ablation_20260629"
    run_dir = OUTPUT_ROOT / run_name
    if run_dir.exists():
        raise RuntimeError(f"run_dir already exists: {run_dir}")
    emit({"event": "start_train", "epochs": epochs, "run_dir": str(run_dir), "device": DEVICE})
    start = time.time()
    summary = train_one_run(
        cfg,
        stage=1,
        seed=SEED,
        run_dir=run_dir,
        max_epochs=epochs,
        overfit_episodes=0,
        resume=False,
        allow_full_training=True,
        dry_run=False,
    )
    elapsed = time.time() - start
    history = summary.get("history") or []
    best_epoch = None
    best_val_loss = None
    if history:
        best_row = min(history, key=lambda row: row.get("val", {}).get("loss", float("inf")))
        best_epoch = best_row.get("epoch")
        best_val_loss = best_row.get("val", {}).get("loss")
    compact = {
        "event": "done_train",
        "epochs": epochs,
        "elapsed_sec": round(elapsed, 3),
        "run_dir": str(run_dir),
        "history_len": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "last_epoch": history[-1].get("epoch") if history else None,
        "last_val_loss": history[-1].get("val", {}).get("loss") if history else None,
        "best_checkpoint": summary.get("best_checkpoint"),
        "full_summary_path": str(run_dir / "summary.json"),
    }
    emit(compact)
    return compact


def main() -> int:
    cfg = load_config(CONFIG)
    cfg["device"] = DEVICE
    manifest_path = canonical_manifest_path(cfg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_summaries = []

    emit({"event": "start_zero_epoch_control"})
    zero_summary = make_zero_epoch_control(cfg, manifest)
    all_summaries.append(zero_summary)
    emit({"event": "done_zero_epoch_control", "run_dir": zero_summary["run_dir"]})

    for epochs in (30, 60, 90):
        all_summaries.append(run_stage1(cfg, epochs))

    SUMMARY_PATH.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    emit({"event": "wrote_ablation_summary", "path": str(SUMMARY_PATH), "count": len(all_summaries)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
