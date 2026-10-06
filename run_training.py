"""Portable launcher for the existing PhaseNet + UNet NPZ/HDF5 flow."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read_list(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row[0].strip() for row in csv.reader(stream) if row and row[0].strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("h5", "npz"), required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--train-hdf5", type=Path)
    parser.add_argument("--val-hdf5", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()
    if opts.batch_size < 1 or opts.epochs < 2 or opts.workers < 0 or opts.lr <= 0:
        parser.error("batch-size/lr must be positive, epochs >= 2, workers >= 0")
    opts.train_list = opts.train_list.resolve()
    opts.val_list = opts.val_list.resolve()
    for path in (opts.train_list, opts.val_list):
        if not path.is_file():
            parser.error(f"List file missing: {path}")
    train_ids, val_ids = read_list(opts.train_list), read_list(opts.val_list)
    if len(train_ids) < opts.batch_size or not val_ids:
        parser.error("Training needs at least one full batch, and validation must be nonempty")
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        parser.error("Duplicate entries in train/validation lists")
    if opts.format == "h5":
        if opts.train_hdf5 is None or opts.val_hdf5 is None:
            parser.error("h5 requires --train-hdf5 and --val-hdf5")
        opts.train_hdf5, opts.val_hdf5 = opts.train_hdf5.resolve(), opts.val_hdf5.resolve()
        for path in (opts.train_hdf5, opts.val_hdf5):
            if not path.is_file():
                parser.error(f"HDF5 file missing: {path}")
        if opts.train_hdf5 == opts.val_hdf5 and set(train_ids).intersection(val_ids):
            parser.error("Training/validation lists overlap inside the same HDF5 file")
    else:
        if opts.train_hdf5 is not None or opts.val_hdf5 is not None:
            parser.error("Do not specify HDF5 files for NPZ training")
        resolved_lists = []
        for list_path, entries in ((opts.train_list, train_ids), (opts.val_list, val_ids)):
            resolved = [(Path(item) if Path(item).is_absolute() else list_path.parent / item).resolve()
                        for item in entries]
            for path in resolved:
                if path.suffix.lower() != ".npz" or not path.is_file():
                    parser.error(f"NPZ file missing/invalid: {path}")
            if len(set(resolved)) != len(resolved):
                parser.error(f"Duplicate paths in {list_path}")
            resolved_lists.append(set(resolved))
        if resolved_lists[0].intersection(resolved_lists[1]):
            parser.error("Training and validation NPZ paths overlap")
    output = (opts.output_dir or ROOT / "runs" / datetime.now().strftime("train_%Y%m%d_%H%M%S")).resolve()
    if output.exists() and opts.resume is None:
        parser.error("Output directory already exists; choose a new one or supply --resume")
    command = [sys.executable, str(ROOT / "train.py"), "--model", "phasenet", "--backbone", "unet",
        "--format", opts.format, "--data-list", str(opts.train_list), "--test-data-list", str(opts.val_list),
        "--device", opts.device, "--batch-size", str(opts.batch_size), "--workers", str(opts.workers),
        "--epochs", str(opts.epochs), "--lr", str(opts.lr), "--sampling-rate", "100",
        "--disable-snr-filter", "--disable-plot", "--opt", "adamw", "--weight-decay", "0.0001",
        "--lr-scheduler", "cosineannealinglr", "--lr-warmup-epochs", "1", "--lr-warmup-method", "linear",
        "--lr-warmup-decay", "0.01", "--lr-min", "0", "--early-stopping-patience", "0",
        "--output-dir", str(output)]
    if opts.format == "h5":
        command.extend(["--hdf5-file", str(opts.train_hdf5), "--test-hdf5-file", str(opts.val_hdf5)])
    if opts.resume is not None:
        opts.resume = opts.resume.resolve()
        if not opts.resume.is_file():
            parser.error(f"Checkpoint missing: {opts.resume}")
        command.extend(["--resume", str(opts.resume), "--resume_opt"])
    print(f"train={len(train_ids)} val={len(val_ids)} format={opts.format} output={output}", flush=True)
    if opts.dry_run:
        print("\n".join(command))
        return
    # Import the packaged entrypoint, using the same checkpoint loader as training.
    import train
    if opts.device == "cuda" and not train.torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the target PyTorch installation/driver, or use --device cpu")
    if opts.resume is not None:
        checkpoint = train.load_checkpoint(str(opts.resume))
        saved = checkpoint["args"]
        saved = vars(saved) if isinstance(saved, argparse.Namespace) else saved
        expected = {"model": "phasenet", "backbone": "unet", "format": opts.format,
                    "batch_size": opts.batch_size, "epochs": opts.epochs, "lr": opts.lr,
                    "opt": "adamw", "lr_scheduler": "cosineannealinglr", "sampling_rate": 100.0,
                    "lr_warmup_epochs": 1}
        differences = {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}
        if differences:
            parser.error(f"Resume configuration differs from checkpoint: {differences}. See README resume instructions")
        if checkpoint["epoch"] + 1 >= opts.epochs:
            parser.error("Checkpoint already completed the configured epochs; a new learning-rate schedule is required to extend training")
        del checkpoint
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
