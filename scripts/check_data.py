"""Validate supplied records through the packaged readers before training."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from eqnet.data.seismic_trace import SeismicTraceIterableDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("h5", "npz"), required=True)
    parser.add_argument("--data-list", type=Path, required=True)
    parser.add_argument("--hdf5-file", type=Path)
    parser.add_argument("--limit", type=int, default=20, help="0 validates all list entries")
    opts = parser.parse_args()
    if opts.limit < 0:
        parser.error("limit must be nonnegative")
    if (opts.format == "h5") != (opts.hdf5_file is not None):
        parser.error("h5 needs --hdf5-file; npz must omit it")
    with opts.data_list.open(encoding="utf-8-sig", newline="") as stream:
        entries = [row[0].strip() for row in csv.reader(stream) if row and row[0].strip()]
    if not entries:
        parser.error("Empty data list")
    random.seed(1337)
    np.random.seed(1337)
    dataset = SeismicTraceIterableDataset(data_list=[str(opts.data_list.resolve())],
        hdf5_file=str(opts.hdf5_file.resolve()) if opts.hdf5_file else None,
        format=opts.format, training=True, shuffle=False, min_snr=None, sampling_rate=100)
    items = dataset.data_list if opts.limit == 0 else dataset.data_list[:opts.limit]
    if opts.format == "h5":
        with h5py.File(opts.hdf5_file, "r") as stream:
            for name in items:
                record = dataset.read_training_h5(name, stream)
                if record is None or not np.isfinite(record["waveform"]).all():
                    raise ValueError(f"Invalid HDF5 waveform: {name}")
                if record["waveform"].shape[0] != 3 or record["waveform"].dtype != np.float32:
                    raise ValueError(f"Unexpected HDF5 shape/dtype: {name}")
                if not any(record["phase_pick"][channel].max() > 0 for channel in (1, 2)):
                    raise ValueError(f"No P/S labels in {name}")
    else:
        for name in items:
            record = dataset.read_training_npz(name)
            if record["data"].shape != (3, 4096, 1) or record["data"].dtype != torch.float32:
                raise ValueError(f"Unexpected NPZ tensor shape/dtype: {name}")
            if not torch.isfinite(record["data"]).all():
                raise ValueError(f"Invalid NPZ waveform: {name}")
    print(json.dumps({"format": opts.format, "list_entries": len(entries),
        "records_checked": len(items), "status": "passed"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
