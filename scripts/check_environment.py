"""Check target-device imports and optionally one real model update."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    opts = parser.parse_args()
    names = ("torch", "torchvision", "numpy", "scipy", "h5py", "matplotlib", "pandas", "tqdm", "fsspec", "obspy")
    for name in names:
        importlib.import_module(name)
    import train
    import torch
    report = {"python": sys.version, "versions": {n: metadata.version(n) for n in names},
              "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    if opts.smoke:
        from eqnet.models.phasenet import build_model
        from eqnet.data.seismic_trace import generate_phase_label
        torch.set_num_threads(2)
        torch.manual_seed(1337)
        label, _ = generate_phase_label([[1024], [2048]], nt=4096, label_width=[40])
        data = torch.randn(1, 3, 4096, 1, dtype=torch.float32, device=opts.device)
        target = torch.from_numpy(label[None, :, :, None]).to(opts.device)
        assert target.shape == data.shape and target.dtype == data.dtype
        assert torch.allclose(target.sum(dim=1), torch.ones(1, 4096, 1, device=opts.device))
        model = build_model(backbone="unet").to(opts.device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        result = model({"data": data, "phase_pick": target})
        assert torch.isfinite(result["loss"])
        optimizer.zero_grad()
        result["loss"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model({"data": data})["phase"]
        assert prediction.shape == (1, 3, 4096, 1) and torch.isfinite(prediction).all()
        report["smoke"] = {"device": opts.device, "input_shape": list(data.shape),
            "output_shape": list(prediction.shape), "loss": result["loss"].item(), "backward_optimizer_step": "passed"}
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
