"""Pick-detection evaluation metrics: precision / recall / F1 + arrival-time error.

Shared by train.py (per-epoch validation), predict.py (prediction evaluation) and
eval_precision_recall.py (standalone full-waveform evaluation).

Matching convention (standard PhaseNet-style): each trace has at most one ground-truth
P and one S pick. A predicted pick is a true positive if it lies within ``tolerance``
samples of the GT pick of the same phase; the closest predicted pick is the match.
Every other predicted pick is a false positive; a GT pick with no predicted pick
nearby is a false negative. Arrival-time error is computed only over matched picks.
"""
import numpy as np
import torch
import torch.nn.functional as F


def find_peaks(prob, vmin=0.3, kernel=128, min_sep=100):
    """Local maxima of a 1-D probability curve above ``vmin``, spaced >= ``min_sep`` samples.

    prob: 1-D numpy array of probabilities.
    Returns sorted numpy array of peak sample indices.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(np.asarray(prob, dtype=np.float32)).to(device).view(1, 1, -1, 1)
    nt = t.shape[2]
    pad = kernel // 2
    smax = F.max_pool2d(t, (kernel, 1), stride=(1, 1), padding=(pad, 0))[:, :, :nt, :]
    keep = (smax == t).float()
    s = t * keep
    pos = torch.arange(nt, 0, -1, device=t.device)[None, None, :, None].float()
    sfirst = F.max_pool2d(keep * pos, (kernel, 1), stride=(1, 1), padding=(pad, 0))[:, :, :nt, :]
    s = s * (sfirst == pos).float()
    s = s.view(-1)
    idx = torch.nonzero(s >= vmin).view(-1).cpu().numpy()
    val = s.cpu().numpy()[idx]
    order = np.argsort(-val)
    picked = []
    for i in order:
        if all(abs(int(idx[i]) - p) >= min_sep for p in picked):
            picked.append(int(idx[i]))
    return np.array(sorted(picked))


def match_picks(pred_picks, gt_pick, tol):
    """Match predicted picks against one ground-truth pick within ``tol`` samples.

    Returns (tp, fp, fn, matched_error_samples_or_None).
    """
    pred_picks = list(pred_picks)
    if gt_pick is None:
        return 0, len(pred_picks), 0, None
    within = [p for p in pred_picks if abs(p - gt_pick) <= tol]
    if not within:
        return 0, len(pred_picks), 1, None
    best = min(within, key=lambda p: abs(p - gt_pick))
    return 1, len(pred_picks) - 1, 0, int(abs(best - gt_pick))


def gt_picks_from_label(phase_pick, phases=("P", "S"), threshold=0.1):
    """Extract ground-truth pick sample indices from a soft phase label (C, nt) array.

    Channels are noise, then one per phase in ``phases`` order. A channel that is
    all ~0 means that phase's pick is not present in the window -> None.
    """
    phase_pick = np.asarray(phase_pick)
    gt = {}
    for i, ph in enumerate(phases, start=1):
        lab = phase_pick[i]
        if lab.size == 0:
            gt[ph] = None
            continue
        pk = int(np.argmax(lab))
        gt[ph] = pk if lab[pk] > threshold else None
    return gt


def pred_picks_from_logits(phase_logits, phases=("P", "S"), vmin=0.3, kernel=128, min_sep=100):
    """Extract predicted picks from phase logits (B, C, nt, 1), using the first batch element."""
    prob = torch.softmax(phase_logits.float(), dim=1)[0]  # (C, nt, 1)
    pred = {}
    for i, ph in enumerate(phases, start=1):
        peaks = find_peaks(prob[i, :, 0].cpu().numpy(), vmin=vmin, kernel=kernel, min_sep=min_sep)
        pred[ph] = [int(x) for x in peaks]
    return pred


def load_npz_gt(path):
    """Load (itp, its) ground-truth pick sample indices from a DiTing-style .npz file."""
    with np.load(path, allow_pickle=False) as z:
        itp = int(np.asarray(z["itp"]).item())
        its = int(np.asarray(z["its"]).item())
    return itp, its


class PickStats:
    """Accumulates precision / recall / F1 and arrival-time error for phase picks."""

    def __init__(self, phases=("P", "S")):
        self.phases = tuple(phases)
        self.tp = {ph: 0 for ph in self.phases}
        self.fp = {ph: 0 for ph in self.phases}
        self.fn = {ph: 0 for ph in self.phases}
        self.time_err = {ph: [] for ph in self.phases}

    def add(self, gt, pred, tol):
        """gt: dict phase -> int | None; pred: dict phase -> list[int]; tol in samples."""
        for ph in self.phases:
            tp, fp, fn, err = match_picks(pred.get(ph, []), gt.get(ph), tol)
            self.tp[ph] += tp
            self.fp[ph] += fp
            self.fn[ph] += fn
            if err is not None:
                self.time_err[ph].append(err)

    def summarize(self, dt_s=0.01):
        out = {}
        for ph in self.phases:
            out[ph] = self._metrics(ph, self.tp[ph], self.fp[ph], self.fn[ph], self.time_err[ph], dt_s)
        tp = sum(self.tp[ph] for ph in self.phases)
        fp = sum(self.fp[ph] for ph in self.phases)
        fn = sum(self.fn[ph] for ph in self.phases)
        errs = [e for ph in self.phases for e in self.time_err[ph]]
        out["combined"] = self._metrics("combined", tp, fp, fn, errs, dt_s)
        return out

    @staticmethod
    def _metrics(name, tp, fp, fn, errs, dt_s):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        return {
            "phase": name,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_matched": len(errs),
            "err_mean_samples": float(np.mean(errs)) if errs else 0.0,
            "err_median_samples": float(np.median(errs)) if errs else 0.0,
            "err_mean_s": float(np.mean(errs)) * dt_s if errs else 0.0,
            "err_median_s": float(np.median(errs)) * dt_s if errs else 0.0,
        }


def print_pick_metrics(summary, tolerance_s=0.5, dt_s=0.01, title="Pick detection"):
    """Print a compact precision/recall/F1 + arrival-time error table."""
    tol = round(tolerance_s / dt_s)
    print(f"\n=== {title} (tolerance={tolerance_s:g} s = {tol} samples @ {1 / dt_s:g} Hz) ===")
    print(f"  {'phase':9s} {'TP':>5s} {'FP':>5s} {'FN':>5s} | {'Prec':>6s} {'Rec':>6s} {'F1':>6s} |"
          f" {'mean_err':>9s} {'med_err':>9s}")
    for ph in summary:
        m = summary[ph]
        print(
            f"  {ph:9s} {m['tp']:5d} {m['fp']:5d} {m['fn']:5d} | "
            f"{m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} | "
            f"{m['err_mean_s']:8.3f}s {m['err_median_s']:8.3f}s"
        )
