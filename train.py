import datetime
import argparse
import csv
import logging
import json
import os
import random
import time
from contextlib import nullcontext
from glob import glob

# 规避 Windows 下 conda MKL 与 torch 自带 libiomp5md.dll 冲突导致的
# "OMP: Error #15 ... already initialized"，必须放在 numpy/torch import 之前。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import eqnet
import utils
from eqnet.data import SeismicTraceIterableDataset
from eqnet.models.unet import moving_normalize
from eqnet.utils.evaluation import PickStats, gt_picks_from_label, pred_picks_from_logits, print_pick_metrics

matplotlib.use("agg")
logger = logging.getLogger("EQNet")
wandb = None
# mp.set_start_method("spawn", force=True)


def load_checkpoint(path):
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is not None:
        with safe_globals([argparse.Namespace]):
            return torch.load(path, map_location="cpu", weights_only=True)

    # 兼容旧版 PyTorch。仅加载自己训练生成的可信 checkpoint。
    return torch.load(path, map_location="cpu")

EPOCH_LOG_FIELDS = [
    "epoch",
    "epoch_number",
    "timestamp",
    "train_loss",
    "validation_loss",
    "best_validation_loss",
    "learning_rate",
    "train_seconds",
    "validation_seconds",
    "improved",
    "sampling_rate_hz",
    "pick_tolerance_seconds",
    "pick_peak_window_seconds",
    "p_threshold",
    "p_tp",
    "p_fp",
    "p_fn",
    "p_precision",
    "p_recall",
    "p_f1",
    "p_mae_seconds",
    "s_threshold",
    "s_tp",
    "s_fp",
    "s_fn",
    "s_precision",
    "s_recall",
    "s_f1",
    "s_mae_seconds",
]


def append_epoch_log(output_dir, record):
    csv_path = os.path.join(output_dir, "training_log.csv")
    jsonl_path = os.path.join(output_dir, "training_log.jsonl")
    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def evaluate(model, data_loader, scaler, args, epoch=0, total_samples=1):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    pick_stats = None
    evaluated_samples = 0
    if args.model == "phasenet":
        pick_stats = {
            phase: {"tp": 0, "fp": 0, "fn": 0, "true": 0, "error_sum_samples": 0.0}
            for phase in ("P", "S")
        }
    if args.model == "phasenet_plus":
        metric_logger.add_meter("loss_phase", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("loss_event_center", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("loss_event_time", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("loss_polarity", utils.SmoothedValue(window_size=1, fmt="{value}"))
    evaluation_stage = "Test" if args.test_only else "Validation"
    header = f"{evaluation_stage}: "

    # pick-detection metrics (0.5 s tolerance), only when labeled NPZ windows are available
    dt_s = 1.0 / args.sampling_rate
    tol = round(args.eval_tolerance_s / dt_s)
    use_pick_eval = (args.format == "npz") and (args.model == "phasenet")
    pick_stats_eval = PickStats(phases=args.phases) if use_pick_eval else None
    n_eval_traces = 0

    processed_samples = 0
    with torch.inference_mode():
        for meta in metric_logger.log_every(data_loader, args.print_freq, header):
            output = model(meta)
            batch_size = meta["data"].shape[0]
            evaluated_samples += batch_size

            metric_logger.meters["loss"].update(output["loss"].item(), n=batch_size)
            if args.model == "phasenet_plus":
                metric_logger.meters["loss_phase"].update(output["loss_phase"].item(), n=batch_size)
                metric_logger.meters["loss_event_center"].update(output["loss_event_center"].item(), n=batch_size)
                metric_logger.meters["loss_event_time"].update(output["loss_event_time"].item(), n=batch_size)
                metric_logger.meters["loss_polarity"].update(output["loss_polarity"].item(), n=batch_size)

            if pick_stats_eval is not None and "phase_pick" in meta:
                phase_pick = meta["phase_pick"].float().cpu().numpy()[:, :, :, 0]  # (B, C, nt)
                for b in range(batch_size):
                    gt = gt_picks_from_label(phase_pick[b], phases=args.phases)
                    gt = {ph: pk for ph, pk in gt.items() if pk is not None}
                    if not gt:
                        continue  # no ground-truth pick in this window, skip
                    pred = pred_picks_from_logits(
                        output["phase"][b : b + 1], phases=args.phases, vmin=args.eval_min_prob
                    )
                    pred = {ph: pk for ph, pk in pred.items() if ph in gt}
                    pick_stats_eval.add(gt, pred, tol)
                    n_eval_traces += 1

            if pick_stats is not None:
                probabilities = torch.softmax(output["phase"], dim=1)
                peak_kernel = max(1, round(args.pick_peak_window_seconds * args.sampling_rate))
                if peak_kernel % 2 == 0:
                    peak_kernel += 1
                peak_scores, peak_indices = eqnet.utils.detect_peaks(
                    probabilities,
                    vmin=min(args.p_pick_threshold, args.s_pick_threshold),
                    kernel=peak_kernel,
                    dt=1.0 / args.sampling_rate,
                )
                targets = meta["phase_pick"].detach().cpu()
                tolerance_samples = round(args.pick_tolerance_seconds * args.sampling_rate)
                thresholds = {"P": args.p_pick_threshold, "S": args.s_pick_threshold}

                for batch_index in range(batch_size):
                    for phase_index, phase in enumerate(("P", "S"), start=1):
                        target_trace = targets[batch_index, phase_index, :, 0]
                        if target_trace.max().item() <= 0:
                            continue
                        true_index = int(torch.argmax(target_trace).item())
                        stats = pick_stats[phase]
                        stats["true"] += 1

                        scores = peak_scores[batch_index, phase_index - 1, 0]
                        indices = peak_indices[batch_index, phase_index - 1, 0]
                        predicted_indices = indices[scores >= thresholds[phase]].tolist()
                        if not predicted_indices:
                            stats["fn"] += 1
                            continue

                        errors = [abs(int(index) - true_index) for index in predicted_indices]
                        best_error = min(errors)
                        if best_error <= tolerance_samples:
                            stats["tp"] += 1
                            stats["fp"] += len(predicted_indices) - 1
                            stats["error_sum_samples"] += best_error
                        else:
                            stats["fp"] += len(predicted_indices)
                            stats["fn"] += 1

            processed_samples += batch_size
            if processed_samples > total_samples:
                break

    metric_logger.synchronize_between_processes()
    print(f"{evaluation_stage} loss = {metric_logger.loss.global_avg:.3e}")
    pick_metrics = None
    if pick_stats is not None:
        stat_names = ("tp", "fp", "fn", "true", "error_sum_samples")
        flat_stats = [pick_stats[phase][name] for phase in ("P", "S") for name in stat_names]
        reduced_stats = utils.reduce_across_processes(flat_stats).cpu().tolist()
        reduced_samples = int(utils.reduce_across_processes([evaluated_samples]).item())
        offset = 0
        for phase in ("P", "S"):
            for name in stat_names:
                value = reduced_stats[offset]
                pick_stats[phase][name] = float(value) if name == "error_sum_samples" else int(value)
                offset += 1

    if pick_stats is not None and utils.is_main_process():
        loss_key = "test_loss" if args.test_only else "validation_loss"
        metrics = {
            loss_key: metric_logger.loss.global_avg,
            "epoch": epoch,
            "evaluated_samples": reduced_samples,
            "sampling_rate_hz": args.sampling_rate,
            "tolerance_seconds": args.pick_tolerance_seconds,
            "peak_window_seconds": args.pick_peak_window_seconds,
            "phases": {},
        }
        stage = "Test" if args.test_only else f"Validation epoch {epoch + 1}"
        print(
            f"{stage} pick metrics (tolerance=+/-{args.pick_tolerance_seconds:.3f}s, "
            f"P threshold={args.p_pick_threshold:.3f}, S threshold={args.s_pick_threshold:.3f})"
        )
        for phase in ("P", "S"):
            stats = pick_stats[phase]
            precision = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
            recall = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            mae_seconds = (
                stats["error_sum_samples"] / stats["tp"] / args.sampling_rate if stats["tp"] else None
            )
            hit_rate = stats["tp"] / stats["true"] if stats["true"] else 0.0
            metrics["phases"][phase] = {
                "threshold": args.p_pick_threshold if phase == "P" else args.s_pick_threshold,
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "true_picks": stats["true"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mae_seconds": mae_seconds,
                "hit_rate_within_tolerance": hit_rate,
            }
            mae_text = f"{mae_seconds:.6f}s" if mae_seconds is not None else "N/A"
            print(
                f"{phase}: TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} "
                f"Precision={precision:.6f} Recall={recall:.6f} F1={f1:.6f} "
                f"MAE={mae_text} Hit@+/-{args.pick_tolerance_seconds:.3f}s={hit_rate:.6f}"
            )

        pick_metrics = metrics
        if args.test_only:
            metrics_path = os.path.join(args.output_dir, "test_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"Saved pick metrics to {metrics_path}")

    if pick_stats_eval is not None and n_eval_traces > 0:
        print_pick_metrics(
            pick_stats_eval.summarize(dt_s=dt_s),
            tolerance_s=args.eval_tolerance_s,
            dt_s=dt_s,
            title=f"Epoch {epoch} validation (pick detection)",
        )
    if args.wandb and utils.is_main_process():
        log = {
            "test/test_loss": metric_logger.loss.global_avg,
            "test/epoch": epoch,
        }
        if args.model == "phasenet_plus":
            log["test/loss_phase"] = metric_logger.loss_phase.global_avg
            log["test/loss_event_center"] = metric_logger.loss_event_center.global_avg
            log["test/loss_event_time"] = metric_logger.loss_event_time.global_avg
            log["test/loss_polarity"] = metric_logger.loss_polarity.global_avg
        wandb.log(log)

    plot_results(meta, model, output, args, epoch, "test")
    del meta, output

    return metric_logger, pick_metrics


def train_one_epoch(
    model,
    optimizer,
    lr_scheduler,
    data_loader,
    model_ema,
    scaler,
    args,
    epoch,
    total_samples,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    if args.model == "phasenet_plus":
        metric_logger.add_meter("loss_phase", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("loss_event_center", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("loss_event_time", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("loss_polarity", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    ctx = (
        nullcontext()
        if args.device in ["cpu", "mps"]
        else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
    )
    model.train()
    processed_samples = 0
    for i, meta in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        with ctx:
            output = model(meta)

        loss = output["loss"]

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        lr_scheduler.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        batch_size = meta["data"].shape[0]
        processed_samples += batch_size

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        if args.model == "phasenet_plus":
            metric_logger.update(loss_phase=output["loss_phase"].item())
            metric_logger.update(loss_event_center=output["loss_event_center"].item())
            metric_logger.update(loss_event_time=output["loss_event_time"].item())
            metric_logger.update(loss_polarity=output["loss_polarity"].item())
        if args.wandb and utils.is_main_process():
            log = {
                "train/train_loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
                "train/batch": i,
            }
            if args.model == "phasenet_plus":
                log["train/loss_phase"] = output["loss_phase"].item()
                log["train/loss_event_center"] = output["loss_event_center"].item()
                log["train/loss_event_time"] = output["loss_event_time"].item()
                log["train/loss_polarity"] = output["loss_polarity"].item()
            wandb.log(log)
        if processed_samples >= total_samples:
            break

    model.eval()
    plot_results(meta, model, output, args, epoch, "train")
    del meta, output, loss
    metric_logger.synchronize_between_processes()
    return metric_logger


def plot_results(meta, model, output, args, epoch, prefix=""):
    if args.disable_plot:
        return

    with torch.inference_mode():
        if args.model == "phasenet":
            phase = torch.softmax(output["phase"], dim=1).cpu().float()
            meta["data"] = moving_normalize(meta["data"])
            print("Plotting...")
            eqnet.utils.plot_phasenet_train(meta, phase, epoch=epoch, figure_dir=args.figure_dir, prefix=prefix)
            del phase

        elif args.model == "phasenet_plus":
            phase = torch.softmax(output["phase"], dim=1).cpu().float()
            event_center = torch.sigmoid(output["event_center"]).cpu().float()
            event_time = output["event_time"].cpu().float()
            polarity = torch.sigmoid(output["polarity"]).cpu().float()
            meta["data"] = moving_normalize(meta["data"])
            print("Plotting...")
            eqnet.utils.plot_phasenet_plus_train(
                meta,
                phase,
                polarity=polarity,
                event_center=event_center,
                event_time=event_time,
                epoch=epoch,
                figure_dir=args.figure_dir,
                prefix=prefix,
            )
            del phase, event_center, polarity



def main(args):
    global wandb

    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be greater than or equal to 0")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be greater than or equal to 0")
    if args.wandb or args.resume_wandb:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError("W&B was requested, but the 'wandb' package is not installed.") from exc
        wandb = wandb_module

    if args.output_dir:
        utils.mkdir(args.output_dir)
        figure_dir = os.path.join(args.output_dir, "figures")
        args.figure_dir = figure_dir
        utils.mkdir(figure_dir)

    utils.init_distributed_mode(args)
    print(args)

    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
    else:
        rank, world_size = 0, 1

    torch.manual_seed(1337 + rank)
    random.seed(1337 + rank)
    np.random.seed(1337 + rank)

    device = torch.device(args.device)
    dtype = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    scaler = torch.cuda.amp.GradScaler(enabled=((dtype == "float16") & torch.cuda.is_available()))
    args.dtype, args.ptdtype = dtype, ptdtype
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    dataset_test = None
    validation_enabled = False
    if args.model in ["phasenet", "phasenet_plus"]:
        dataset_min_snr = None if args.disable_snr_filter else 3.0
        if args.format == "npz" and args.model != "phasenet":
            raise ValueError("NPZ training currently supports --model phasenet only")
        if args.format == "npz" and args.hdf5_file is not None:
            raise ValueError("--hdf5-file cannot be combined with --format npz")
        if args.format == "npz" and args.test_only and args.test_data_list is None:
            raise ValueError("--test-data-list is required for NPZ test-only evaluation")
        if args.format == "npz" and not args.test_only and args.data_list is None:
            raise ValueError("--data-list is required for NPZ training")

        primary_data_list = args.test_data_list if args.format == "npz" and args.test_only else args.data_list
        dataset = SeismicTraceIterableDataset(
            data_path=args.data_path,
            data_list=primary_data_list,
            hdf5_file=args.hdf5_file,
            format=args.format,
            training=True,
            shuffle=not args.test_only,
            stack_event=args.stack_event,
            flip_polarity=args.flip_polarity,
            drop_channel=args.drop_channel,
            min_snr=dataset_min_snr,
            sampling_rate=args.sampling_rate,
            rank=rank,
            world_size=world_size,
        )
        train_sampler = None
        if args.format == "npz":
            validation_enabled = args.test_data_list is not None
        else:
            validation_enabled = args.test_hdf5_file is not None

        if validation_enabled:
            dataset_test = SeismicTraceIterableDataset(
                data_path=args.test_data_path,
                data_list=args.test_data_list,
                hdf5_file=args.test_hdf5_file,
                format=args.format,
                training=True,
                shuffle=False,
                stack_event=False,
                flip_polarity=False,
                drop_channel=False,
                min_snr=dataset_min_snr,
                sampling_rate=args.sampling_rate,
                rank=rank,
                world_size=world_size,
            )
        else:
            dataset_test = None
        test_sampler = None
    else:
        raise ValueError("Only phasenet and phasenet_plus are supported")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        # collate_fn=utils.collate_fn,
        collate_fn=None,
        # shuffle=True,
        drop_last=True,
        pin_memory=(args.workers > 0),
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
    )

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            collate_fn=None,
            drop_last=False,
            pin_memory=(args.workers > 0),
            persistent_workers=(args.workers > 0),
            prefetch_factor=4 if args.workers > 0 else None,
        )
    else:
        data_loader_test = None

    if args.early_stopping_patience > 0 and not validation_enabled:
        print("Early stopping is disabled because no validation dataset was provided.")

    model = eqnet.models.__dict__[args.model].build_model(
        backbone=args.backbone,
        in_channels=1,
        out_channels=(len(args.phases) + 1),
    )
    logger.info("Model:\n{}".format(model))

    print("Model:\n{}".format(model))

    model.to(device)
    if args.compile:
        print("compiling the model...")
        model = torch.compile(model)  # requires PyTorch 2.0

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # logging
    if args.wandb and utils.is_main_process():
        wandb.init(
            project=args.wandb_project if args.wandb_project else args.model,
            name=args.wandb_name,
            entity=args.wandb_group,
            dir=args.wandb_dir,
            config=args,
        )
        if args.wandb_watch:
            wandb.watch(model, log="all", log_freq=args.print_freq)

    if args.resume_wandb:
        if utils.is_main_process():
            if args.model == "phasenet_das":
                model_url = "ai4eps/model-registry/PhaseNet-DAS:latest"
                artifact = wandb.use_artifact(model_url, type="model")
                artifact_dir = artifact.download()
                checkpoint = torch.load(glob(os.path.join(artifact_dir, "*.pth"))[0], map_location="cpu")
                model.load_state_dict(checkpoint["model"], strict=True)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    iters_per_epoch = len(data_loader)
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size * iters_per_epoch, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(args.epochs - args.lr_warmup_epochs) * iters_per_epoch, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "polynomiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=0.9
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and PolynomialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs * iters_per_epoch
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs * iters_per_epoch
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs * iters_per_epoch],
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )  # , find_unused_parameters=True)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    best_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading model and optimizer from checkpoint '{args.resume}'")
            checkpoint = load_checkpoint(args.resume)
        elif os.path.isfile(args.output_dir + "/checkpoint.pth"):
            print(f"Loading model and optimizer from checkpoint '{args.output_dir}/checkpoint.pth'")
            checkpoint = load_checkpoint(args.output_dir + "/checkpoint.pth")
        else:
            print(f"No checkpoint found")
        if checkpoint is not None:
            model_without_ddp.load_state_dict(checkpoint["model"])
            if model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if args.resume_opt:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1
                best_loss = checkpoint.get("best_loss", best_loss)
                epochs_without_improvement = checkpoint.get(
                    "epochs_without_improvement", epochs_without_improvement
                )
                if scaler:
                    scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        if checkpoint is None:
            raise ValueError("--test-only requires a valid checkpoint supplied with --resume")
        if data_loader_test is None:
            raise ValueError("--test-only requires a test dataset")
        checkpoint_epoch = checkpoint.get("epoch", 0)
        metric, _ = evaluate(model, data_loader_test, scaler, args, checkpoint_epoch, len(dataset_test))
        print(f"Test-only loss = {metric.loss.global_avg:.6e}")
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        tmp_time = time.time()
        train_metric = train_one_epoch(
            model,
            optimizer,
            lr_scheduler,
            data_loader,
            model_ema,
            scaler,
            args,
            epoch,
            len(dataset),
        )
        train_seconds = time.time() - tmp_time
        print(f"Training time of epoch {epoch} of rank {rank}: {train_seconds:.3f}")

        validation_seconds = None
        pick_metrics = None
        if validation_enabled and (epoch % args.validate_every == 0):
            tmp_time = time.time()
            metric, pick_metrics = evaluate(model, data_loader_test, scaler, args, epoch, len(dataset_test))
            if model_ema:
                metric, pick_metrics = evaluate(
                    model_ema, data_loader_test, scaler, args, epoch, len(dataset_test)
                )
            validation_seconds = time.time() - tmp_time
            print(f"Testing time of epoch {epoch} of rank {rank}: {validation_seconds:.3f}")

        improved = False
        current_loss = None
        if validation_enabled and (epoch % args.validate_every == 0):
            current_loss = metric.loss.global_avg
            improved = current_loss < (best_loss - args.early_stopping_min_delta)
            if improved:
                best_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if utils.is_main_process():
            phase_metrics = pick_metrics["phases"] if pick_metrics is not None else {}
            p_metrics = phase_metrics.get("P", {})
            s_metrics = phase_metrics.get("S", {})
            epoch_record = {
                "epoch": epoch,
                "epoch_number": epoch + 1,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "train_loss": train_metric.loss.global_avg,
                "validation_loss": current_loss,
                "best_validation_loss": best_loss if current_loss is not None else None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "improved": improved,
                "sampling_rate_hz": args.sampling_rate,
                "pick_tolerance_seconds": args.pick_tolerance_seconds,
                "pick_peak_window_seconds": args.pick_peak_window_seconds,
                "p_threshold": p_metrics.get("threshold"),
                "p_tp": p_metrics.get("tp"),
                "p_fp": p_metrics.get("fp"),
                "p_fn": p_metrics.get("fn"),
                "p_precision": p_metrics.get("precision"),
                "p_recall": p_metrics.get("recall"),
                "p_f1": p_metrics.get("f1"),
                "p_mae_seconds": p_metrics.get("mae_seconds"),
                "s_threshold": s_metrics.get("threshold"),
                "s_tp": s_metrics.get("tp"),
                "s_fp": s_metrics.get("fp"),
                "s_fn": s_metrics.get("fn"),
                "s_precision": s_metrics.get("precision"),
                "s_recall": s_metrics.get("recall"),
                "s_f1": s_metrics.get("f1"),
                "s_mae_seconds": s_metrics.get("mae_seconds"),
            }
            csv_log_path, jsonl_log_path = append_epoch_log(args.output_dir or ".", epoch_record)
            print(f"Saved epoch log to {csv_log_path} and {jsonl_log_path}")

        tmp_time = time.time()
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "best_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        if model_ema:
            checkpoint["model_ema"] = model_ema.state_dict()
        if scaler:
            checkpoint["scaler"] = scaler.state_dict()

        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
        if improved:
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "model_best.pth"))
            if utils.is_main_process():
                print(f"Validation loss improved to {best_loss:.6e}; saved model_best.pth")
        elif utils.is_main_process() and current_loss is not None:
            print(
                f"Validation loss did not improve by at least "
                f"{args.early_stopping_min_delta:.3e} "
                f"({epochs_without_improvement}/{args.early_stopping_patience} epochs)"
            )

        if improved and args.wandb and utils.is_main_process():
            best_model = wandb.Artifact(
                args.wandb_name or args.model,
                type="model",
                metadata=dict(epoch=epoch, loss=best_loss),
            )
            best_model.add_file(os.path.join(args.output_dir, "model_best.pth"))
            wandb.log_artifact(best_model)

        should_stop = (
            validation_enabled
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        )
        if should_stop:
            if args.wandb and utils.is_main_process():
                wandb.log(
                    {
                        "early_stopping/epoch": epoch,
                        "early_stopping/best_loss": best_loss,
                    }
                )
            if utils.is_main_process():
                print(
                    f"Early stopping at epoch {epoch}: validation loss did not improve "
                    f"for {args.early_stopping_patience} consecutive epochs. "
                    f"Best loss: {best_loss:.6e}"
                )
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    if args.wandb and utils.is_main_process():
        wandb.finish()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="./", type=str, help="dataset path")
    parser.add_argument("--data-list", nargs="+", default=None, type=str, help="dataset list")
    parser.add_argument("--format", choices=("h5", "npz"), default="h5",
                        help="training data format (h5 or npz)")
    parser.add_argument("--label-path", nargs="+", default=None, type=str, help="label path")
    parser.add_argument("--label-list", nargs="+", default=None, type=str, help="label path")
    parser.add_argument("--noise-list", nargs="+", default=None, type=str, help="noise list")
    parser.add_argument("--hdf5-file", default=None, type=str, help="hdf5 file for training")
    parser.add_argument(
        "--disable-snr-filter",
        action="store_true",
        help="Disable SNR filtering for PhaseNet/PhaseNet+ HDF5 training data",
    )
    parser.add_argument("--test-data-path", default="./", type=str, help="test dataset path")
    parser.add_argument("--test-data-list", nargs="+", default=None, type=str, help="test dataset list")
    parser.add_argument("--test-label-path", default="+", type=None, help="test label path")
    parser.add_argument("--test-label-list", default="+", type=None, help="test label path")
    parser.add_argument("--test-noise-list", default="+", type=None, help="test noise list")
    parser.add_argument("--test-hdf5-file", default=None, type=str, help="hdf5 file for testing")
    parser.add_argument(
        "--validate-every",
        default=2,
        type=int,
        metavar="N",
        help="run full validation every N epochs (default: 2; set to 1 for every-epoch validation)",
    )
    parser.add_argument("--dataset", default="", type=str, help="dataset name")
    parser.add_argument("--model", choices=("phasenet", "phasenet_plus"), default="phasenet", type=str, help="model name")
    parser.add_argument("--backbone", default="unet", type=str, help="model backbone")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile model to torchscript",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=8,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )

    ## training hyper params
    parser.add_argument("--opt", default="adamw", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--clip-grad-norm",
        default=None,
        type=float,
        help="clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=1,
        type=int,
        help="the number of epochs to warmup (default: 1)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="linear",
        type=str,
        help="the warmup method (default: linear)",
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument(
        "--lr-scheduler", default="cosineannealinglr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./output", type=str, help="path to save outputs")
    parser.add_argument("--p-pick-threshold", default=0.3, type=float, help="P-pick probability threshold")
    parser.add_argument("--s-pick-threshold", default=0.3, type=float, help="S-pick probability threshold")
    parser.add_argument(
        "--pick-tolerance-seconds",
        default=0.5,
        type=float,
        help="maximum timing error for a true-positive pick",
    )
    parser.add_argument(
        "--pick-peak-window-seconds",
        default=1.0,
        type=float,
        help="non-maximum-suppression window for distinct predicted picks",
    )
    parser.add_argument(
        "--eval-tolerance-s",
        default=0.5,
        type=float,
        help="pick-match tolerance in seconds for detection metrics printed each epoch (default: 0.5)",
    )
    parser.add_argument(
        "--eval-min-prob",
        default=0.3,
        type=float,
        help="minimum phase probability for a predicted pick in detection metrics (default: 0.3)",
    )
    parser.add_argument("--sampling-rate", default=100.0, type=float, help="waveform sampling rate in Hz")
    parser.add_argument(
        "--disable-plot",
        action="store_true",
        help="disable training/validation figure generation",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=10,
        type=int,
        help="stop after this many validation epochs without improvement; use 0 to disable (default: 10)",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        default=0.0,
        type=float,
        help="minimum validation-loss decrease counted as an improvement (default: 0.0)",
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--resume_opt", action="store_true", help="if resume optimizer states")
    parser.add_argument("--resume_wandb", action="store_true", help="resume from wandb")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync-bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # model moving average
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    ## Data Augmentation
    parser.add_argument("--phases", default=["P", "S"], type=str, nargs="+", help="phases to use")
    parser.add_argument("--nt", default=1024 * 3, type=int, help="number of time samples")
    parser.add_argument("--nx", default=1024 * 5, type=int, help="number of space samples")
    parser.add_argument("--stack-noise", action="store_true", help="Stack noise")
    parser.add_argument("--stack-event", action="store_true", help="Stack event")
    parser.add_argument("--flip-polarity", action="store_true", help="Flip polarity")
    parser.add_argument("--drop-channel", action="store_true", help="Drop channel")
    parser.add_argument("--resample-space", action="store_true", help="Resample space resolution")
    parser.add_argument("--resample-time", action="store_true", help="Resample time  resolution")
    parser.add_argument("--masking", action="store_true", help="Masking of the input data")
    parser.add_argument("--random-crop", action="store_true", help="Random size")
    parser.add_argument("--crop-nt", default=1024, type=int, help="Crop time samples")
    parser.add_argument("--crop-nx", default=1024, type=int, help="Crop space samples")

    # wandb
    parser.add_argument("--wandb", action="store_true", help="use wandb")
    parser.add_argument("--wandb-project", default=None, type=str, help="wandb project name")
    parser.add_argument("--wandb-name", default=None, type=str, help="wandb run name")
    parser.add_argument("--wandb-group", default=None, type=str, help="wandb group name")
    parser.add_argument("--wandb-dir", default="./", type=str, help="wandb dir")
    parser.add_argument("--wandb-watch", action="store_true", help="wandb watch model")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
