"""Single-process, multi-GPU inference and performance report for MADE.

Unlike the legacy test.py launcher, this script uses DataParallel: one process
owns the evaluator and splits every global batch across the selected GPUs.
Consequently, metrics and logs are emitted exactly once.
"""

import argparse
import contextlib
import os
import statistics
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="MADE evaluation with selectable GPUs, batch size and profiling."
    )
    parser.add_argument(
        "--config-file",
        "--config_file",
        dest="config_file",
        required=True,
        help="Path to a YAML config.",
    )
    parser.add_argument(
        "--weight",
        default="",
        help="Checkpoint path. Defaults to TEST.WEIGHT from the config.",
    )
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated physical GPU IDs, for example: 0 or 0,1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Global inference batch size. Defaults to DATA.TEST_BATCH.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults to DATA.NUM_WORKERS.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=3,
        help="Warm-up forwards before measured inference.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision during inference.",
    )
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="Skip the extra one-image FLOPs profiling forward.",
    )
    args, config_opts = parser.parse_known_args()
    if config_opts and config_opts[0] == "--":
        config_opts = config_opts[1:]
    if any(option.startswith("--") for option in config_opts):
        parser.error(
            "Unknown script option in: {}. Config overrides must be KEY VALUE pairs."
            .format(" ".join(config_opts))
        )
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be greater than 0")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches cannot be negative")
    return args, config_opts


def human_count(value):
    if value >= 1e9:
        return "{:.3f} G".format(value / 1e9)
    if value >= 1e6:
        return "{:.3f} M".format(value / 1e6)
    if value >= 1e3:
        return "{:.3f} K".format(value / 1e3)
    return str(value)


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def synchronize_all(torch):
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def prepare_meta(cfg, meta, device, torch):
    if not cfg.MODEL.ADD_META:
        return None
    if meta is None:
        raise RuntimeError("MODEL.ADD_META=True, but the dataset did not return attributes.")
    meta = meta.to(device=device, dtype=torch.float32, non_blocking=True)
    if cfg.MODEL.MASK_META:
        meta[:, 5:21] = 0
        meta[:, 35:57] = 0
        meta[:, 84] = 0
        meta[:, 90] = 0
        meta[:, 92:96] = 0
        meta[:, 97] = 0
        meta[:, 100] = 0
        meta[:, 102:105] = 0
    if cfg.TEST.TYPE == "image_only":
        meta = torch.zeros_like(meta)
    return meta


def unpack_batch(batch):
    if len(batch) == 6:
        images, pids, camids, clothes_id, _, meta = batch
    elif len(batch) == 5:
        images, pids, camids, clothes_id, _ = batch
        meta = None
    else:
        raise RuntimeError("Unexpected validation batch with {} fields.".format(len(batch)))
    return images, pids, camids, clothes_id, meta


def forward_model(model, images, meta, cfg):
    if cfg.MODEL.ADD_META:
        return model(images, meta)
    return model(images)


def build_eval_jobs(cfg, dataset):
    from torch.utils.data import ConcatDataset, DataLoader
    from data import build_img_transforms
    from data.dataset_loader import ImageDataset
    from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC

    _, transform = build_img_transforms(cfg)

    def wrapped(items):
        return ImageDataset(items, transform=transform, aux_info=cfg.DATA.AUX_INFO)

    gallery = wrapped(dataset.gallery)
    common = {
        "batch_size": cfg.DATA.TEST_BATCH,
        "num_workers": cfg.DATA.NUM_WORKERS,
        "pin_memory": cfg.DATA.PIN_MEMORY,
        "drop_last": False,
        "shuffle": False,
    }
    jobs = []

    if cfg.DATA.DATASET == "prcc":
        definitions = [
            ("clothes-changing", dataset.query_diff, dataset.num_query_imgs_diff),
            ("standard", dataset.query_same, dataset.num_query_imgs_same),
        ]
        for name, query_items, num_query in definitions:
            loader = DataLoader(ConcatDataset([wrapped(query_items), gallery]), **common)
            evaluator = R1_mAP_eval(
                num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM
            )
            jobs.append((name, loader, evaluator, False))
    elif cfg.DATA.DATASET == "ltcc":
        combined = ConcatDataset([wrapped(dataset.query), gallery])
        jobs.append(
            (
                "clothes-changing",
                DataLoader(combined, **common),
                R1_mAP_eval_LTCC(
                    dataset.num_query_imgs,
                    max_rank=50,
                    feat_norm=cfg.TEST.FEAT_NORM,
                ),
                True,
            )
        )
        jobs.append(
            (
                "standard",
                DataLoader(combined, **common),
                R1_mAP_eval(
                    dataset.num_query_imgs,
                    max_rank=50,
                    feat_norm=cfg.TEST.FEAT_NORM,
                ),
                False,
            )
        )
    else:
        combined = ConcatDataset([wrapped(dataset.query), gallery])
        jobs.append(
            (
                "default",
                DataLoader(combined, **common),
                R1_mAP_eval(
                    dataset.num_query_imgs,
                    max_rank=50,
                    feat_norm=cfg.TEST.FEAT_NORM,
                ),
                False,
            )
        )
    return jobs


def profile_flops(model, cfg, device, use_amp, torch, logger):
    try:
        from torch.profiler import ProfilerActivity, profile

        images = torch.zeros(
            1,
            3,
            cfg.DATA.IMG_HEIGHT,
            cfg.DATA.IMG_WIDTH,
            device=device,
        )
        meta = None
        if cfg.MODEL.ADD_META:
            meta = torch.zeros(1, sum(cfg.MODEL.META_DIMS), device=device)
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        synchronize_all(torch)
        amp_context = (
            torch.cuda.amp.autocast(enabled=True)
            if use_amp
            else contextlib.nullcontext()
        )
        with profile(activities=activities, with_flops=True) as profiler:
            with torch.no_grad(), amp_context:
                forward_model(model, images, meta, cfg)
            synchronize_all(torch)
        flops = sum((event.flops or 0) for event in profiler.key_averages())
        if flops:
            logger.info(
                "FLOPs: %s/image (PyTorch operator estimate; MACs ≈ %s/image)",
                human_count(flops),
                human_count(flops / 2.0),
            )
        else:
            logger.warning(
                "FLOPs profiler returned 0; this PyTorch build does not count "
                "the model's operators. Parameters and runtime metrics remain valid."
            )
        del images, meta
        torch.cuda.empty_cache()
        return flops
    except Exception as error:
        logger.warning("FLOPs profiling unavailable: %s", error)
        return None


def warm_up(model, loader, cfg, device, count, use_amp, torch):
    if count == 0:
        return
    completed = 0
    with torch.no_grad():
        for batch in loader:
            images, _, _, _, meta = unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            meta = prepare_meta(cfg, meta, device, torch)
            with torch.cuda.amp.autocast(enabled=use_amp):
                forward_model(model, images, meta, cfg)
            completed += 1
            if completed >= count:
                break
    synchronize_all(torch)


def evaluate_job(
    name, model, loader, evaluator, clothes_changing, cfg, device, use_amp, torch, logger
):
    evaluation_start = time.perf_counter()
    evaluator.reset()
    batch_latencies = []
    sample_count = 0
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    synchronize_all(torch)
    wall_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            images, pids, camids, clothes_id, meta = unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            meta = prepare_meta(cfg, meta, device, torch)

            synchronize_all(torch)
            model_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=use_amp):
                features = forward_model(model, images, meta, cfg)
            synchronize_all(torch)
            batch_latencies.append(time.perf_counter() - model_start)
            sample_count += images.shape[0]

            if clothes_changing:
                evaluator.update((features, pids, camids, clothes_id))
            else:
                evaluator.update((features, pids, camids))
    wall_seconds = time.perf_counter() - wall_start
    model_seconds = sum(batch_latencies)
    cmc, mean_ap, _, _, _, _, _ = evaluator.compute()
    evaluation_seconds = time.perf_counter() - evaluation_start

    logger.info("[%s] Rank-1: %.1f%%", name, cmc[0] * 100)
    for rank in (5, 10):
        if len(cmc) >= rank:
            logger.info("[%s] Rank-%d: %.1f%%", name, rank, cmc[rank - 1] * 100)
    logger.info("[%s] mAP: %.1f%%", name, mean_ap * 100)
    logger.info(
        "[%s] model latency/batch: mean %.2f ms, P50 %.2f ms, P95 %.2f ms",
        name,
        statistics.mean(batch_latencies) * 1000,
        percentile(batch_latencies, 0.50) * 1000,
        percentile(batch_latencies, 0.95) * 1000,
    )
    logger.info(
        "[%s] model throughput: %.2f images/s; end-to-end: %.2f images/s "
        "(%d images, %.3f s including data loading)",
        name,
        sample_count / model_seconds,
        sample_count / wall_seconds,
        sample_count,
        wall_seconds,
    )
    logger.info("[%s] total model forward time: %.3f s", name, model_seconds)
    logger.info(
        "[%s] total feature extraction time: %.3f s (data loading + forward)",
        name,
        wall_seconds,
    )
    logger.info(
        "[%s] total evaluation time: %.3f s "
        "(data loading + forward + distance matrix + Rank/mAP)",
        name,
        evaluation_seconds,
    )
    for index in range(torch.cuda.device_count()):
        logger.info(
            "[%s] GPU %d peak memory: allocated %.2f MiB, reserved %.2f MiB",
            name,
            index,
            torch.cuda.max_memory_allocated(index) / (1024 ** 2),
            torch.cuda.max_memory_reserved(index) / (1024 ** 2),
        )


def main():
    program_start = time.perf_counter()
    args, config_opts = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("--devices must contain at least one GPU ID")

    # CUDA_VISIBLE_DEVICES must be assigned before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    import torch
    from config import cfg
    from data import build_dataset
    from model import build_model
    from utils.logger import setup_logger

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(config_opts)
    if args.batch_size is not None:
        cfg.DATA.TEST_BATCH = args.batch_size
    if args.workers is not None:
        cfg.DATA.NUM_WORKERS = args.workers
    if args.weight:
        cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    if not cfg.TEST.WEIGHT:
        raise SystemExit(
            "No checkpoint specified. Use --weight PATH or set TEST.WEIGHT in the config."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the current Python environment.")
    if torch.cuda.device_count() != len(devices):
        raise SystemExit(
            "Requested {} GPU(s) via --devices {}, but PyTorch sees {}."
            .format(len(devices), args.devices, torch.cuda.device_count())
        )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger(
        "EVA-attribure.inference",
        cfg.OUTPUT_DIR,
        if_train=False,
        filename="inference_log.txt",
    )
    logger.info("Physical GPUs: %s", ",".join(devices))
    logger.info(
        "Parallel mode: %s; global batch size: %d; approximately %d per GPU",
        "DataParallel" if len(devices) > 1 else "single GPU",
        cfg.DATA.TEST_BATCH,
        (cfg.DATA.TEST_BATCH + len(devices) - 1) // len(devices),
    )
    logger.info("AMP: %s; output directory: %s", args.amp, cfg.OUTPUT_DIR)

    dataset = build_dataset(cfg)
    model = build_model(cfg, dataset.num_train_pids)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(torch.device("cuda:0"))
    model.eval()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    logger.info(
        "Parameters: %s total (%s trainable)",
        human_count(parameters),
        human_count(trainable),
    )
    logger.info(
        "Model state size: %.2f MiB (parameters + buffers)",
        (parameter_bytes + buffer_bytes) / (1024 ** 2),
    )

    if not args.skip_flops:
        profile_flops(model, cfg, torch.device("cuda:0"), args.amp, torch, logger)

    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(len(devices))))

    jobs = build_eval_jobs(cfg, dataset)
    for name, loader, evaluator, clothes_changing in jobs:
        logger.info("Starting %s evaluation (%d batches)", name, len(loader))
        warm_up(
            model,
            loader,
            cfg,
            torch.device("cuda:0"),
            args.warmup_batches,
            args.amp,
            torch,
        )
        evaluate_job(
            name,
            model,
            loader,
            evaluator,
            clothes_changing,
            cfg,
            torch.device("cuda:0"),
            args.amp,
            torch,
            logger,
        )
    synchronize_all(torch)
    logger.info(
        "Total inference script time: %.3f s",
        time.perf_counter() - program_start,
    )


if __name__ == "__main__":
    main()
