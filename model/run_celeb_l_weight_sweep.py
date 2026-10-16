import argparse
import os
import subprocess
import sys
from decimal import Decimal


def decimal_range(start, end, step):
    current = Decimal(str(start))
    end = Decimal(str(end))
    step = Decimal(str(step))
    if step <= 0:
        raise ValueError("--weight-step must be greater than 0")

    while current <= end:
        yield current
        current += step


def format_weight(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def join_output_dir(root, name):
    separator = "\\" if "\\" in root and "/" not in root else "/"
    return root.rstrip("/\\") + separator + name


def main():
    parser = argparse.ArgumentParser(
        description="Run LTCC training with all combinations of ID and Triplet loss weights."
    )
    parser.add_argument(
        "--config-file",
        default="configs/Celeb_light/eva02_l_maskmeta_random.yml",
        help="Training config file.",
    )
    parser.add_argument("--devices", default="1,2", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=6666)

    parser.add_argument("--id-start", default="0.1", help="First ID loss weight.")
    parser.add_argument("--id-end", default="1.0", help="Last ID loss weight.")
    parser.add_argument("--id-step", default="0.1", help="ID loss weight step.")

    parser.add_argument(
        "--triplet-start",
        default="0.1",
        help="First triplet loss weight.",
    )
    parser.add_argument(
        "--triplet-end",
        default="1.0",
        help="Last triplet loss weight.",
    )
    parser.add_argument(
        "--triplet-step",
        default="0.1",
        help="Triplet loss weight step.",
    )

    parser.add_argument(
        "--output-root",
        default="./logs/celeb-light-weight-sweep",
        help="Root directory for all sweep outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without starting training.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices

    total_runs = 0
    for _ in decimal_range(args.id_start, args.id_end, args.id_step):
        for _ in decimal_range(args.triplet_start, args.triplet_end, args.triplet_step):
            total_runs += 1

    run_idx = 0
    for id_weight in decimal_range(args.id_start, args.id_end, args.id_step):
        id_weight_text = format_weight(id_weight)

        for triplet_weight in decimal_range(
            args.triplet_start, args.triplet_end, args.triplet_step
        ):
            triplet_weight_text = format_weight(triplet_weight)
            run_idx += 1

            output_dir = join_output_dir(
                args.output_root,
                f"id-{id_weight_text}-tri-{triplet_weight_text}"
            )

            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.launch",
                f"--nproc_per_node={args.nproc_per_node}",
                f"--master_port={args.master_port}",
                "train.py",
                "--config_file",
                args.config_file,
                "MODEL.DIST_TRAIN",
                "True",
                "MODEL.ID_LOSS_WEIGHT",
                id_weight_text,
                "MODEL.TRIPLET_LOSS_WEIGHT",
                triplet_weight_text,
                "OUTPUT_DIR",
                output_dir,
            ]

            print("=" * 80, flush=True)
            print(
                f"[{run_idx}/{total_runs}] "
                f"Running ID_LOSS_WEIGHT={id_weight_text}, "
                f"TRIPLET_LOSS_WEIGHT={triplet_weight_text}",
                flush=True,
            )
            print("CUDA_VISIBLE_DEVICES=" + args.devices, flush=True)
            print("OUTPUT_DIR=" + output_dir, flush=True)
            print(" ".join(cmd), flush=True)

            if args.dry_run:
                continue

            subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
