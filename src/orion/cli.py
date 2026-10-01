"""Command-line interface for AeroNetra-Orion.

    orion demo            # run the download-free synthetic end-to-end demo
    orion run --config …  # run with an explicit configuration file
    orion list-datasets   # list supported event & aerial datasets
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .datasets import list_datasets


def _cmd_demo(args) -> int:
    from .pipeline import run_demo

    cfg = load_config(args.config) if args.config else load_config(None)
    if args.out:
        cfg.output_dir = args.out
    summary = run_demo(cfg, out_dir=cfg.output_dir)
    print("AeroNetra-Orion — demo complete")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def _cmd_run(args) -> int:
    from .pipeline import run_demo

    cfg = load_config(args.config)
    if args.out:
        cfg.output_dir = args.out
    summary = run_demo(cfg, out_dir=cfg.output_dir)
    print("AeroNetra-Orion — run complete")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def _cmd_list_datasets(_args) -> int:
    print(f"{'key':<14}{'kind':<9}{'modality':<26}name")
    print("-" * 78)
    for spec in list_datasets():
        print(f"{spec.key:<14}{spec.kind:<9}{spec.modality:<26}{spec.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orion",
        description="AeroNetra-Orion — event-driven predictive perception for UAVs.",
    )
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="Run the download-free synthetic demo")
    demo.add_argument("--config", default=None, help="Path to a YAML config")
    demo.add_argument("--out", default=None, help="Output directory")
    demo.set_defaults(func=_cmd_demo)

    run = sub.add_parser("run", help="Run with an explicit config file")
    run.add_argument("--config", required=True, help="Path to a YAML config")
    run.add_argument("--out", default=None, help="Output directory")
    run.set_defaults(func=_cmd_run)

    ls = sub.add_parser("list-datasets", help="List supported datasets")
    ls.set_defaults(func=_cmd_list_datasets)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
