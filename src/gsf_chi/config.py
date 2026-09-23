"""JSON experiment configuration with explicit command-line overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "--config", type=Path, help="JSON configuration; CLI arguments override its values."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved configuration and exit."
    )
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    path = probe.parse_known_args()[0].config
    if path is not None:
        try:
            config = json.loads(path.read_text())
            values = config["args"]
            if not isinstance(values, dict):
                raise ValueError("args must be an object")
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"{path}: {exc}")
        actions = {a.dest: a for a in parser._actions}
        unknown = values.keys() - actions.keys()
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        defaults = {}
        for key, value in values.items():
            action = actions[key]
            items = value if isinstance(value, list) else [value]
            if action.choices and any((item not in action.choices for item in items)):
                parser.error(f"Invalid value for {key}: {value!r}; expected {action.choices}")
            try:
                if action.type is not None and value is not None:
                    value = (
                        [action.type(item) for item in value]
                        if isinstance(value, list)
                        else action.type(value)
                    )
            except (TypeError, ValueError) as exc:
                parser.error(f"{key}: {exc}")
            defaults[key] = value
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if getattr(args, "epochs", 1) < 1:
        parser.error("--epochs must be positive")
    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size must be positive")
    if getattr(args, "micro_batch_size", None) is not None and args.micro_batch_size < 1:
        parser.error("--micro-batch-size must be positive")
    if not 0 <= getattr(args, "mirror_loss_weight", 0) <= 1:
        parser.error("--mirror-loss-weight must be in [0, 1]")
    if getattr(args, "mirror_consistency_weight", 0) and (not args.mirror_loss_weight):
        parser.error("--mirror-consistency-weight requires --mirror-loss-weight")
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        parser.exit()
    return args
