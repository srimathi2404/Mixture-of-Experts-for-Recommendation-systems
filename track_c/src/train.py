"""Training entry point for Track C (self-attention-expert MoSE).

Loads configs/dataset.yaml + configs/sasrec_mose.yaml via RecBole's Config,
builds the dataset/dataloaders, instantiates SASRecMoSE, and trains it with
RecBole's stock Trainer (handles the optimizer loop, early stopping on
valid_metric, checkpointing, and tensorboard logging for us - see
src/models/sasrec_mose.py's calculate_loss docstring for how per-task losses
still get logged separately without a custom training loop).

Every run gets a `--run-name` and writes to track_c/results/<run-name>/:
    checkpoints/                 RecBole model checkpoint(s)
    metrics.json                 valid/test metrics + config actually used
    expert_utilization.png       core deliverable #2 (also extra experiment #3)
    utilization_summary.json     same data as the plot, machine-readable

Arbitrary config overrides (for the ablations in docs/extra_experiments.md)
are passed as `--override key=value ...`; values are parsed with YAML so
`true`/`false`/numbers/lists work as expected. Examples:

    python -m track_c.src.train --run-name core
    python -m track_c.src.train --run-name no_pos --override use_positional_embedding=false
    python -m track_c.src.train --run-name seqlen20 --override MAX_ITEM_LIST_LENGTH=20
    python -m track_c.src.train --run-name lb_on --seed 3 --override load_balancing_coef=0.01

Run from the repo root.
"""

import argparse
import json
from pathlib import Path

import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from .models.sasrec_mose import SASRecMoSE
from .utils.expert_utilization import ExpertUtilizationTracker

TRACK_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = TRACK_DIR / "configs"
RESULTS_DIR = TRACK_DIR / "results"

DEFAULT_CONFIG_FILES = [
    str(CONFIG_DIR / "dataset.yaml"),
    str(CONFIG_DIR / "sasrec_mose.yaml"),
]

# Config keys whose *resolved* value (default or overridden) is worth saving
# unconditionally, so downstream analysis scripts (seqlen_sweep.py,
# polarization.py) don't have to guess whether a run used a default or an
# explicit override.
TRACKED_KEYS = [
    "seed",
    "MAX_ITEM_LIST_LENGTH",
    "n_experts",
    "use_positional_embedding",
    "load_balancing_coef",
    "task_loss_weights",
]


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--override expects key=value, got: {pair!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def build_config(config_files, overrides, run_name):
    config_dict = dict(overrides)
    config_dict.setdefault("checkpoint_dir", str(RESULTS_DIR / run_name / "checkpoints"))
    return Config(
        model=SASRecMoSE,
        config_file_list=[str(f) for f in config_files],
        config_dict=config_dict,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="core", help="results/<run-name>/ output folder")
    parser.add_argument("--config-files", nargs="+", default=DEFAULT_CONFIG_FILES)
    parser.add_argument("--override", nargs="*", default=[], metavar="key=value")
    parser.add_argument("--seed", type=int, default=None, help="shorthand for --override seed=N")
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    if args.seed is not None:
        overrides["seed"] = args.seed

    config = build_config(args.config_files, overrides, args.run_name)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = SASRecMoSE(config, train_data.dataset).to(config["device"])

    tracker = ExpertUtilizationTracker(
        config["n_tasks"], config["n_experts"], task_names=["next_item", "rating"]
    )
    model.util_tracker = tracker

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    run_dir = RESULTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker.plot(str(run_dir / "expert_utilization.png"))
    tracker.save_json(str(run_dir / "utilization_summary.json"))

    report = {
        "run_name": args.run_name,
        "checkpoint": str(trainer.saved_model_file),
        "config_overrides": overrides,
        "resolved": {key: config[key] for key in TRACKED_KEYS},
        "best_valid_score": best_valid_score,
        "best_valid_result": {str(k): float(v) for k, v in best_valid_result.items()},
        "test_result": {str(k): float(v) for k, v in test_result.items()},
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
