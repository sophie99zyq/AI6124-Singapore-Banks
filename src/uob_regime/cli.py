"""Command-line entry points; held-out evaluation requires a frozen protocol."""

import argparse
import hashlib
import json
from pathlib import Path

from .config import ROOT, load_config, workspace_root


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, help="Custom JSON configuration")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download the configured bank and STI snapshots")
    commands.add_parser("prepare", help="Validate and export features, samples and fold dates")
    analysis = commands.add_parser("analyze", help="Analyze saved forecasts without retraining")
    analysis.add_argument("--run-dir", type=Path, required=True)
    analysis.add_argument("--cost-bps", type=float, nargs="+", default=[0, 5, 10, 20])
    historical = commands.add_parser("run", help="Run historical folds only")
    historical.add_argument("--folds", type=int, nargs="+")
    historical.add_argument("--models", nargs="+")
    historical.add_argument("--seeds", type=int, nargs="+")
    historical.add_argument("--epochs", type=int)
    freeze = commands.add_parser("freeze", help="Snapshot a reviewed final protocol")
    freeze.add_argument("--output", type=Path, default=ROOT / "configs/frozen_final.json")
    heldout = commands.add_parser("run-holdout", help="Explicit final held-out evaluation")
    heldout.add_argument("--frozen-config", type=Path, required=True)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.command == "analyze":
        from .analysis import analyze_run
        print(analyze_run(arguments.run_dir, arguments.cost_bps))
        return
    if arguments.command == "run-holdout":
        if arguments.config:
            raise SystemExit("Use only --frozen-config for held-out evaluation")
        from .experiment import run_experiments

        path = arguments.frozen_config.resolve()
        frozen = json.loads(path.read_text())
        config = frozen["config"]
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if frozen["config_sha256"] != digest:
            raise ValueError("Frozen protocol hash mismatch")
        # Validate the same schema used by historical experiments.
        from .config import validate_config
        validate_config(config)
        print(run_experiments(config, mode="holdout", frozen_protocol={"path": str(path), "sha256": digest}))
        return
    config = load_config(arguments.config)
    if arguments.command == "download":
        from .data import download
        print(json.dumps(download(config, workspace_root(config) / "data/raw"), indent=2))
    elif arguments.command == "prepare":
        from .experiment import prepare
        print(prepare(config).to_string(index=False))
    elif arguments.command == "freeze":
        from .models import validate_model_names
        validate_model_names(config["models"])
        output = arguments.output.resolve()
        if output.exists():
            raise FileExistsError("Choose a new protocol path; existing protocols are retained")
        frozen = {"config": config, "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "note": "Settings frozen before final evaluation; this is not evidence of unseen data."}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(frozen, indent=2))
        print(output)
    elif arguments.command == "run":
        from .config import validate_config
        from .experiment import run_experiments
        for key in ("models", "seeds", "epochs"):
            value = getattr(arguments, key)
            if value is not None:
                config[key] = value
        validate_config(config)
        print(run_experiments(config, years=arguments.folds))


if __name__ == "__main__":
    main()
