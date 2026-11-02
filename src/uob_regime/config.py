"""Load project settings without depending on the current working directory."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def workspace_root(config):
    """Resolve an optional study workspace without allowing path traversal."""
    value = config.get("workspace")
    if value is None:
        return ROOT
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("workspace must be a safe path relative to the project root")
    return ROOT / relative


def load_config(path=None):
    config_path = Path(path).resolve() if path else ROOT / "configs/default.json"
    config = json.loads(config_path.read_text())
    return validate_config(config)


def validate_config(config):
    workspace_root(config)
    if config["price_basis"] not in {"adjusted", "close"}:
        raise ValueError("price_basis must be adjusted or close")
    for key in ("lookback", "volatility_window", "trend_window", "drawdown_window",
                "epochs", "batch_size", "patience", "hidden_units", "annualization"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= config["cost_bps"] < 10000:
        raise ValueError("cost_bps must be in [0, 10000)")
    if config["entropy_strength"] < 0:
        raise ValueError("entropy_strength must be nonnegative")
    if not config["seeds"] or not config["models"]:
        raise ValueError("At least one model and seed are required")
    if config["risk_free_rate"] <= -1:
        raise ValueError("risk_free_rate must exceed -1")
    if max(config["test_years"]) >= int(config["holdout_start"][:4]):
        raise ValueError("Historical test years must precede the final held-out period")
    if config.get("final_refit", False):
        epochs = config.get("refit_epochs")
        trained = set(config["models"]) - {"zero", "persistence"}
        if not isinstance(epochs, dict) or set(epochs) != trained:
            raise ValueError("refit_epochs must contain every trainable model exactly once")
        if any(not isinstance(value, int) or value < 1 for value in epochs.values()):
            raise ValueError("Every fixed refit epoch count must be a positive integer")
    return config
