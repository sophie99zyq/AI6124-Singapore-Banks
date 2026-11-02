"""Model identifiers and a single integration point for teammate extensions."""

CORE_MODELS = {"zero", "persistence", "lstm", "fuzzy_feature", "fuzzy_moe",
               "capacity_lstm", "equal_moe", "dlinear", "anfis"}


def validate_model_names(names):
    unknown = set(names) - CORE_MODELS
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
