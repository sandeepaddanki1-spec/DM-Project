"""Validation and prediction using the saved pipelines. Preprocessing lives inside the pipelines."""
from __future__ import annotations

import pandas as pd

from utils.config import CATEGORY_ORDER, TARGET_COL
from utils.model_loader import Models


def validate_inputs(values: dict, models: Models, ranges: dict[str, tuple[int, int]]):
    """Returns (errors, warnings). Errors block prediction; warnings do not."""
    errors, warnings = [], []
    schema = models.schema
    for col in schema.features:
        if col not in values or values[col] in (None, ""):
            errors.append(f"'{col}' is missing.")
    for col in schema.numeric:
        v = values.get(col)
        if v is None:
            continue
        if v <= 0:
            errors.append(f"'{col}' must be greater than 0.")
        elif col in ranges and not (ranges[col][0] <= v <= ranges[col][1]):
            lo, hi = ranges[col]
            warnings.append(
                f"{col} = {v:,} is outside the range in the supplied dataset ({lo:,}-{hi:,}). "
                "The model has not seen values like this, so treat the result with extra caution."
            )
    for col, cats in schema.categorical.items():
        v = values.get(col)
        if v is not None and v not in cats:
            errors.append(f"'{col}' value '{v}' was not seen in training.")
    gq = values.get("Quantity of Food")
    gg = values.get("Number of Guests")
    if gq and gg and gq < gg * 0.25:
        warnings.append("Quantity of food is very low relative to the number of guests.")
    return errors, warnings


def build_input_frame(values: dict, models: Models) -> pd.DataFrame:
    """One-row DataFrame with the exact column names and order the pipelines were fitted on."""
    return pd.DataFrame([{c: values[c] for c in models.schema.features}], columns=models.schema.features)


def predict(values: dict, models: Models) -> dict:
    X = build_input_frame(values, models)
    reg_value = float(models.regressor.predict(X)[0])
    category = str(models.classifier.predict(X)[0])
    proba = models.classifier.predict_proba(X)[0]
    probs = {str(c): float(p) for c, p in zip(models.classifier.classes_, proba)}
    return {
        "input": X,
        "predicted_amount": reg_value,
        "category": category,
        "probabilities": {k: probs[k] for k in CATEGORY_ORDER if k in probs},
    }


def category_boundaries(dataset: pd.DataFrame | None) -> dict | None:
    """Tertile cut-points the notebook used (pd.qcut(q=3)) to create Low/Medium/High, recomputed from the dataset."""
    if dataset is None or TARGET_COL not in dataset:
        return None
    _, edges = pd.qcut(dataset[TARGET_COL], 3, labels=CATEGORY_ORDER, retbins=True)
    return {"low_max": float(edges[1]), "medium_max": float(edges[2]), "min": float(edges[0]), "max": float(edges[3])}


def band_for_amount(amount: float, bounds: dict | None) -> str | None:
    if not bounds:
        return None
    if amount <= bounds["low_max"]:
        return "Low"
    return "Medium" if amount <= bounds["medium_max"] else "High"
