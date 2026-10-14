"""Loads the two saved pipelines from the project's models/ directory only."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import joblib
import pandas as pd
import sklearn
import streamlit as st

from utils.config import (
    CLASSIFICATION_MODEL_PATH,
    DATASET_PATH,
    REGRESSION_MODEL_PATH,
    SAVED_SKLEARN_VERSION,
    TARGET_COL,
)


class ModelLoadError(RuntimeError):
    """Raised with a user-readable message when a saved model cannot be used safely."""


@dataclass
class Schema:
    """Input schema read from the saved pipelines (not typed in by hand)."""
    features: list[str]
    numeric: list[str]
    categorical: dict[str, list[str]]  # column -> categories seen in training
    classes: list[str]


@dataclass
class Models:
    regressor: object
    classifier: object
    schema: Schema


def _read_schema(reg, cls) -> Schema:
    reg_features = list(reg.feature_names_in_)
    cls_features = list(cls.feature_names_in_)
    if reg_features != cls_features:
        raise ModelLoadError(
            "The regression and classification models expect different input columns, "
            "so one shared form cannot be built safely."
        )
    pre = reg.named_steps["preprocessor"]
    numeric, categorical = [], {}
    for name, transformer, cols in pre.transformers_:
        if name == "num":
            numeric = list(cols)
        elif name == "cat":
            enc = transformer.named_steps["onehot"]
            categorical = {c: list(cats) for c, cats in zip(cols, enc.categories_)}
    return Schema(reg_features, numeric, categorical, list(cls.classes_))


@st.cache_resource(show_spinner="Loading saved models...")
def load_models() -> Models:
    for path in (REGRESSION_MODEL_PATH, CLASSIFICATION_MODEL_PATH):
        if not path.exists():
            raise ModelLoadError(f"Missing model file: models/{path.name}")
    if sklearn.__version__ != SAVED_SKLEARN_VERSION:
        raise ModelLoadError(
            f"The models were saved with scikit-learn {SAVED_SKLEARN_VERSION} but this environment "
            f"has {sklearn.__version__}. Run: pip install scikit-learn=={SAVED_SKLEARN_VERSION}"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a version-mismatch warning must not pass silently
            reg = joblib.load(REGRESSION_MODEL_PATH)
            cls = joblib.load(CLASSIFICATION_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Could not load the saved models: {exc}") from exc
    return Models(reg, cls, _read_schema(reg, cls))


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    if not DATASET_PATH.exists():
        return None
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_csv(path_str: str) -> pd.DataFrame | None:
    from pathlib import Path
    p = Path(path_str)
    return pd.read_csv(p) if p.exists() else None


def numeric_ranges(df: pd.DataFrame | None, schema: Schema) -> dict[str, tuple[int, int]]:
    """Observed min/max of numeric inputs in the supplied dataset (used for soft warnings only)."""
    if df is None:
        return {}
    return {c: (int(df[c].min()), int(df[c].max())) for c in schema.numeric if c in df}


def feature_importance(models: Models) -> pd.DataFrame:
    """Impurity-based importances stored in the fitted forests, summed back to the original columns."""
    rows = []
    for label, pipe in (("Regression", models.regressor), ("Classification", models.classifier)):
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
        imp = pipe.named_steps["model"].feature_importances_
        for n, v in zip(names, imp):
            orig = n.split("__", 1)[1]
            for col in models.schema.features:
                if orig == col or orig.startswith(col + "_"):
                    orig = col
                    break
            rows.append({"Model": label, "Feature": orig, "Importance": float(v)})
    out = pd.DataFrame(rows).groupby(["Model", "Feature"], as_index=False)["Importance"].sum()
    return out.sort_values(["Model", "Importance"], ascending=[True, False])


__all__ = ["load_models", "load_dataset", "load_csv", "numeric_ranges", "feature_importance",
           "ModelLoadError", "Models", "Schema", "TARGET_COL"]
