"""Decision-support guidance driven by the predicted category and the user's own inputs."""
from __future__ import annotations

import pandas as pd

from utils.config import TARGET_COL

BASE = {
    "High": [
        "Review preparation quantities before cooking starts.",
        "Consider preparing in smaller batches rather than all at once.",
        "Monitor actual demand before preparing additional food.",
        "Review previous wastage records for similar events.",
    ],
    "Medium": [
        "Monitor preparation quantities during service.",
        "Compare expected demand with previous observations.",
        "Adjust batch sizes where appropriate.",
    ],
    "Low": [
        "Continue monitoring wastage.",
        "Maintain current preparation and storage practices.",
    ],
}


def build_recommendations(category: str, values: dict, dataset: pd.DataFrame | None) -> dict:
    """Returns {'core': [...], 'context': [...]} using only inputs the saved models actually take."""
    core = list(BASE.get(category, []))
    context = []

    guests, qty = values.get("Number of Guests"), values.get("Quantity of Food")
    if dataset is not None and guests and qty:
        ratio = qty / guests
        ref = (dataset["Quantity of Food"] / dataset["Number of Guests"])
        lo, hi = ref.quantile(0.25), ref.quantile(0.75)
        if ratio > hi:
            context.append(
                f"You plan {ratio:.2f} units of food per guest, above the typical {lo:.2f}-{hi:.2f} in the "
                "supplied dataset. Check whether the quantity can be trimmed."
            )
        elif ratio < lo:
            context.append(
                f"You plan {ratio:.2f} units of food per guest, below the typical {lo:.2f}-{hi:.2f} in the "
                "supplied dataset. Make sure supply still matches demand."
            )

    if values.get("Storage Conditions") == "Room Temperature" and values.get("Type of Food") in (
            "Dairy Products", "Meat", "Fruits", "Vegetables"):
        context.append(
            f"{values['Type of Food']} is planned at room temperature. Review whether refrigeration "
            "is practical for the unserved portion."
        )

    if dataset is not None:
        for col, label in (("Event Type", "event type"), ("Preparation Method", "preparation method")):
            v = values.get(col)
            sub = dataset[dataset[col] == v][TARGET_COL]
            if len(sub) >= 30:
                context.append(
                    f"In the supplied dataset, '{v}' ({label}) records average {sub.mean():.1f} "
                    f"wastage vs {dataset[TARGET_COL].mean():.1f} overall."
                )
    return {"core": core, "context": context}
