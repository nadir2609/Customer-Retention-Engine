from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .schemas import CustomerFeatures


@dataclass(frozen=True)
class ModelArtifacts:
    model: Any
    pipeline: Any
    feature_columns: list[str]
    numerical_features: list[str]
    best_model_name: str


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

CATEGORIES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "subscription_type": ["Basic", "Premium", "Standard"],
    "contract_length": ["Monthly", "Quarterly", "Annual"],
    "age_group": ["Child", "Youth", "Adult", "Senior"],
}

NUMERIC_RANGES: dict[str, dict[str, float]] = {
    "tenure": {"min": 1, "max": 60, "default": 32},
    "usage_frequency": {"min": 1, "max": 30, "default": 16},
    "support_calls": {"min": 0, "max": 10, "default": 3},
    "payment_delay": {"min": 0, "max": 30, "default": 12},
    "total_spend": {"min": 100, "max": 1000, "default": 661},
    "last_interaction": {"min": 1, "max": 30, "default": 14},
}

NUMERIC_ACTIONS: dict[str, tuple[str, float, float, float, str]] = {
    "Total Spend": ("increase", 50, 0, 2000, "$"),
    "Payment Delay": ("decrease", 1, 0, 50, "days"),
    "Support Calls": ("decrease", 1, 0, 15, "calls"),
}

CONTRACT_ORDER = ["Monthly", "Quarterly", "Annual"]


@lru_cache(maxsize=1)
def get_model_artifacts() -> ModelArtifacts:
    best_model = joblib.load(MODEL_DIR / "best_model.joblib")
    pipeline = joblib.load(MODEL_DIR / "preprocessing_pipeline.joblib")
    column_info = joblib.load(MODEL_DIR / "column_info.joblib")

    return ModelArtifacts(
        model=best_model,
        pipeline=pipeline,
        feature_columns=column_info["feature_columns"],
        numerical_features=column_info["numerical_features"],
        best_model_name=column_info["best_model_name"],
    )


def get_metadata() -> dict[str, Any]:
    artifacts = get_model_artifacts()
    return {
        "model_name": artifacts.best_model_name,
        "feature_columns": artifacts.feature_columns,
        "categories": CATEGORIES,
        "numeric_ranges": NUMERIC_RANGES,
    }


def predict_customer(customer: CustomerFeatures) -> dict[str, Any]:
    artifacts = get_model_artifacts()
    row = _customer_to_dataframe(customer, artifacts)
    prediction, churn_probability = _predict(row, artifacts)
    recommendation = _recommend(row, artifacts)

    return {
        "prediction": "Churn" if prediction == 1 else "Not Churn",
        "churn_probability": churn_probability,
        "retention_probability": 1 - churn_probability,
        "recommendation": recommendation,
    }


def recommend_customer(customer: CustomerFeatures) -> dict[str, Any]:
    artifacts = get_model_artifacts()
    row = _customer_to_dataframe(customer, artifacts)
    return _recommend(row, artifacts)


def _customer_to_dataframe(customer: CustomerFeatures, artifacts: ModelArtifacts) -> pd.DataFrame:
    row = pd.DataFrame([customer.to_model_row()])[artifacts.feature_columns].copy()
    for feature in artifacts.numerical_features:
        row[feature] = pd.to_numeric(row[feature], errors="coerce").astype(float)
    return row


def _predict(row_df: pd.DataFrame, artifacts: ModelArtifacts) -> tuple[int, float]:
    processed = artifacts.pipeline.transform(row_df)
    prediction = int(artifacts.model.predict(processed)[0])
    churn_probability = float(artifacts.model.predict_proba(processed)[0][1])
    return prediction, churn_probability


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _try_numeric_change(
    base_row: pd.DataFrame,
    feature: str,
    direction: str,
    step: float,
    min_value: float,
    max_value: float,
    artifacts: ModelArtifacts,
) -> tuple[float | None, int | None]:
    original = float(base_row.at[0, feature])

    extreme = max_value if direction == "increase" else min_value
    test = base_row.copy()
    test.at[0, feature] = extreme
    prediction, _ = _predict(test, artifacts)
    if prediction == 1:
        return None, None

    low, high = 0.0, abs(extreme - original)
    best_delta = high

    for _ in range(60):
        mid = (low + high) / 2
        candidate = original + mid if direction == "increase" else original - mid
        test = base_row.copy()
        test.at[0, feature] = candidate
        prediction, _ = _predict(test, artifacts)
        if prediction == 0:
            best_delta = mid
            high = mid
        else:
            low = mid
        if high - low < 0.5 * step:
            break

    steps_needed = max(1, int(np.ceil(best_delta / step)))
    new_value = original + (steps_needed * step if direction == "increase" else -steps_needed * step)
    new_value = max(min_value, min(max_value, new_value))

    test = base_row.copy()
    test.at[0, feature] = new_value
    prediction, _ = _predict(test, artifacts)
    if prediction == 1:
        new_value = new_value + step if direction == "increase" else new_value - step
        new_value = max(min_value, min(max_value, new_value))
        test.at[0, feature] = new_value
        prediction, _ = _predict(test, artifacts)
        if prediction == 1:
            return None, None

    return float(new_value), steps_needed


def _try_contract_change(base_row: pd.DataFrame, artifacts: ModelArtifacts) -> str | None:
    current = str(base_row.at[0, "Contract Length"])
    if current not in CONTRACT_ORDER:
        return None

    index = CONTRACT_ORDER.index(current)

    for new_index in range(index + 1, len(CONTRACT_ORDER)):
        test = base_row.copy()
        test.at[0, "Contract Length"] = CONTRACT_ORDER[new_index]
        prediction, _ = _predict(test, artifacts)
        if prediction == 0:
            return CONTRACT_ORDER[new_index]

    for new_index in range(index - 1, -1, -1):
        test = base_row.copy()
        test.at[0, "Contract Length"] = CONTRACT_ORDER[new_index]
        prediction, _ = _predict(test, artifacts)
        if prediction == 0:
            return CONTRACT_ORDER[new_index]

    return None


def _single_feature_candidates(base_row: pd.DataFrame, artifacts: ModelArtifacts) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for feature, (direction, step, min_value, max_value, unit) in NUMERIC_ACTIONS.items():
        new_value, _ = _try_numeric_change(
            base_row=base_row,
            feature=feature,
            direction=direction,
            step=step,
            min_value=min_value,
            max_value=max_value,
            artifacts=artifacts,
        )
        if new_value is None:
            continue

        original = float(base_row.at[0, feature])
        candidates.append(
            {
                "features_changed": [feature],
                "changes": {
                    feature: {
                        "from": original,
                        "to": new_value,
                        "delta": abs(new_value - original),
                        "unit": unit,
                        "direction": direction,
                    }
                },
            }
        )

    new_contract = _try_contract_change(base_row, artifacts)
    if new_contract is not None:
        original_contract = str(base_row.at[0, "Contract Length"])
        candidates.append(
            {
                "features_changed": ["Contract Length"],
                "changes": {
                    "Contract Length": {
                        "from": original_contract,
                        "to": new_contract,
                        "delta": 0,
                        "unit": "",
                        "direction": "change",
                    }
                },
            }
        )

    return candidates


def _multi_feature_candidate(base_row: pd.DataFrame, artifacts: ModelArtifacts) -> dict[str, Any] | None:
    actionable = list(NUMERIC_ACTIONS.keys()) + ["Contract Length"]
    partials: dict[str, Any] = {}

    for feature, (direction, step, min_value, max_value, _) in NUMERIC_ACTIONS.items():
        new_value, _ = _try_numeric_change(
            base_row=base_row,
            feature=feature,
            direction=direction,
            step=step,
            min_value=min_value,
            max_value=max_value,
            artifacts=artifacts,
        )
        partials[feature] = new_value if new_value is not None else (max_value if direction == "increase" else min_value)

    new_contract = _try_contract_change(base_row, artifacts)
    if new_contract is not None:
        partials["Contract Length"] = new_contract
    else:
        current = str(base_row.at[0, "Contract Length"])
        if current in CONTRACT_ORDER:
            index = CONTRACT_ORDER.index(current)
            partials["Contract Length"] = CONTRACT_ORDER[(index + 1) % len(CONTRACT_ORDER)]

    for size in range(2, len(actionable) + 1):
        for combo in combinations(actionable, size):
            test = base_row.copy()
            changes: dict[str, Any] = {}
            for feature in combo:
                if feature not in partials:
                    continue

                original = _to_python(test.at[0, feature])
                test.at[0, feature] = partials[feature]

                if feature == "Contract Length":
                    changes[feature] = {
                        "from": original,
                        "to": partials[feature],
                        "delta": 0,
                        "unit": "",
                        "direction": "change",
                    }
                else:
                    direction, _, _, _, unit = NUMERIC_ACTIONS[feature]
                    changes[feature] = {
                        "from": float(original),
                        "to": float(partials[feature]),
                        "delta": abs(float(partials[feature]) - float(original)),
                        "unit": unit,
                        "direction": direction,
                    }

            if len(changes) != len(combo):
                continue

            prediction, _ = _predict(test, artifacts)
            if prediction != 0:
                continue

            minimized = _minimize_combo(base_row, combo, partials, artifacts)
            if minimized is not None:
                return {"features_changed": list(combo), "changes": minimized}

    return None


def _minimize_combo(
    base_row: pd.DataFrame,
    combo: tuple[str, ...],
    partials: dict[str, Any],
    artifacts: ModelArtifacts,
) -> dict[str, Any] | None:
    current = base_row.copy()
    for feature in combo:
        current.at[0, feature] = partials[feature]

    changes: dict[str, Any] = {}

    for feature in combo:
        if feature == "Contract Length":
            changes[feature] = {
                "from": str(base_row.at[0, feature]),
                "to": str(partials[feature]),
                "delta": 0,
                "unit": "",
                "direction": "change",
            }
            continue

        direction, step, min_value, max_value, unit = NUMERIC_ACTIONS[feature]
        original = float(base_row.at[0, feature])
        target = float(partials[feature])
        low, high = 0.0, abs(target - original)
        best_delta = high

        for _ in range(60):
            mid = (low + high) / 2
            candidate = original + mid if direction == "increase" else original - mid
            test = current.copy()
            test.at[0, feature] = candidate
            prediction, _ = _predict(test, artifacts)
            if prediction == 0:
                best_delta = mid
                high = mid
            else:
                low = mid
            if high - low < 0.5 * step:
                break

        steps_needed = max(1, int(np.ceil(best_delta / step)))
        new_value = original + (steps_needed * step if direction == "increase" else -steps_needed * step)
        new_value = max(min_value, min(max_value, new_value))

        test = current.copy()
        test.at[0, feature] = new_value
        prediction, _ = _predict(test, artifacts)
        if prediction == 1:
            new_value = new_value + step if direction == "increase" else new_value - step
            new_value = max(min_value, min(max_value, new_value))

        current.at[0, feature] = new_value
        changes[feature] = {
            "from": original,
            "to": float(new_value),
            "delta": abs(float(new_value) - original),
            "unit": unit,
            "direction": direction,
        }

    prediction, _ = _predict(current, artifacts)
    if prediction == 1:
        return None

    return changes


def _recommend(base_row: pd.DataFrame, artifacts: ModelArtifacts) -> dict[str, Any]:
    prediction, churn_probability = _predict(base_row, artifacts)

    if prediction == 0:
        return {
            "status": "NO_ACTION",
            "message": "Customer is not predicted to churn. No action needed.",
            "churn_probability": churn_probability,
        }

    single_candidates = _single_feature_candidates(base_row, artifacts)
    if single_candidates:
        best_single = min(
            single_candidates,
            key=lambda candidate: sum(change["delta"] for change in candidate["changes"].values()),
        )
        test = base_row.copy()
        for feature, change in best_single["changes"].items():
            test.at[0, feature] = change["to"]
        _, new_churn_probability = _predict(test, artifacts)
        return {
            "status": "SINGLE_CHANGE",
            "message": "One change is enough to prevent churn.",
            "original_churn_prob": churn_probability,
            "new_churn_prob": new_churn_probability,
            "recommendations": best_single["changes"],
        }

    multi_candidate = _multi_feature_candidate(base_row, artifacts)
    if multi_candidate is not None:
        test = base_row.copy()
        for feature, change in multi_candidate["changes"].items():
            test.at[0, feature] = change["to"]
        _, new_churn_probability = _predict(test, artifacts)
        return {
            "status": "MULTI_CHANGE",
            "message": f"Combined changes across {len(multi_candidate['features_changed'])} features needed.",
            "original_churn_prob": churn_probability,
            "new_churn_prob": new_churn_probability,
            "recommendations": multi_candidate["changes"],
        }

    return {
        "status": "HIGH_RISK",
        "message": "Cannot prevent churn with available actions. Escalate to retention team.",
        "churn_probability": churn_probability,
    }
