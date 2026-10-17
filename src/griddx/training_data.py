from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import joblib
import pandas as pd

from .labels import RankReference
from .model_zoo import make_train_test_indices


def prepare_training_table(
    raw: pd.DataFrame,
    feature_builder: Callable,
    label_builder: Callable,
    target_col: str | None,
    split_strategy: str,
    group_col: str,
    reference_path: Path | None = None,
) -> pd.DataFrame:
    raw = raw.reset_index(drop=True).copy()
    split_input = raw.copy()
    if target_col:
        if target_col not in raw:
            raise ValueError(f"Target column not found: {target_col}")
        targets = pd.to_numeric(raw[target_col], errors="coerce")
        if not targets.isin([0, 1, 2, 3]).all():
            raise ValueError("Target labels must be integers from 0 to 3 without missing values.")
        split_input["state_label"] = targets.astype(int)
    else:
        split_input["state_label"] = 0
    train_idx, test_idx, metadata = make_train_test_indices(
        split_input, "state_label", split_strategy=split_strategy,
        group_col=group_col if split_strategy == "group" else None,
    )
    inputs = raw.drop(columns=list({target_col, "state_label", "state_score", "risk_level", "risk_score"} & set(raw.columns)))
    train_raw = inputs.iloc[train_idx].copy()
    train = feature_builder(train_raw, reference_data=train_raw)
    test_source = inputs if split_strategy == "temporal" else inputs.iloc[test_idx]
    test = feature_builder(test_source, reference_data=train_raw).loc[test_idx]
    test = test.reindex(columns=train.columns)
    reference = None
    if target_col:
        for part in (train, test):
            part["state_label"] = targets.loc[part.index].astype(int)
            part["state_score"] = raw.loc[part.index, "state_score"] if "state_score" in raw else part["state_label"] * 25.0
            part["label_source"] = f"target:{target_col}"
    else:
        reference = RankReference()
        train = label_builder(train, reference=reference)
        frozen = RankReference(reference.distributions)
        test = label_builder(test, reference=frozen)
        if frozen.position != len(reference.distributions):
            raise ValueError("Rank reference does not match the scoring formula.")
    train["__split_partition"] = "train"
    test["__split_partition"] = "test"
    table = pd.concat([train, test]).sort_index().reset_index(drop=True)
    table.attrs["training_split"] = {
        **metadata, "test_size": 0.25, "random_state": 42,
        "group_col": group_col if split_strategy == "group" else None, "date_col": "date",
    }
    table.attrs["target_columns"] = [target_col] if target_col else []
    if reference_path is not None:
        joblib.dump({"rank_reference": reference, "feature_reference": train_raw,
                     "label_builder": label_builder.__name__ if not target_col else None}, reference_path)
    return table
