from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from griddx.external_validation import (
    PROXY_LABEL_REQUIRED_COLUMNS,
    build_proxy_reference_labels,
    classification_metrics,
    population_stability_index,
)


class ExternalValidationTest(unittest.TestCase):
    def test_proxy_reference_excludes_incomplete_profile_rows(self) -> None:
        rows = 8
        table = pd.DataFrame(
            {
                "suggested_model_group": ["line_or_load_like"] * rows,
                "history_defect_count": np.arange(rows) % 3,
                "history_trip_count": np.arange(rows) % 2,
                "history_defect_level_code": np.arange(rows) % 4,
                "history_trip_level_code": np.arange(rows) % 3,
                "defect_maintenance_status_code": np.arange(rows) % 3,
                "trip_maintenance_status_code": np.arange(rows) % 3,
                "family_history_defect_count": np.arange(rows) + 1,
                "family_history_trip_count": np.arange(rows) % 4,
                "defect_related_maintenance_count": np.arange(rows) % 2,
                "trip_related_maintenance_count": np.arange(rows) % 2,
                "family_device_count": np.full(rows, 10),
                "device_age_days": np.arange(rows) + 100,
            }
        )
        self.assertTrue(set(PROXY_LABEL_REQUIRED_COLUMNS).issubset(table.columns))
        table.loc[0, "history_defect_count"] = np.nan
        labels, eligible, metadata = build_proxy_reference_labels(table)
        self.assertFalse(eligible.iloc[0])
        self.assertTrue(labels.iloc[0] != labels.iloc[0])
        self.assertEqual(int(eligible.sum()), rows - 1)
        self.assertEqual(metadata["label_source"], "proxy_enriched_weak_rule")

    def test_population_stability_index_marks_large_shift(self) -> None:
        reference = pd.Series(np.linspace(0.0, 1.0, 100))
        shifted = pd.Series(np.linspace(10.0, 11.0, 100))
        self.assertGreater(population_stability_index(reference, shifted), 0.25)

    def test_classification_metrics_include_risk_metrics(self) -> None:
        metrics = classification_metrics(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 0]))
        self.assertIn("high_risk_precision_label_ge_2", metrics)
        self.assertAlmostEqual(metrics["high_risk_recall_label_ge_2"], 0.5)
        self.assertAlmostEqual(metrics["majority_class_accuracy"], 0.25)


if __name__ == "__main__":
    unittest.main()
