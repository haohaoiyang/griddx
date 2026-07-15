from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from griddx.features import add_enriched_device_features, add_enriched_station_features, modeling_columns
from griddx.economic_dispatch import solve_dispatch
from griddx.labels import make_device_enriched_weak_labels, make_station_enriched_weak_labels
from griddx.model_zoo import (
    DeviceAdaptiveMultiDiscriminator,
    build_discriminator_feature_groups,
    make_train_test_indices,
)
from griddx.station_fusion import StationHierarchicalFusionNetwork, build_station_feature_views


class EnrichedPipelineTest(unittest.TestCase):
    def test_risk_aware_dispatch_respects_budget_and_balance(self) -> None:
        table = pd.DataFrame(
            {
                "required_adjustment_mw": [10.0, 8.0, 6.0],
                "max_adjustable_mw": [10.0, 8.0, 3.0],
                "dispatch_cost_per_mw": [0.4, 0.5, 0.5],
                "risk_cost_per_mw": [0.2, 0.8, 1.4],
                "shortfall_cost_per_mw": [2.5, 2.0, 1.8],
                "marginal_net_value": [1.9, 0.7, -0.1],
                "state_label": [0, 1, 3],
            }
        )
        result = solve_dispatch(table, supply_budget_mw=12.0)
        self.assertLessEqual(result["recommended_allocation_mw"].sum(), 12.0 + 1e-6)
        balance = result["recommended_allocation_mw"] + result["unserved_adjustment_mw"]
        self.assertTrue((balance + 1e-6 >= result["required_adjustment_mw"]).all())
        self.assertTrue((result["total_objective_cost"] >= 0).all())
        self.assertTrue((result["solver_status"] == "optimal").all())

    def test_station_hierarchical_fusion_outputs_two_tasks(self) -> None:
        feature_names = [
            "voltage_mean",
            "history_defect_count",
            "maintenance_coverage_proxy",
            "main_transformer_count",
            "lightning_risk_level_code",
        ]
        views = build_station_feature_views(feature_names)
        self.assertEqual(set(views), {"operation", "history_maintenance", "infrastructure"})
        model = StationHierarchicalFusionNetwork(
            n_features=len(feature_names),
            n_classes=4,
            feature_views=views,
        )
        logits, risk, weights = model(torch.randn(5, len(feature_names)))
        self.assertEqual(tuple(logits.shape), (5, 4))
        self.assertEqual(tuple(risk.shape), (5,))
        self.assertEqual(tuple(weights.shape), (5, 3))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-6))

    def test_multi_discriminator_has_personalized_gate(self) -> None:
        feature_names = [
            "current_3phase",
            "history_defect_count",
            "defect_maintenance_status_code",
            "family_history_defect_count",
            "device_age_days",
        ]
        groups = build_discriminator_feature_groups(feature_names)
        self.assertEqual(set(groups), {"operation", "history", "maintenance", "family_profile"})
        self.assertTrue(all(indices for indices in groups.values()))
        model = DeviceAdaptiveMultiDiscriminator(
            n_features=len(feature_names),
            n_classes=4,
            feature_groups=groups,
        )
        logits, gates = model(torch.randn(6, len(feature_names)))
        self.assertEqual(tuple(logits.shape), (6, 4))
        self.assertEqual(tuple(gates.shape), (6, 4))
        self.assertTrue(torch.allclose(gates.sum(dim=1), torch.ones(6), atol=1e-6))

    def test_group_split_has_no_entity_overlap(self) -> None:
        df = pd.DataFrame(
            {
                "unified_device_id": np.repeat([f"D{i}" for i in range(20)], 4),
                "state_label": np.tile([0, 1, 2, 3], 20),
            }
        )
        train_idx, test_idx, metadata = make_train_test_indices(
            df,
            "state_label",
            split_strategy="group",
            group_col="unified_device_id",
        )
        train_devices = set(df.iloc[train_idx]["unified_device_id"])
        test_devices = set(df.iloc[test_idx]["unified_device_id"])
        self.assertFalse(train_devices & test_devices)
        self.assertEqual(metadata["split_strategy"], "group")

    def test_device_enriched_features_and_labels(self) -> None:
        rows = 40
        base = pd.DataFrame(
            {
                "suggested_model_group": ["line_or_load_like"] * rows,
                "history_defect_count": np.arange(rows) % 5,
                "history_trip_count": np.arange(rows) % 4,
                "history_defect_level_code": np.arange(rows) % 5,
                "history_trip_level_code": np.arange(rows) % 5,
                "defect_related_maintenance_count": np.arange(rows) % 3,
                "trip_related_maintenance_count": np.arange(rows) % 2,
                "defect_maintenance_status_code": np.arange(rows) % 3,
                "trip_maintenance_status_code": np.arange(rows) % 3,
                "family_history_defect_count": np.arange(rows) + 1,
                "family_history_trip_count": np.arange(rows) % 9,
                "family_device_count": np.full(rows, 10),
                "device_age_days": np.arange(rows) * 20 + 180,
                "measurement_coverage_ratio": np.linspace(0.5, 1.0, rows),
                "device_type_code": np.arange(rows) % 4,
                "manufacturer_code": np.arange(rows) % 5,
            }
        )
        featured = add_enriched_device_features(base)
        labeled = make_device_enriched_weak_labels(featured)
        self.assertIn("unresolved_defect_count", labeled)
        self.assertIn("family_event_count_per_device", labeled)
        self.assertTrue(labeled["state_label"].between(0, 3).all())
        numeric, categorical = modeling_columns(labeled, "state_label")
        self.assertIn("device_type_code", categorical)
        self.assertNotIn("device_type_code", numeric)

    def test_station_enriched_features_and_labels(self) -> None:
        rows = 40
        base = pd.DataFrame(
            {
                "history_defect_count": np.arange(rows) + 1,
                "history_trip_count": np.arange(rows) % 8,
                "history_maintenance_count": np.arange(rows) % 12,
                "main_transformer_count": np.arange(rows) % 4 + 1,
                "lightning_risk_level_code": np.arange(rows) % 4 + 1,
                "ice_area_level_code": np.arange(rows) % 4 + 1,
                "operation_coverage_rate": np.linspace(0.6, 1.0, rows),
                "device_count_with_operation": np.arange(rows) + 10,
                "voltage_spread": np.linspace(0, 4, rows),
                "current_peak_ratio": np.linspace(1, 2, rows),
            }
        )
        featured = add_enriched_station_features(base)
        labeled = make_station_enriched_weak_labels(featured)
        self.assertIn("unresolved_event_count_proxy", labeled)
        self.assertIn("maintenance_coverage_proxy", labeled)
        self.assertTrue(labeled["state_label"].between(0, 3).all())


if __name__ == "__main__":
    unittest.main()
