import unittest

import numpy as np
import pandas as pd
import torch

from griddx.model_zoo import DeviceAdaptiveMultiDiscriminator
from griddx.paper_experiments import (
    DeviceAblationNetwork, ExperimentSpec, StationAblationNetwork, grouped_partitions,
)
from griddx.station_fusion import StationHierarchicalFusionNetwork


class PaperExperimentsTest(unittest.TestCase):
    def test_full_variants_match_production_forward(self):
        groups = {"a": [0, 1], "b": [2, 3]}
        x = torch.randn(7, 4)
        for base, experiment in [
            (DeviceAdaptiveMultiDiscriminator(4, 4, groups), DeviceAblationNetwork(4, groups, ExperimentSpec("full"))),
            (StationHierarchicalFusionNetwork(4, 4, groups), StationAblationNetwork(4, groups, ExperimentSpec("full"))),
        ]:
            experiment.load_state_dict(base.state_dict())
            base.eval()
            experiment.eval()
            for expected, actual in zip(base(x), experiment(x)):
                torch.testing.assert_close(expected, actual)

    def test_branch_is_removed_and_uniform_gate_is_constant(self):
        groups = {"a": [0], "b": [1], "c": [2]}
        model = DeviceAblationNetwork(3, groups, ExperimentSpec("drop", removed="b", uniform=True, residual=False))
        self.assertNotIn("b", model.discriminators)
        self.assertIsNone(model.gate)
        self.assertIsNone(model.global_residual)
        _, weights = model(torch.randn(8, 3))
        torch.testing.assert_close(weights, torch.full((8, 2), 0.5))

    def test_partitions_disjoint_and_exhaustive(self):
        frame = pd.DataFrame({"id": np.repeat(np.arange(100), 4), "state_label": np.tile(np.arange(4), 100)})
        parts = grouped_partitions(frame, "id")
        self.assertEqual(set(np.concatenate(list(parts.values()))), set(range(len(frame))))
        sets = {k: set(frame.iloc[v]["id"]) for k, v in parts.items()}
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])

    def test_station_risk_head_removal_still_trains_classification(self):
        model = StationAblationNetwork(3, {"a": [0], "b": [1], "c": [2]}, ExperimentSpec("no_risk", risk_task=False))
        logits, risk, _ = model(torch.randn(8, 3))
        self.assertIsNone(model.risk_head)
        torch.testing.assert_close(risk, torch.zeros(8))
        logits.sum().backward()
        self.assertIsNotNone(model.state_head[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
