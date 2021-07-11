#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Tuple, List
import numpy as np

from src.data import Node, Application, DisruptionBudget


class SampleGenerator:
    def __init__(self, node_count=5000, app_count=4000, app_replica_range=(2, 200),
                 disruption_ratio=0.2,
                 force_disruption=True,
                 seed=0):
        self.node_count = node_count
        self.app_count = app_count
        self.app_replica_range = app_replica_range
        self.disruption_ratio = disruption_ratio
        self.force_disruption = force_disruption
        self.seed = seed

        self._nodes = [Node(f"node{i}") for i in range(1, 1+self.node_count)]
        self._app_names = [f"app{i}" for i in range(1, 1 + self.app_count)]

    def generate_sample(self) -> Tuple[List[Node], List[Application], List[DisruptionBudget]]:
        node_app_array = np.zeros(shape=(self.node_count, self.app_count), dtype=np.int16)
        apps = []

        for app_idx, app_name in enumerate(self._app_names):
            app_replica = np.random.randint(self.app_replica_range[0], self.app_replica_range[1])
            node_indices = np.random.randint(0, self.node_count, size=app_replica)

            for node_idx in node_indices:
                node_app_array[node_idx, app_idx] += 1
                apps.append(Application(app_name, self._nodes[node_idx].node_name))

        disruption_budgets = []
        app_disruption_budget = np.ceil(np.sum(node_app_array, axis=0) * self.disruption_ratio).astype(np.uint16)
        for app_idx, value in enumerate(app_disruption_budget):
            min_value = 1 if not self.force_disruption else np.max(node_app_array[:, app_idx])
            disruption_budgets.append(DisruptionBudget(self._app_names[app_idx], max(value, min_value)))

        return self._nodes, apps, disruption_budgets


if __name__ == "__main__":
    generator = SampleGenerator()
    generator.generate_sample()
