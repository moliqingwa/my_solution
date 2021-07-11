#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
from src.solution import Solution

from tests.sample_generator import SampleGenerator

if __name__ == "__main__":
    node_count = 5000
    app_count = 4000
    app_replica_range = (2, node_count * 50 // app_count)
    generator = SampleGenerator(node_count=node_count, app_count=app_count, app_replica_range=app_replica_range)

    solution = Solution()
    # solution.set_nodes([Node("node1"),
    #                     Node("node2"),
    #                     Node("node3"),
    #                     Node("node4")])
    # solution.set_apps([Application("app1", "node1"),
    #                    Application("app1", "node2"),
    #                    Application("app2", "node1"),
    #                    Application("app2", "node2"),
    #                    Application("app3", "node2"),
    #                    Application("app3", "node3")])
    # solution.set_disruption_budget([DisruptionBudget("app1", 1),
    #                                 DisruptionBudget("app2", 1),
    #                                 DisruptionBudget("app3", 1)])

    nodes, apps, disruption_budgets = generator.generate_sample()
    solution.set_nodes(nodes)
    solution.set_apps(apps)
    solution.set_disruption_budget(disruption_budgets)

    t0 = time.time()
    solution.solve()
    print(f"solved used: {time.time() - t0: .3f}s")
