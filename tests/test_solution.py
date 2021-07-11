#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import unittest

from src.data import Node, Application, DisruptionBudget
from src.solution import Solution
from src.exceptions import *


class TestSolution(unittest.TestCase):

    def test_set_disruption_budget_value_invalid(self):
        """
        测试DisruptionBudget值为负数
        """
        solution = Solution()
        solution.set_nodes([Node("node1"), Node("node2")])
        solution.set_apps([Application("app1", "node1"), Application("app1", "node2")])
        self.assertRaises(DisruptionBudgetsValueInvalidException, solution.set_disruption_budget,
                          [DisruptionBudget("app1", -1)])

    def test_solve_node_not_found(self):
        """
        测试Node不存在
        """
        solution = Solution()
        solution.set_nodes([Node("node1"), Node("node2")])
        solution.set_apps([Application("app1", "node1"), Application("app1", "node3")])
        solution.set_disruption_budget([DisruptionBudget("app1", 1)])
        self.assertRaises(NodeNotFoundException, solution.solve)

    def test_solve_app_not_found(self):
        """
        测试App不存在
        """
        solution = Solution()
        solution.set_nodes([Node("node1"), Node("node2")])
        solution.set_apps([Application("app1", "node1"), Application("app1", "node2")])
        solution.set_disruption_budget([DisruptionBudget("app2", 1)])
        self.assertRaises(AppNotFoundException, solution.solve)

    def test_solve_disruption_budget_violation(self):
        """
        测试某Node不符合DisruptionBudget异常
        """
        solution = Solution()
        solution.set_nodes([Node("node1"), Node("node2")])
        solution.set_apps([Application("app1", "node1"), Application("app1", "node1")])
        solution.set_disruption_budget([DisruptionBudget("app1", 1)])
        self.assertRaises(DisruptionConstraintViolationException, solution.solve)

    def test_solve_disruption_budget_violation(self):
        """
        测试app缺少DisruptionBudget约束异常
        """
        solution = Solution()
        solution.set_nodes([Node("node1"), Node("node2")])
        solution.set_apps([Application("app1", "node1"), Application("app1", "node1")])
        self.assertRaises(DisruptionConstraintMissingException, solution.solve)

    def test_solve_node_only(self):
        """
        最优解（Node节点数较小）
        """
        solution = Solution()
        solution.set_nodes([Node("node1"), Node("node2"), Node("node3")])
        result = solution.solve()
        self.assertListEqual(result, [['node1', 'node2', 'node3']])

    def test_solve_small_nodes(self):
        """
        最优解（Node节点数较小）
        """
        solution = Solution()
        solution.set_nodes([Node("node1"), Node("node2"), Node("node3")])
        solution.set_apps([Application("app1", "node1"), Application("app1", "node2"),
                           Application("app2", "node1"), Application("app2", "node2"),
                           Application("app3", "node2"), Application("app3", "node3")])
        solution.set_disruption_budget([DisruptionBudget("app1", 1),
                                        DisruptionBudget("app2", 1),
                                        DisruptionBudget("app3", 1)])
        result = solution.solve()
        self.assertListEqual(result, [['node1', 'node3'], ['node2']])

    def test_solve_large_nodes(self):
        """
        最优解（Node节点数较多）
        """
        from sample_generator import SampleGenerator

        node_count = 5000
        app_count = 4000
        app_replica_range = (2, node_count * 50 // app_count)
        generator = SampleGenerator(node_count=node_count, app_count=app_count, app_replica_range=app_replica_range)

        nodes, apps, disruption_budgets = generator.generate_sample()

        solution = Solution()
        solution.set_nodes(nodes)
        solution.set_apps(apps)
        solution.set_disruption_budget(disruption_budgets)

        t0 = time.time()
        result = solution.solve()
        t1 = time.time()

        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)
        self.assertEqual(sum([1 for nodes in result for node in nodes]), node_count)
        self.assertLess(t1 - t0, 10*60, "execute time(s)")


if __name__ == "__main__":
    unittest.main()
