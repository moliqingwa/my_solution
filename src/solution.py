#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
from typing import Tuple, List
from collections import defaultdict
from enum import Enum

import numpy as np
from ortools.linear_solver import pywraplp
from loguru import logger

from src.data import Node, Application, DisruptionBudget

from src.exceptions import *


class PreciseAlgorithm(Enum):
    OPTIMIZE = 1
    DYNAMIC_PLANNING = 2
    BACKTRACE = 3


class Solution:
    def __init__(self):
        super().__init__()
        self.node_2_index_dict = dict()
        self.index_2_node_dict = dict()

        self.apps = []
        self.app_2_index_dict = dict()
        self.index_2_app_dict = dict()

        self.disruption_budgets = defaultdict()

        self._node_app_array = None
        self._app_disruption_array = None
        self._sorted_node_indices = None  # 按照Node中App的数量排序的Node索引

    def set_nodes(self, nodes: List[Node]):
        """
        设置节点

        :param nodes: 节点列表
        """
        self.node_2_index_dict = dict(zip(map(lambda n: n.node_name, nodes),
                                          range(len(nodes))))
        assert len(self.node_2_index_dict) == len(nodes)

        self.index_2_node_dict = {v: k for k, v in self.node_2_index_dict.items()}
        assert len(self.index_2_node_dict) == len(self.node_2_index_dict)

    def set_apps(self, apps: List[Application]):
        """
        设置Application列表

        :param apps: Application列表
        """
        self.apps = apps

        app_sets = {a.app_name for a in self.apps}
        self.app_2_index_dict = dict(zip(list(app_sets), range(len(app_sets))))
        self.index_2_app_dict = {v: k for k, v in self.app_2_index_dict.items()}
        assert len(self.index_2_app_dict) == len(self.app_2_index_dict)

    def set_disruption_budget(self, disruption_budgets: List[DisruptionBudget]):
        """
        设置中断约束.

        :param disruption_budgets: 中断约束

        异常类型：DisruptionBudgetsValueInvalidException
        """
        self.disruption_budgets.clear()

        disruption_budgets = [] if not disruption_budgets else disruption_budgets
        for budget in disruption_budgets:
            if budget.disruption_allowed < 0:
                raise DisruptionBudgetsValueInvalidException(budget.app_name, budget.disruption_allowed)
            self.disruption_budgets[budget.app_name] = budget.disruption_allowed

    def solve(self,
              greedy_node_thresh: int = 100,
              precise_algorithm: PreciseAlgorithm = PreciseAlgorithm.OPTIMIZE,
              debug: bool = True) -> List[List[Node]]:
        """
        利用贪心算法、回溯法、动态规划或约束优化算法，求解.

        :param greedy_node_thresh: 指定贪心算法阈值. 若求解的node数目超过该数值,则使用贪心,反之，使用精确求解(时间复杂度高)
        :param precise_algorithm: 指定精确求解算法. 分别为约束优化算法, 回溯法和动态规划算法,默认为约束优化算法.
        :param debug: 是否显示调试细节
        :return: 返回求解的节点列表

        异常类型：NodeNotFoundException
            AppNotFoundException
            DisruptionConstraintViolationException
        """
        app_count = len(self.app_2_index_dict)
        node_count = len(self.node_2_index_dict)

        # 构建 Node x App 的约束矩阵
        self._node_app_array = np.zeros(shape=(node_count, app_count), dtype=np.int16)
        for app in self.apps:
            app_index = self.app_2_index_dict[app.app_name]
            node_index = self.node_2_index_dict.get(app.node_name, None)
            if node_index is None:
                raise NodeNotFoundException(app.node_name)
            self._node_app_array[node_index, app_index] += 1

        # 构建排序的Node索引（按Node中App的数量升序）
        self._sorted_node_indices = np.argsort(np.sum(self._node_app_array, axis=1))  # shape: node_count

        # 构建App的中断约束向量
        self._app_disruption_array = np.zeros(shape=app_count, dtype=np.int16)
        for app_name, disruption in self.disruption_budgets.items():
            app_index = self.app_2_index_dict.get(app_name, None)
            if app_index is None:
                raise AppNotFoundException(app_name)
            self._app_disruption_array[app_index] = disruption

        # 检查Application Disruption约束是否缺失
        if np.any(self._app_disruption_array <= 0):
            node_names = [self.index_2_node_dict[i] for i in np.where(self._app_disruption_array <= 0)[0]]
            raise DisruptionConstraintMissingException(node_names)

        # 检查是否有Node当前不满足约束条件(单台节点上的App数目已经不符合约束条件)
        disruption_matrix = self._app_disruption_array - self._node_app_array  # n_N x n_A
        if np.any(disruption_matrix < 0):
            node_app_pairs = ",".join(map(lambda na:
                                          f"{self.index_2_node_dict[na[0]]} - {self.index_2_app_dict[na[1]]}",
                                          zip(*np.where(disruption_matrix < 0))))
            raise DisruptionConstraintViolationException(f"Node App pair(s) [{node_app_pairs}] "
                                                         f"violates disruption constraint.")

        step, return_value = 0, []
        node_mask = np.ones(node_count, dtype=np.bool_)  # True表示当前节点可重启; False表示已经重启过,不需要操作
        while np.any(node_mask):
            t0 = time.time()

            current_node_count = np.sum(node_mask)
            if current_node_count >= greedy_node_thresh:
                node_indices = self._greedy_solve([i for i in self._sorted_node_indices if node_mask[i]],
                                                  self._app_disruption_array)
            elif precise_algorithm == PreciseAlgorithm.BACKTRACE:
                # 回溯法求解
                visited_nodes = set()  # 表示当前DFS所遍历到的节点，节点列表允许本次重启
                cum_node_app_array = np.zeros(app_count, dtype=np.int32)  # visited_nodes访问的节点的累积求和向量
                node_indices = self._backtrace_solve(node_mask, visited_nodes, cum_node_app_array)
            elif precise_algorithm == PreciseAlgorithm.DYNAMIC_PLANNING:
                # 动态规划求解
                memo = dict()  # cache
                node_indices = self._dp_solve(node_mask, node_count-1, self._app_disruption_array, memo)
            else:
                # 优化算法求解
                node_indices = self._op_solve(node_mask)

                if not node_indices:  # 若无解，可使用贪心算法兜底
                    node_indices = self._greedy_solve([i for i in self._sorted_node_indices if node_mask[i]],
                                                      self._app_disruption_array)

            node_names = [self.index_2_node_dict[node_idx] for node_idx in node_indices]
            logger.info(f"Step: {step}, nodes: {len(node_indices)}, consume: {time.time() - t0: .3f}s")
            logger.info(node_names)

            step += 1
            node_mask[node_indices] = np.False_
            return_value.append(node_names)
        return return_value

    def _greedy_solve(self,
                      candidate_node_indices: List[int],
                      remain_disruption: np.ndarray) -> List[int]:
        """
        贪心法，每次从可用节点中获取一个符合最小约束的节点返回。
        时间复杂度：O(n_N)，其中n_N为Node数量

        :param candidate_node_indices: Node节点索引列表，作为本次求解的候选集（已排序）
        :param mask: 表示当前节点是否可选择, True表示当前索引的节点可选择
        :param remain_disruption: 当前中断约束向量
        :return: 选择的节点的索引列表（近似最优）
        """
        if not candidate_node_indices or np.any(remain_disruption < 0):
            return []

        # 贪心获取占用最小约束（remain_distruption值最大）的节点索引
        best_node_i, best_node_idx, best_remain_disruption = 0, None, None
        for i, node_idx in enumerate(candidate_node_indices):
            if np.any(remain_disruption < self._node_app_array[node_idx]):  # 不符合约束的节点
                continue

            best_node_i = i
            best_node_idx = node_idx  # 因node索引已经排序过，所以第一个即为当前最符合的
            best_remain_disruption = remain_disruption - self._node_app_array[node_idx]
            break

        if best_node_idx is None:  # 已经没有可选择的节点，返回
            return []

        return_value = self._greedy_solve(candidate_node_indices[best_node_i + 1:], best_remain_disruption)

        return_value.append(best_node_idx)
        return return_value

    def _backtrace_solve(self, mask: np.ndarray, visited_nodes: set, cum_node_app_array: np.ndarray) -> List[int]:
        """
        回溯法（全组合）求解，深度遍历多叉树，返回最优组合。
        时间复杂度：约为O(2^n_N)，其中n_N为Node数量

        :param mask: 表示当前节点是否可选择, True表示当前索引的节点可选择
        :param visited_nodes: 当前访问的节点列表
        :param cum_node_app_array: 已访问节点列表的中断向量求和
        :return: 最长的节点的索引列表
        """
        if np.any(self._app_disruption_array < cum_node_app_array):  # 不满足约束,直接返回
            return []

        available_node_indices = np.where(mask)[0]

        # 遍历后继节点，返回最长的有效子序列
        max_successor_nodes = []
        for idx in available_node_indices:
            idx = idx.item()
            visited_nodes.add(idx)
            cum_node_app_array += self._node_app_array[idx]

            mask_ = mask.copy()
            mask_[: idx + 1] = np.False_
            successor_nodes = self._backtrace_solve(mask_, visited_nodes, cum_node_app_array)
            if len(successor_nodes) > len(max_successor_nodes):
                max_successor_nodes = successor_nodes
            cum_node_app_array -= self._node_app_array[idx]
            visited_nodes.remove(idx)

        # 筛选出最长的子序列，并返回
        if len(visited_nodes) > len(max_successor_nodes):
            return list(visited_nodes)
        else:
            return max_successor_nodes

    def _dp_solve(self, mask: np.ndarray, i: int, remain_disruption: np.ndarray, memo: dict) -> List[int]:
        """
        动态规划求解(0-1背包），返回最优组合。
        时间复杂度：O(n_N*n_A),其中n_N为Node数量，n_A为应用数量

        :param mask: 表示当前节点是否可选择, True表示当前索引的节点可选择
        :param i: 表示第i个Node节点
        :param remain_disruption: 当前的约束中断向量
        :param memo: 中间过程缓存cache
        :return: 最长的节点的索引列表
        """
        if i < 0 or np.any(remain_disruption < 0):
            return []

        memo_key = (i, remain_disruption.tobytes())
        if memo.get(memo_key, None) is not None:
            return memo.get(memo_key)

        max_nodes = self._dp_solve(mask, i - 1, remain_disruption, memo)  # 不选择第i个Node
        if mask[i] and not np.any(remain_disruption < self._node_app_array[i]):  # 允许选择第i个Node
            max_i_nodes = self._dp_solve(mask, i - 1, remain_disruption - self._node_app_array[i], memo)
            if len(max_i_nodes) + 1 > len(max_nodes):
                max_nodes = max_i_nodes
                max_nodes.append(i)

        memo[memo_key] = max_nodes
        return max_nodes

    def _op_solve(self, mask: np.ndarray) -> List[int]:
        """
        最优化方法求解，返回最优组合。

        :param mask: 表示当前节点是否可选择, True表示当前索引的节点可选择
        :return: 最长的节点的索引列表。若无解，则返回空
        """
        available_node_indices = np.where(mask)[0]  # 获取可重启的Node索引，作为本次求解的输入

        # 整数规划求解器
        solver = pywraplp.Solver('SolveIntegerProblem',
                                 pywraplp.Solver.SAT_INTEGER_PROGRAMMING)

        # 定义变量,各个Node节点的权重 weight,并指定其定义域[0, 1]
        # Node节点是否选择, 0: 不重启, 1: 重启
        weights = [solver.IntVar(0, 1, f"w_{node_idx.item()}") for node_idx in available_node_indices]

        # 添加约束条件
        # Constraint: weights[i] * node_app_vector <= b
        constraints = []
        for j in range(len(self.app_2_index_dict)):  # the j-th App
            if np.any(self._node_app_array[available_node_indices, j] > 0):  # 待重启的节点中有该App，则添加约束
                c = solver.Constraint(0, self._app_disruption_array[j].item())

                app_vector = self._node_app_array[available_node_indices, j]  # shape: (n_N, )
                for i, v in enumerate(app_vector):  # the i-th Node
                    if v > 0:
                        c.SetCoefficient(weights[i], v.item())
                constraints.append(c)

        logger.debug(f'待求解变量的数量 = {solver.NumVariables()}，约束方程的个数 = {solver.NumConstraints()}')

        # 定义目标函数，指定目标函数的变量的系数,求最大值
        # Objective function: max(sum(weights)).
        objective = solver.Objective()
        for w in weights:
            objective.SetCoefficient(w, 1)
        objective.SetMaximization()

        # 求解并打印结果
        result_status = solver.Solve()
        if result_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return []

        best_weights = np.array([int(w.solution_value()) for w in weights])
        # 更新可重启节点
        current_node_indices = available_node_indices[np.where(best_weights > 0)]

        # 打印信息
        # logger.debug(f'最优解(重启的Node数量) = {solver.Objective().Value()}')
        # logger.debug(f'重启节点索引: {current_node_indices.tolist()}')

        return current_node_indices.tolist()

