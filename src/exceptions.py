#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import List


class NodeNotFoundException(Exception):
    """
    没有Node所引起的异常
    """
    def __init__(self, node_name: str):
        self.node_name = node_name

    def __str__(self):
        return f"Node ({self.node_name}) can NOT be found!"

    def __repr__(self):
        return self.__str__()


class AppNotFoundException(Exception):
    """
    没有Application所引起的异常
    """
    def __init__(self, app_name: str):
        self.app_name = app_name

    def __str__(self):
        return f"Application ({self.app_name}) can NOT be found!"

    def __repr__(self):
        return self.__str__()


class DisruptionConstraintMissingException(Exception):
    """
    缺少某应用的Disruption Budget条件
    """
    def __init__(self, app_names: List[str]):
        self.app_names = app_names

    def __str__(self):
        return f"Disruption Budget constraint of applications {self.app_names} can NOT be found."

    def __repr__(self):
        return self.__str__()


class DisruptionConstraintViolationException(Exception):
    """
    当某个节点上面部署的应用不满足Disruption Budget条件，触发该异常
    """
    def __init__(self, ex_msg: str):
        self.ex_msg = ex_msg

    def __str__(self):
        return self.ex_msg

    def __repr__(self):
        return self.__str__()


class DisruptionBudgetsValueInvalidException(Exception):
    """
    Disruption Budget数值无效(<0)错误
    """
    def __init__(self, app_name: str, value: int):
        self.app_name = app_name
        self.value = value

    def __str__(self):
        return f"Disruption Budget of application ({self.app_name} is {self.value}, which should be >= 0"

    def __repr__(self):
        return self.__str__()


class DisruptionBudgetsMissingException(Exception):
    """
    某个Application的Disruption Budget缺失异常
    """
    def __init__(self, app_name: str):
        self.app_name = app_name

    def __str__(self):
        return f"Missing disruption budget of application ({self.app_name})"

    def __repr__(self):
        return self.__str__()

