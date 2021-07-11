#!/usr/bin/env python
# -*- encoding: utf-8 -*-


class DisruptionBudget:
    def __init__(self, app_name: str = None, disruption_allowed: int = 0):
        self.app_name = app_name
        self.disruption_allowed = disruption_allowed

    def __str__(self):
        return f"Disruption Budget: {self.app_name}: {self.disruption_allowed}"

    def __repr__(self):
        return self.__str__()


class Application:
    def __init__(self, app_name: str = None, node_name: str = None):
        self.app_name = app_name
        self.node_name = node_name

    def __str__(self):
        return f"App: {self.app_name} -> {self.node_name}"

    def __repr__(self):
        return self.__str__()


class Node:
    def __init__(self, node_name: str = None):
        self.node_name = node_name

    def __str__(self):
        return f"Node: {self.node_name}"

    def __repr__(self):
        return self.__str__()
