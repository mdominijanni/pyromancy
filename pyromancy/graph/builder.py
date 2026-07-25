from .components import NodeSrc, OpSrc, InputSrc, NodeCfg, OpInput
from ..nodes.base import Node

import itertools
from typing import Self

import networkx as nx
import torch.nn as nn


class GraphBuilder:

    topology: nx.DiGraph  # node: (is_node: bool, assigned: bool)
    nodes: dict[str, Node]
    node_configs: dict[str, NodeCfg]
    ops: dict[str, nn.Module]
    op_inputs: dict[str, OpInput]

    def __init__(self) -> None:
        self.topology = nx.DiGraph()
        self.nodes = {}
        self.node_configs = {}
        self.ops = {}
        self.op_inputs = {}

    def _add_node(
        self,
        name: str,
        node: Node,
        input: OpSrc | None,
        cfg: NodeCfg | None = None,
    ) -> Self:
        # ensure name is valid
        if not isinstance(name, str):
            raise TypeError("`name` must be a `str`")
        if "." in name:
            raise ValueError("`name` cannot contain character '.'")
        if len(name) == 0:
            raise ValueError("`name` cannot be an empty string")

        # ensure name is free
        vtx = self.topology.nodes.get(name, None)
        if vtx is not None:
            if vtx["assigned"]:
                if vtx["is_node"]:
                    raise RuntimeError("`name` is already assigned to a node")
                else:
                    raise RuntimeError("`name` is already assigned to an op")
            elif not vtx["is_node"]:
                raise RuntimeError("`name` must be assigned to an op")

        # ensure input is not another node
        if input is not None:
            if not isinstance(input, OpSrc):
                raise TypeError("`input` must be a `OpRef`")
            if input.name == name:
                raise RuntimeError("`input` cannot specify a node")
            vtx = self.topology.nodes.get(input.name, None)
            if vtx is not None and vtx["is_node"]:
                raise RuntimeError("`input` cannot specify a node")

        # assignments
        self.nodes[name] = node
        self.node_configs[name] = cfg if cfg is not None else NodeCfg()

        self.topology.add_node(name, is_node=True, assigned=True)
        if input is not None:
            if input.name not in self.topology.nodes:
                self.topology.add_node(input.name, is_node=False, assigned=False)
            self.topology.add_edge(input.name, name)

        return self

    def add_node(
        self,
        name: str,
        node: Node,
        input: OpSrc | None,
        cfg: NodeCfg | None = None,
    ) -> Self:
        if not isinstance(name, str):
            raise TypeError("`name` must be a `str`")
        elif "/" in name:
            raise ValueError("`name` cannot contain character '/'")
        else:
            return self._add_node(name, node, input, cfg)

    def _add_op(
        self,
        name: str,
        op: nn.Module,
        input: OpInput | None,
    ) -> Self:
        # ensure name is valid
        if not isinstance(name, str):
            raise TypeError("`name` must be a `str`")
        if "." in name:
            raise ValueError("`name` cannot contain character '.'")

        # ensure name is free
        vtx = self.topology.nodes.get(name, None)
        if vtx is not None:
            if vtx["assigned"]:
                if vtx["is_node"]:
                    raise RuntimeError("`name` is already assigned to a node")
                else:
                    raise RuntimeError("`name` is already assigned to an op")
            elif vtx["is_node"]:
                raise RuntimeError("`name` must be assigned to a node")

        # ensure inputs correspond to correct vertex types
        if input is not None:
            for arg in itertools.chain(input.args, input.kwargs.values()):
                if isinstance(arg, NodeSrc):
                    if arg.name == name:
                        raise RuntimeError(f"input `{arg}` references an op")
                    vtx = self.topology.nodes.get(arg.name, None)
                    if vtx is not None and not vtx["is_node"]:
                        raise RuntimeError(f"input `{arg}` references an op")
                elif isinstance(arg, OpSrc):
                    vtx = self.topology.nodes.get(arg.name, None)
                    if vtx is not None and vtx["is_node"]:
                        raise RuntimeError(f"input `{arg}` references a node")

        # assignments
        self.ops[name] = op
        self.op_inputs[name] = input if input is not None else OpInput()

        self.topology.add_node(name, is_node=False, assigned=True)
        if input is not None:
            for arg in itertools.chain(input.args, input.kwargs.values()):
                if isinstance(arg, NodeSrc | OpSrc):
                    if arg.name not in self.topology.nodes:
                        self.topology.add_node(
                            arg.name, is_node=isinstance(arg, NodeSrc), assigned=False
                        )
                    self.topology.add_edge(arg.name, name)

        return self

    def add_op(
        self,
        name: str,
        op: nn.Module,
        input: OpInput,
    ) -> Self:
        if not isinstance(name, str):
            raise TypeError("`name` must be a `str`")
        elif "/" in name:
            raise ValueError("`name` cannot contain character '/'")
        else:
            return self._add_op(name, op, input)
