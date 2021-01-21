from synthesis.synthesizer.dplyr_to_pd.code_analysis.visitor import DplyrTransformer, DependencyFinder, RWriter
from graphviz import Digraph
from collections import OrderedDict


class Graph:

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.r_writer = RWriter()

    def edge(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)

        self.nodes[node2] = False # non root node
        self.edges[node1] += [node2]

    def add_node(self, node1):
        if node1 not in self.nodes:
            self.nodes[node1] = True
            self.edges[node1] = []

    def linearize(self):
        root_node = next(filter(lambda x: self.nodes[x], self.nodes))
        return list(OrderedDict.fromkeys(self.linearize_aux(root_node)))

    def linearize_aux(self, node):
        res = []
        children = self.edges[node]
        if children:
            for child in children:
                res += self.linearize_aux(child)
            code = node.accept(self.r_writer)
            res += [code]
        return res

    def to_graphviz(self):
        dot = Digraph(comment='Dependencies')
        writer = RWriter()
        for node1 in self.nodes:
            for node2 in self.edges[node1]:
                dot.edge(node1.accept(writer), node2.accept(writer))
        dot.view("viz.png")

    def dfs(self):
        root_node = next(filter(lambda x: self.nodes[x], self.nodes))
        return list(OrderedDict.fromkeys(self.dfs_aux(root_node)))

    def dfs_aux(self, node):
        result = []
        for child in self.edges[node]:
            result += self.dfs_aux(child)
        if self.edges[node]:
            result += [node.accept(self.r_writer)]
        return result
