from lark import Tree
import os
from abc import abstractmethod


class Node(Tree):

    def __init__(self, data, children, meta=None):
        super().__init__(data, children, meta=meta)

    @abstractmethod
    def accept(self, visitor):
        raise NotImplementedError


class IdentifierNode(Node):

    def __init__(self, token, meta=None):
        super().__init__('Identifier Node', [str(token)], meta)

    @property
    def name(self):
        return self.children[0]

    def accept(self, visitor):
        return visitor.visit_identifier_node(self)


class SequenceNode(Node):

    def __init__(self, nodes, meta=None):
        super().__init__('Sequence Node', nodes, meta)

    @property
    def arguments(self):
        return self.children

    def replace_arg(self, i: int, node: Node):
        self.children[i] = node

    def accept(self, visitor):
        return visitor.visit_sequence_node(self)


class BlockNode(Node):

    def __init__(self, nodes, meta=None):
        super().__init__('Block Node', nodes, meta)

    @property
    def lines(self):
        return self.children

    def accept(self, visitor):
        return visitor.visit_block_node(self)


class FunctionNode(Node):

    def __init__(self, name, args, meta=None):
        super().__init__('Function Node', [name, args], meta)

    @property
    def name(self):
        return self.children[0]

    @property
    def arguments(self):
        return self.children[1]

    def accept(self, visitor):
        return visitor.visit_function_node(self)


class PredicateNode(Node):

    def __init__(self, predicate, meta=None):
        super().__init__('Predicate Node', [predicate], meta)

    @property
    def predicate(self):
        return self.children[0]

    def accept(self, visitor):
        return visitor.visit_predicate_node(self)


class AssignmentNode(Node):

    def __init__(self, lvalue: IdentifierNode, expr: Node, meta=None):
        super().__init__('Assignment Node', [lvalue, expr], meta)

    @property
    def left_value(self):
        return self.children[0]

    @property
    def right_value(self):
        return self.children[1]

    def accept(self, visitor):
        return visitor.visit_assignment_node(self)


class RValueNode(Node):

    def __init__(self, lvalue: IdentifierNode, meta=None):
        super().__init__('RValue Node', [lvalue], meta)

    @property
    def value(self):
        return self.children[0]

    def accept(self, visitor):
        return visitor.visit_right_value_node(self)


class LiteralNode(Node):

    def __init__(self, literal, meta=None):
        super().__init__('LiteralNode', [literal], meta)

    @property
    def value(self):
        return self.children[0]

    def accept(self, visitor):
        return visitor.visit_literal_node(self)


class EmptyNode(Node):

    def __init__(self, meta=None):
        super().__init__('Empty Node', [], meta)

    def accept(self, visitor):
        return visitor.visit_empty_node(self)

