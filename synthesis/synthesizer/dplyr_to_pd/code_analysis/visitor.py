from lark import Transformer, v_args, Token
from synthesis.synthesizer.dplyr_to_pd.code_analysis.nodes import *
from abc import ABC, abstractmethod


class DplyrTransformer(Transformer):
    """ Lark built in visitor for grammar construction rules """

    @v_args(inline=True)
    def identifier_node(self, arg):
        return IdentifierNode(arg)

    @v_args(inline=True)
    def single_block_node(self, arg):
        nodes = [arg]
        return BlockNode(nodes)

    @v_args(inline=True)
    def block_node(self, arg, other_bock: BlockNode):
        nodes = [arg] + other_bock.lines
        return BlockNode(nodes)

    @v_args(inline=True)
    def single_sequence_node(self, arg):
        nodes = [arg]
        return SequenceNode(nodes)

    @v_args(inline=True)
    def sequence_node(self, arg, other_seq: SequenceNode):
        nodes = [arg] + other_seq.arguments
        return SequenceNode(nodes)

    @v_args(inline=True)
    def function_node(self, name: Token, args: SequenceNode):
        return FunctionNode(str(name), args)

    @v_args(inline=True)
    def collapse_function_node(self, arg: Tree, fn: FunctionNode):
        return FunctionNode(str(fn.name), self.sequence_node(arg, fn.arguments))

    @v_args(inline=True)
    def predicate_node(self, arg: Token, op: Token, expr: Node, lc: Token, rest: Node):
        visitor = RWriter()
        args = [str(arg), str(op), expr.accept(visitor), str(lc), rest.accept(visitor)]
        return PredicateNode(' '.join(args))

    @v_args(inline=True)
    def single_predicate_node(self, arg: Token, op: Token, expr: Node):
        visitor = RWriter()
        args = [str(arg), str(op), expr.accept(visitor)]
        return PredicateNode(' '.join(args))

    @v_args(inline=True)
    def empty_node(self):
        return EmptyNode()

    @v_args(inline=True)
    def assignment_node(self, lvalue: IdentifierNode, expr: Node):
        return AssignmentNode(lvalue, expr)

    @v_args(inline=True)
    def rvalue_node(self, lvalue: IdentifierNode):
        return RValueNode(lvalue)

    @v_args(inline=True)
    def literal_node(self, lit: Token):
        return LiteralNode(lit)

    @v_args(inline=True)
    def collapse(self, arg):
        return arg


class Visitor(ABC):
    """Generic visitor used to the traverse the AST"""

    @abstractmethod
    def visit_block_node(self, sq: BlockNode):
        raise NotImplementedError

    @abstractmethod
    def visit_function_node(self, fn: FunctionNode):
        raise NotImplementedError

    @abstractmethod
    def visit_identifier_node(self, ide: IdentifierNode):
        raise NotImplementedError

    @abstractmethod
    def visit_sequence_node(self, sq: SequenceNode):
        raise NotImplementedError

    @abstractmethod
    def visit_predicate_node(self, pr: PredicateNode):
        raise NotImplementedError

    @abstractmethod
    def visit_empty_node(self, pr: EmptyNode):
        raise NotImplementedError

    @abstractmethod
    def visit_assignment_node(self, an: AssignmentNode):
        raise NotImplementedError

    @abstractmethod
    def visit_right_value_node(self, rv: RValueNode):
        raise NotImplementedError

    @abstractmethod
    def visit_literal_node(self, rv: LiteralNode):
        raise NotImplementedError


class RWriter(Visitor):
    """Visitor used to write R"""

    def visit_block_node(self, sq: BlockNode):
        args = []
        for arg in sq.lines:
            args += [arg.accept(self)]
        return '\n'.join(args)

    def visit_function_node(self, fn: FunctionNode):
        return f'{fn.name}({fn.arguments.accept(self)})'

    def visit_identifier_node(self, ide: IdentifierNode):
        return ide.name

    def visit_sequence_node(self, sq: SequenceNode):
        args = []
        for arg in sq.arguments:
            args += [arg.accept(self)]
        return ', '.join(args)

    def visit_predicate_node(self, pr: PredicateNode):
        return pr.predicate

    def visit_empty_node(self, pr: EmptyNode):
        return ''

    def visit_assignment_node(self, an: AssignmentNode):
        return f'{an.left_value.accept(self)} <- {an.right_value.accept(self)}'

    def visit_right_value_node(self, rv: RValueNode):
        return rv.value.accept(self)

    def visit_literal_node(self, lit: LiteralNode):
        return lit.value


class DependencyFinder(Visitor):
    """ For each line find its depedencies on inputs"""""

    def __init__(self, n_inputs: int):
        self.count = 0
        self.left_values = {IdentifierNode(f'input{i+1}'): IdentifierNode(f'input{i+1}') for i in range(n_inputs)}
        self.fn_dependencies = {}
        self.new_assignments = {}

    def visit_block_node(self, sq: BlockNode):
        for line in sq.lines:
            line.accept(self)
        return self.fn_dependencies

    def visit_function_node(self, fn: FunctionNode):
        result = fn.arguments.accept(self)
        return result

    def visit_identifier_node(self, ide: IdentifierNode):
        dep = next(filter(lambda x: x == ide, self.left_values), None)
        if dep is not None:
            return [self.left_values[dep]]
        return []

    def visit_sequence_node(self, sq: SequenceNode):
        dependencies = []
        for i in range(len(sq.arguments)):
            if isinstance(sq.arguments[i], FunctionNode) and sq.arguments[i] not in self.new_assignments:
                if sq.arguments[i].accept(self):
                    new_id = IdentifierNode(f'tmp_{self.count}')
                    an = AssignmentNode(new_id, sq.arguments[i])
                    self.count += 1
                    self.left_values[an.left_value] = an
                    self.new_assignments[sq.children[i]] = new_id
                    self.fn_dependencies[an] = an.right_value.accept(self)
                    sq.replace_arg(i, new_id)
            dependencies += sq.arguments[i].accept(self)
        return dependencies

    def visit_predicate_node(self, pr: PredicateNode):
        return []

    def visit_empty_node(self, pr: EmptyNode):
        return []

    def visit_assignment_node(self, an: AssignmentNode):
        self.left_values[an.left_value] = an
        self.fn_dependencies[an] = an.right_value.accept(self)

    def visit_right_value_node(self, rv: RValueNode):
        return rv.value.accept(self)

    def visit_literal_node(self, lit: LiteralNode):
        return []
