import itertools

from typing import List, Dict
from collections import namedtuple
from commons.library_api import LibraryAPI
import re

Var = namedtuple('Variable', ['type', 'identifier'])


class VarPool:
    types = ['bool', 'tensor', 'int', 'string', 'float', 'others', tuple, list]

    def __init__(self):
        self.pool = VarPool.get_empty_pool()

    def __getitem__(self, item):
        out = self.pool.get(item, None)
        if out is None:
            out = self.pool.get(tuple)
        return out

    def __setitem__(self, key, value):
        return self.pool.__setitem__(key, value)

    def append(self, t, item):
        self.pool[t].insert(0, item)

    def get_all_vals(self):
        return list(itertools.chain.from_iterable(self.pool.values()))

    def wrap_for_tf_tensor(self):
        self['tensor'] = list(map(lambda t: 'tf.convert_to_tensor(' + t + ')', self['tensor']))

    def wrap_for_torch_tensor(self):
        self['tensor'] = list(map(lambda t: 'torch.Tensor(' + t + ')', self['tensor']))

    @staticmethod
    def combine_pool(pool_1, pool_2):
        result = VarPool()
        for t in VarPool.types:
            result[t] = pool_1[t][:]
            for val in pool_2[t]:
                if val not in result[t]:
                    result[t].append(val)
        return result

    @staticmethod
    def get_empty_pool():
        result = dict()
        for t in VarPool.types:
            result[t] = []
        return result

    @staticmethod
    def get_preset_vals():
        result = VarPool()
        result['bool'] = ['True', 'False']
        result['tensor'] = [
            'np.random.randint(1, 6, size=(3))',
            'np.random.randint(1, 6, size=(3, 4))',
            'np.random.randint(1, 6, size=(3, 4, 5))',
            'np.random.randint(1, 6, size=(3, 4, 5, 6))',
            'np.random.rand(3)', 'np.random.rand(3, 4)',
            'np.random.rand(3, 4, 5)', 'np.random.rand(3, 4, 5, 6)',
        ]
        result['int'] = [str(i) for i in range(-2, 4)]
        result['string'] = ["''", "same", 'valid']
        result['float'] = [str(0.1)]
        result['others'] = ['None'] + [str(i) for i in range(-2, 4)]
        result[tuple] = [(i,) for i in range(-2, 4)]
        result[list] = [(i,) for i in range(-2, 4)]
        return result

    @staticmethod
    def get_preset_pool():
        result = VarPool()
        result.pool = VarPool.get_preset_vals()
        return result


class Node:

    def __init__(self, val: str):
        self.value: str = val
        self.value_candidates: List[str] = None
        self.children: List['Node'] = []


class SearchableSyntaxTree:

    def __init__(self):
        self.root: Node = None
        self.search_nodes: List[Node] = []

    def build_tree(self, api: LibraryAPI, variables: VarPool):
        # build root node
        self.root = Node(api.id)
        args = [Node('(')]
        for arg in api.arguments:
            arg_node = Node('')
            candidate_list = []
            prefix = ''
            # FIXME: this is hard code
            if arg.is_optional and arg.name != 'stride':
                candidate_list.append(arg.default_value)
                arg_node.value = arg.name + '=' + arg.default_value
            else:
                if not arg.type == '':
                    candidate_list += variables[arg.type]
                elif arg.type == 'others':
                    candidate_list += variables['tensor']
                else:
                    candidate_list += variables.get_all_vals()

                if arg.is_optional:
                    candidate_list = [arg.name + '=' + cand for cand in candidate_list]

                arg_node.value_candidates = candidate_list
                self.search_nodes.append(arg_node)

            args.append(arg_node)

        args.append(Node(')'))
        self.root.children = args

    def linearize_tree(self, choices: List[int]):
        result = ''
        result += self.root.value
        search_counter = 0
        for child in self.root.children:
            if child.value_candidates is not None:
                result += child.value_candidates[choices[search_counter]]
                search_counter += 1
            else:
                result += child.value

            if child.value == ')':
                result = result[:-2] + ')'
            elif not child.value == '(':
                result = result + ','
        return result


def get_values_from_code(line: str) -> VarPool:
    result = VarPool()
    if line.find('(') == -1:
        return result
    line = re.match(r'(.*)\)', line[line.find('(') + 1:]).group(1)
    argument_fields = re.sub(r'(\(|\[)[^(]*(\)|\])', lambda x: x.group(0).replace(',', '&'), line).split(',')
    argument_fields = list(map(lambda x: x.strip().replace("&", ","), argument_fields))

    if argument_fields == ['']:
        return result
    for argument in argument_fields:
        argument = argument.strip()

        if '=' in argument:
            argument = argument.split('=')[1]

        if re.match('\(.*\)', argument):
            result.append(tuple, eval(argument))
        if re.match('\[.*\]', argument):
            result.append(list, eval(argument))

        if argument.isdecimal():
            result.append('int', argument)
            result.append(tuple, (eval(argument),))
        elif argument.isnumeric():
            result.append('float', argument)
        elif argument[0] == '\'' and argument[-1] == '\'':
            result.append('string', argument)
        else:
            result.append('others', argument)

    return result


def get_tryout_combinations(sizes: List[int]):
    if len(sizes) == 0:
        return [[]]

    multipliers = []
    for i in range(len(sizes)):
        multiplier = 1
        for j in range(i + 1, len(sizes)):
            multiplier = multiplier * sizes[j]
        multipliers.append(multiplier)

    result = []
    for i in range(multipliers[0] * sizes[0]):
        lst = []
        num = i
        for multiplier in multipliers:
            lst.append(num // multiplier)
            num = num % multiplier
        result.append(lst)

    return result


def get_tryout_name(indices: List[int]):
    name = ''
    for i in indices:
        name += str(i)
    return name
