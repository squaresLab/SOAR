from z3 import *
import re

# hack
BooleanSort = IntSort
TensorSort = IntSort
OtherSort = IntSort


def Boolean(name):
    return Const(name, BooleanSort())


def Tensor(name):
    return Const(name, TensorSort())


def Other(name):
    return Const(name, OtherSort())


def create_var(var_type, var_name):
    if var_type == 'int':
        return Int(var_name)
    elif var_type == 'bool':
        return Boolean(var_name)
    elif var_type == 'float':
        return Real(var_name)
    elif var_type == 'string' or var_type == 'str':
        return String(var_name)
    elif var_type == 'tensor' or var_type == 'torch.Tensor':
        return Tensor(var_name)
    elif var_type == 'others':
        return Other(var_name)
    elif isinstance(var_type, tuple):
        return IntVector(var_name, len(var_type))
    elif var_type == 'List[str]':
        return [Int(var_name + '_size')] + IntVector(var_name, 2)
    elif var_type == 'Dict[str,str]':
        return [Int(var_name + '_size')] + [IntVector(f'{var_name}_{i}', 2) for i in range(2)]
    else:
        return Other(var_name)


def blocking_template(vars, types):
    block = []
    j = 0
    for i in range(len(vars)):
        if types[i] == 'int':
            block.append(vars[i] == Var(j, IntSort()))
        elif types[i] == 'bool':
            block.append(vars[i] == Var(j, BooleanSort()))
        elif types[i] == 'float':
            block.append(vars[i] == Var(j, RealSort()))
        elif types[i] == 'string' or types[i] == 'str':
            block.append(vars[i] == Var(j, StringSort()))
        elif types[i] == 'tensor':
            block.append(vars[i] == Var(j, TensorSort()))
        elif types[i] == 'others':
            block.append(vars[i] == Var(j, OtherSort()))
        elif types[i] == 'List[str]':
            for var in vars[i]:
                block.append(var == Var(j, IntSort()))
                j += 1
            j -= 1 # hack
        elif types[i] == 'Dict[str,str]':
            block.append(vars[i][0] == Var(j, IntSort()))
            j += 1
            for var in vars[i][1:]:
                block.append(var[0] == Var(j, IntSort()))
                block.append(var[1] == Var(j+1, IntSort()))
                j += 2
            j -= 1 # hack
        elif isinstance(types[i], tuple) or isinstance(types[i], list):
            for var in vars[i]:
                block.append(var == Var(j, IntSort()))
                j += 1
            j -= 1  # hack
        j += 1

    ctr = And(block)
    return Not(ctr)


def block_model(z3_solver, template, values):
    ctr = substitute_vars(template, *values)
    z3_solver.add(ctr)


def analyze_type(name, type):
    match_union = re.match(r'Union\[(.*)\]', type)
    match_optional = re.match(r'Optional\[(.*)\]', type)
    match_tuple = re.match(r'Tuple\[(.*)\]', type)
    match_list = re.match(r'List\[(.*)\]', type)
    match_dict = re.match(r'Dict\[(.*)\]', type)

    if match_union:
        union = match_union.group(1)
        types = re.sub(r'\[.*?\]', lambda x: x.group(0).replace(',', '&'), union).split(',')
        possible_types = set()
        for type in types:
            infer = analyze_type(name, type.replace("&", ","))
            possible_types = possible_types.union(infer)
        return possible_types
    elif match_optional:
        union = match_optional.group(1)
        return analyze_type(name, union)
    elif match_tuple:
        union = match_tuple.group(1)
        if name.find("1d") != -1:
            return {('int',)}
        elif name.find("2d") != -1:
            return {('int', 'int')}
        elif name.find("3d") != -1:
            return {('int', 'int', 'int')}
        else:
            return {tuple(['int' for _ in range(len(union.split(',')))])}
    elif match_list or match_dict:
        return {type}
    elif type == 'T':
        return {'int'}
    else :
        return {'others'}
