from synthesis.synthesizer.enumerator import *
from synthesis.search_structure import *
from synthesis.synthesizer.tf_to_torch.torch_spec import *
from functools import reduce
import json

torch_dict = {}

class OneToOne:

    def __init__(self, primary_api: LibraryAPI, var_pool: VarPool, values_from_code={}):
        self.primary_api = primary_api
        self.variables = var_pool
        self.values_from_code = values_from_code
        self.vars = []
        self.all_vars = []
        self.var_names = []
        self.var_types = []
        self.all_var_types = []
        self.blocking_template = None
        self.model = None
        self.mapping = {}
        self.depth = 1
        self.special_optionals = ("stride", "dim", "padding")
        self.others_mapping = {}
        self.build_tree()

    def build_tree(self):
        self.primary_api.knowledge_base.push()
        for i in range(len(self.primary_api.arguments)):
            if self.primary_api.arguments[i].is_optional:
                if self.primary_api.arguments[i].name not in self.special_optionals:
                    continue

            values = self.variables[self.primary_api.arguments[i].type]
            self.vars += [self.primary_api.argument_vars[i]]
            self.all_vars += [self.primary_api.argument_vars[i]]
            self.var_names += [self.primary_api.arguments[i].name]
            self.var_types += [self.primary_api.arguments[i].type]
            self.all_var_types += [self.primary_api.arguments[i].type]

            if isinstance(self.primary_api.argument_vars[i], list):
                for k in range(len(self.primary_api.argument_vars[i])):
                    self.mapping[self.primary_api.arguments[i].name + f'_{k}'] = self.primary_api.argument_vars[i][k]
            else:
                self.mapping[self.primary_api.arguments[i].name] = self.primary_api.argument_vars[i]

            constraint = []
            j = 1
            other_count = 0
            available_values = []

            for value in sorted(values):
                if value in self.values_from_code:
                    pass
                if self.primary_api.arguments[i].type == 'int':
                    constraint.append(self.primary_api.argument_vars[i] == eval(value))
                elif self.primary_api.arguments[i].type == 'bool':
                    mapping = {False: 0, True: 1}
                    constraint.append(self.primary_api.argument_vars[i] == mapping[eval(value)])
                elif self.primary_api.arguments[i].type == 'string':
                    constraint.append(self.primary_api.argument_vars[i] == StringVal(value))
                elif isinstance(self.primary_api.arguments[i].type, tuple) and isinstance(value, tuple):
                    inner_constraint = []
                    if len(value) == len(self.primary_api.argument_vars[i]) > 1:
                        for j in range(len(value)):
                            inner_constraint.append(self.primary_api.argument_vars[i][j] == value[j])
                    if len(value) == 1:
                        for j in range(len(self.primary_api.argument_vars[i])):
                            inner_constraint.append(self.primary_api.argument_vars[i][j] == value[0])
                    constraint.append(And(inner_constraint))
                elif self.primary_api.arguments[i].type == 'others':
                    self.others_mapping[other_count] = value
                    constraint.append(self.primary_api.argument_vars[i] == other_count)
                    other_count += 1

                j += 1
            self.primary_api.knowledge_base.add(Or(constraint))

        self.blocking_template = blocking_template(self.vars, self.var_types)

    def next(self, trial_name):
        self.model = self.primary_api.knowledge_base.model()
        values = []
        for val in self.all_vars:
            if isinstance(val, list):
                for sub_val in val:
                    values += [self.model[sub_val]]
            else:
                values += [self.model[val]]
        values = list(filter(lambda x: x is not None, values))
        block_model(self.primary_api.knowledge_base, self.blocking_template, values)

        # now we have a model with the args, we just have to build the actual program
        function = self.primary_api.id + '({})'
        arguments = []
        for i in range(len(self.vars)):
            value = None
            var_type = self.var_types[i]
            if var_type == 'int' or var_type == 'float' or var_type == 'string':
                value = f'{str(self.model[self.vars[i]])}'
            elif var_type == 'bool':
                mapping = {0: False, 1: True}
                value = f'{mapping[self.model[self.vars[i]].as_long()]}'
            elif var_type == 'others':
                value = f'{self.others_mapping[self.model[self.vars[i]].as_long()]}'
            elif isinstance(var_type, tuple) or isinstance(var_type, list):
                mapping = {tuple: ('(', ')'), list: ('[', ']')}
                par = mapping[type(var_type)]
                value = []
                for val in self.vars[i]:
                    value += [str(self.model[val])]
                value = f'{par[0]}{",".join(value)}{par[1]}'

            if self.var_names[i] in self.special_optionals:
                arguments += [f'{self.var_names[i]}={value}']
            else:
                arguments += [f'{value}']

        if self.primary_api.id.find('torch.nn') != -1:
            function = function.format(','.join(arguments))
        else:
            function = 'lambda x: ' + function.format(','.join(['x'] + arguments))

        s_src_line = 'self.var' + trial_name + ' = ' + function
        return TorchProgram([s_src_line], self.var_names)

    def has_next(self):
        if self.primary_api.knowledge_base.check() == sat:
            return True
        else:
            self.primary_api.knowledge_base.pop()
            return False

    def add_constraint(self, ctr: Constraint, var_names: List[str]):
        values = list(map(lambda x: self.mapping[x], var_names))
        ctr = substitute_vars(ctr, *values)
        self.primary_api.knowledge_base.add(ctr)

    def delete(self):
        self.primary_api.knowledge_base.pop()


class OneToMany:

    def __init__(self, primary_api: LibraryAPI, surrounding_apis: List[LibraryAPI],
                 var_pool: VarPool, spec: TorchSpec, depth: int = 2, values_from_code={}):
        self.primary_api = primary_api
        self.surrounding_apis = surrounding_apis
        self.current_surrounding = None
        self.var_pool = var_pool
        self.spec = spec
        self.depth = depth
        self.values_from_code = values_from_code
        self.current_depth = 1

        # build structure
        self.primary = OneToOne(primary_api, var_pool, values_from_code=self.values_from_code)
        self.add_spec_constraints()
        self.vars = {}
        self.var_types = {}

    def add_spec_constraints(self):
        self.primary_api.knowledge_base.push()
        try:
            ctr, var_names = self.spec.infer_ctr(self.primary_api)
            if ctr is not None:
                self.add_constraint(ctr, var_names)
        except Exception as e:
            self.add_constraint(Not(True), [])
            logger.error(e)

    def build_tree(self, surrounding_api):
        primary_api = self.primary_api
        self.current_surrounding = surrounding_api
        if surrounding_api.id.find('torch.Tensor.permute') != -1:
            shape = self.spec.test_case.input['torch'].shape
            permute_vars = IntVector('permute_dim', len(shape))
            primary_api.knowledge_base.add(Distinct(permute_vars))
            [primary_api.knowledge_base.append(And(var >= 0, var < len(permute_vars))) for var in permute_vars]
            primary_api.knowledge_base.append(permute_vars[0] == 0)

            self.vars['permute'] = permute_vars
            self.var_types['permute'] = ['int' for _ in permute_vars]

            # change primary formula
            self.primary.all_vars += self.vars['permute']
            for i in range(len(shape)):
                self.primary.mapping[f'permute_dim_{i}'] = permute_vars[i]
            self.primary.all_var_types += self.var_types['permute']
            self.primary.blocking_template = blocking_template(self.primary.all_vars, self.primary.all_var_types)
        elif surrounding_api.id.find('torch.Tensor.long') != -1:
            pass
        elif surrounding_api.id.find('torch.Tensor.view') != -1:
            max_view_size = 4
            shape = self.spec.test_case.input['torch'].shape
            total = reduce(lambda x, y: x*y, shape)
            v_size = Int('view_size')
            primary_api.knowledge_base.append(And(v_size > 1, v_size <= max_view_size))

            view_vars = IntVector('view_dim', max_view_size)
            primary_api.knowledge_base.append(view_vars[0] == shape[0])  # batch size
            for i in range(max_view_size):
                primary_api.knowledge_base.append(view_vars[i] > 0)
            for i in range(2, max_view_size):
                ctr = []
                for j in range(i, max_view_size):
                    ctr += [view_vars[j] == 1]
                primary_api.knowledge_base.append(Implies(v_size == i, And(ctr)))
            primary_api.knowledge_base.append(Product(view_vars) == total)

            self.vars['view'] = [v_size] + view_vars
            self.var_types['view'] = ['int' for _ in self.vars['view']]

            # change primary formula
            self.primary.all_vars += self.vars['view']
            self.primary.mapping['view_size'] = v_size
            for i in range(len(shape)):
                self.primary.mapping[f'view_dim_{i}'] = view_vars[i]
            self.primary.all_var_types += self.var_types['view']
            self.primary.blocking_template = blocking_template(self.primary.all_vars, self.primary.all_var_types)

    def next(self, trial_name):
        program = self.primary.next(trial_name)
        if self.current_surrounding is not None:
            self.add_preprocessing(program)
        return program

    def add_preprocessing(self, program):
        model = self.primary.model
        if self.current_surrounding.id.find('torch.Tensor.permute') != -1:
            permute_values = [str(model[var].as_long()) for var in self.vars['permute']]
            program.before = ['{input}.permute' + f'({",".join(permute_values)})']
        elif self.current_surrounding.id.find('torch.Tensor.long') != -1:
            program.before = ['{input}.long()']
        elif self.current_surrounding.id.find('torch.Tensor.view') != -1:
            view_vars = self.vars['view']
            view_size = model[view_vars[0]].as_long()
            view_values = [str(model[view_vars[i+1]].as_long()) for i in range(view_size)]
            program.before = ['{input}.view' + f'({",".join(view_values)})']

    def has_next(self):
        if not self.primary.has_next():
            self.primary.primary_api.knowledge_base.pop()
            if self.surrounding_apis:
                api = self.surrounding_apis[0]
                self.surrounding_apis = self.surrounding_apis[1:]
                self.primary = OneToOne(self.primary_api, self.var_pool)
                self.build_tree(api)
                self.add_spec_constraints()
                return self.primary.has_next()
            return False
        return True

    def add_constraint(self, ctr: Constraint, var_names: List[str]):
        self.primary.add_constraint(ctr, var_names)

    def delete(self):
        self.primary.primary_api.knowledge_base.pop()
        self.primary.delete()


class TorchEnumerator(Enumerator):
    """ Unit Program Enumerator based on Z3. """
    counter = 0

    def __init__(self, source_program: Program, matching_apis: List[LibraryAPI],
                 surrounding_apis: List[LibraryAPI], spec: TorchSpec, depth: int = 1):
        super().__init__(source_program)
        self.matching_apis = matching_apis
        self.surrounding_apis = surrounding_apis
        self.spec = spec
        self.depth = depth
        self.var_pool = VarPool.get_preset_vals()
        self.var_pool = VarPool.combine_pool(self.var_pool, self.spec.values_from_test())
        self.var_pool = VarPool.combine_pool(self.var_pool, get_values_from_code(self.source_program.code[0]))
        self.values_from_code = {} #self.spec.values_from_test()
        self.current_tree = self.create_new_tree()

    def next(self) -> Program:
        TorchEnumerator.counter += 1
        trial_name = f'{TorchEnumerator.counter}'
        program_lines = self.current_tree.next(trial_name)
        return program_lines

    def has_next(self) -> bool:
        if self.current_tree.has_next():
            return True
        elif len(self.matching_apis) > 0:
            self.current_tree = self.create_new_tree()
            return self.has_next()
        else:
            return False

    def create_new_tree(self):
        current_tree = OneToMany(self.matching_apis[0], self.surrounding_apis, self.var_pool, self.spec, self.depth, values_from_code = self.values_from_code)
        self.matching_apis = self.matching_apis[1:]
        return current_tree

    def update(self, ctr: Constraint, var_names: List[str]) -> None:
        if isinstance(ctr, Constraint):
            self.current_tree.add_constraint(ctr, var_names)

    def add_constraint(self, ctr: Constraint, var_names: List[str]):
        self.current_tree.add_constraint(ctr, var_names)

    def delete(self):
        self.current_tree.delete()
