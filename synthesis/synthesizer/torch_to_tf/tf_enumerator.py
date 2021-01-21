from synthesis.synthesizer.enumerator import *
from synthesis.search_structure import *
from synthesis.synthesizer.torch_to_tf.tf_spec import *
from functools import reduce


class OneToOne:

    def __init__(self, primary_api: LibraryAPI, var_pool: VarPool):
        self.primary_api = primary_api
        self.variables = var_pool
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
        self.primary_api.knowledge_base.add(And(self.primary_api.permanent_constraints))
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
            for value in sorted(values):
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
            if self.primary_api.arguments[i].type == 'int':
                self.primary_api.knowledge_base.add(self.primary_api.argument_vars[i] < 10000)

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

        function = function.format(','.join(arguments))

        s_src_line = 'self.var' + trial_name + ' = ' + function
        return s_src_line

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


class TFEnumerator(Enumerator):
    """ Unit Program Enumerator based on Z3. """
    counter = 0

    def __init__(self, source_program: Program, matching_apis: List, spec: TFSpec):
        super().__init__(source_program)
        self.matching_apis = matching_apis[1:]
        self.spec = spec
        self.var_pool = VarPool.get_preset_vals()
        self.var_pool = VarPool.combine_pool(self.var_pool, self.spec.values_from_test())
        self.var_pool = VarPool.combine_pool(self.var_pool, get_values_from_code(self.source_program.code[0]))
        self.current_tree = OneToOne(matching_apis[0], self.var_pool)
        ctr, var_names = self.spec.infer_ctr(matching_apis[0])
        if ctr is not None: self.current_tree.add_constraint(ctr, var_names)

    def next(self) -> Program:
        TFEnumerator.counter += 1
        trial_name = f'{TFEnumerator.counter}'
        s_src_line = self.current_tree.next(trial_name)
        return Program([s_src_line])

    def has_next(self) -> bool:
        if self.current_tree.has_next():
            return True
        elif len(self.matching_apis) > 0:
            self.current_tree = OneToOne(self.matching_apis[0], self.var_pool)
            ctr, var_names = self.spec.infer_ctr(self.matching_apis[0])
            if ctr is not None: self.current_tree.add_constraint(ctr, var_names)
            self.matching_apis = self.matching_apis[1:]
            return self.has_next()
        else:
            return False

    def update(self, ctr: Constraint, var_names: List[str]) -> None:
        if isinstance(ctr, Constraint):
            self.current_tree.add_constraint(ctr, var_names)

    def add_constraint(self, ctr: Constraint, var_names: List[str]):
        self.current_tree.add_constraint(ctr, var_names)

    def delete(self):
        self.current_tree.delete()
