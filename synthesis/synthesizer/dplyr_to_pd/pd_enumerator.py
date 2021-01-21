from synthesis.synthesizer.enumerator import *
from synthesis.search_structure import *
from synthesis.synthesizer.dplyr_to_pd.pd_spec import *
import pandas as pd


class OneToOne:

    def __init__(self, primary_api: LibraryAPI, var_pool: VarPool):
        self.primary_api = primary_api
        self.var_pool = var_pool
        self.blocking_template = None
        self.vars = []
        self.var_names = []
        self.var_types = []
        self.str_mapping = {}
        self.other_mapping = {}
        self.build_tree()

    def build_tree(self):
        logger.debug(self.primary_api)
        self.primary_api.knowledge_base.push()
        for i in range(len(self.primary_api.argument_vars)):
            if self.primary_api.arguments[i].is_optional or self.primary_api.arguments[i].name.startswith('self'):
                continue
            self.vars += [self.primary_api.argument_vars[i]]
            self.var_types += [self.primary_api.arguments[i].type]
            self.var_names += [self.primary_api.arguments[i].name]

            if self.primary_api.arguments[i].type == 'List[str]':
                values = self.var_pool['string']
                self.str_mapping.update({i: values[i] for i in range(len(values))})
                self.str_mapping.update({values[i]: i for i in range(len(values))})

                length = len(self.primary_api.argument_vars[i]) - 1
                constraint = [self.primary_api.argument_vars[i][0] >= 1, self.primary_api.argument_vars[i][0] <= length]
                for j in range(length):
                    # values for each arg
                    arg_values = Or([self.primary_api.argument_vars[i][j+1] == self.str_mapping[value] for value in values])
                    constraint.append(arg_values)

                    # block models that don't make sense
                    block = []
                    for k in range(j+1, length):
                        block += [self.primary_api.argument_vars[i][k+1] == 0]
                    constraint.append(Implies(self.primary_api.argument_vars[i][0] == j+1, And(block)))
                    constraint.append(Distinct([self.primary_api.argument_vars[i][j+1] for j in range(length)]))
                self.primary_api.knowledge_base.append(constraint)
            elif self.primary_api.arguments[i].type == 'Dict[str,str]':
                values = self.var_pool['string']
                self.str_mapping.update({i: values[i] for i in range(len(values))})
                self.str_mapping.update({values[i]: i for i in range(len(values))})

                length = len(self.primary_api.argument_vars[i]) - 1
                constraint = [self.primary_api.argument_vars[i][0] >= 1, self.primary_api.argument_vars[i][0] <= 1]
                for j in range(length):
                    # values for each arg
                    arg_values = Or([self.primary_api.argument_vars[i][j+1][0] == self.str_mapping[value] for value in values])
                    constraint.append(arg_values)

                    arg_values = Or([self.primary_api.argument_vars[i][j+1][1] == self.str_mapping[value] for value in values])
                    constraint.append(arg_values)

                    # block models that don't make sense
                    block = []
                    for k in range(j+1, length):
                        block += [self.primary_api.argument_vars[i][k+1][0] == 0]
                        block += [self.primary_api.argument_vars[i][k + 1][1] == 0]
                    constraint.append(Implies(self.primary_api.argument_vars[i][0] == j+1, And(block)))
                    constraint.append(Distinct([self.primary_api.argument_vars[i][j+1][0] for j in range(length)]))
                    constraint.append(Distinct([self.primary_api.argument_vars[i][j+1][1] for j in range(length)]))

                self.primary_api.knowledge_base.append(constraint)
            elif self.primary_api.arguments[i].type == 'str':
                values = self.var_pool['string']
                constraint = Or([self.primary_api.argument_vars[i] == StringVal(value) for value in values])
                self.primary_api.knowledge_base.append(constraint)
            elif self.primary_api.arguments[i].type == 'int':
                values = self.var_pool['int']
                constraint = Or([self.primary_api.argument_vars[i] == value for value in values])
                self.primary_api.knowledge_base.append(constraint)
            elif self.primary_api.arguments[i].type == 'bool':
                values = [0, 1]
                constraint = Or([self.primary_api.argument_vars[i] == value for value in values])
                self.primary_api.knowledge_base.append(constraint)
            elif self.primary_api.arguments[i].type == 'others':
                values = self.var_pool['others']
                self.other_mapping.update({i: values[i] for i in range(len(values))})
                self.other_mapping.update({values[i]: i for i in range(len(values))})
                values = self.var_pool['others']
                constraint = Or([self.primary_api.argument_vars[i] == self.other_mapping[value] for value in values])
                self.primary_api.knowledge_base.append(constraint)

        self.blocking_template = blocking_template(self.vars, self.var_types)

    def next(self):
        model = self.primary_api.knowledge_base.model()
        values = []
        for val in self.vars:
            values += self.extract_model(model, val)
        block_model(self.primary_api.knowledge_base, self.blocking_template, values)
        function = self.primary_api.name + '({})'
        arguments = []

        for i in range(len(self.vars)):
            var_type = self.var_types[i]
            if self.var_names[i] == '':
                arguments = [model[self.vars[i]].as_string()]
            elif var_type == 'Dict[str,str]':
                size = model[self.vars[i][0]].as_long()
                value = {}
                for j in range(size):
                    str_code1 = model[self.vars[i][j+1][0]].as_long()
                    str_code2 = model[self.vars[i][j+1][1]].as_long()
                    value[self.str_mapping[str_code1]] = self.str_mapping[str_code2]
                arguments += [f'{self.var_names[i]}={str(value)}']
            elif var_type == 'List[str]' and self.var_names[i] != 'dict_idx':
                size = model[self.vars[i][0]].as_long()
                value = []
                for j in range(size):
                    str_code = model[self.vars[i][j+1]].as_long()
                    value += [f'{self.str_mapping[str_code]}']
                arguments += [f'{self.var_names[i]}={str(value)}']
            elif var_type == 'str':
                str_code = model[self.vars[i]].as_string()
                arguments += [f'{self.var_names[i]}=\'{str_code}\'']
            elif var_type == 'int':
                integer = model[self.vars[i]].as_long()
                arguments += [f'{self.var_names[i]}={integer}']
            elif var_type == 'bool':
                mapping = {0: False, 1: True}
                integer = model[self.vars[i]].as_long()
                arguments += [f'{self.var_names[i]}={mapping[integer]}']
            elif var_type == 'others':
                other_code = model[self.vars[i]].as_long()
                arguments += [f'{self.var_names[i]}={self.other_mapping[other_code]}']

        if 'dict_idx' in self.var_names:
            i = self.var_names.index("dict_idx")
            size = model[self.vars[i][0]].as_long()
            value = []
            for j in range(size):
                str_code = model[self.vars[i][j + 1]].as_long()
                value += [f'{self.str_mapping[str_code]}']
            return Program(['lambda x: x.' + function.format(','.join(arguments)) + f'[{value}]'])
        else:
            return Program(['lambda x: x.' + function.format(','.join(arguments))])

    def extract_model(self, model, val):
        values = []
        if isinstance(val, list):
            for sub_val in val:
                values += self.extract_model(model, sub_val)
        else:
            values += [model[val]]
        return values

    def has_next(self):
        if self.primary_api.knowledge_base.check() == sat:
            return True
        else:
            self.primary_api.knowledge_base.pop()
            return False

    def delete(self):
        self.primary_api.knowledge_base.pop()


class PDEnumerator(Enumerator):
    """ Unit Program Enumerator based on Z3. """
    counter = 0

    def __init__(self, source_program: Program, matching_apis: List, spec: PDSpec):
        super().__init__(source_program)
        self.matching_apis = matching_apis
        self.spec = spec
        self.var_pool = VarPool.get_preset_vals()
        self.var_pool = self.get_values_from_table()
        self.var_pool = self.get_values_from_code()
        self.current_tree = self.create_new_tree()
        pass

    def next(self) -> Program:
        return self.current_tree.next()

    def has_next(self) -> bool:
        if self.current_tree.has_next():
            return True
        elif len(self.matching_apis) > 0:
            self.current_tree = self.create_new_tree()
            return self.has_next()
        else:
            return False

    def update(self, ctr: Constraint, var_names: List[str]) -> None:
        pass

    def delete(self):
        self.current_tree.delete()

    def create_new_tree(self):
        current_tree = OneToOne(self.matching_apis[0], self.var_pool)
        self.matching_apis = self.matching_apis[1:]
        return current_tree

    def get_values_from_table(self):
        if not isinstance(self.spec.input, pd.core.groupby.DataFrameGroupBy):
            for col in self.spec.input.columns:
                self.var_pool['string'] += [str(col)]
        return self.var_pool

    def get_values_from_code(self):
        raw_arguments = re.search(r'\((.*)\)', str(self.source_program))
        raw_arguments = raw_arguments.group(1)
        args = []
        concat = ''
        inside_brackets = 1
        for c in raw_arguments:
            if c == ',' and inside_brackets == 1:
                args += self.process_string(concat)
                concat = ''
                continue
            elif inside_brackets > 0:
                concat += c
            elif c == '(' or c == '[':
                inside_brackets += 1
            elif c == ')' or c == ']':
                inside_brackets -= 1
        args += self.process_string(concat)
        for i in range(len(args)):
            self.convert_arg(args[i])
        return self.var_pool

    def convert_arg(self, string):
        is_int = re.compile(r"^([+-]?([1-9]\d*|0))$")
        is_float = re.compile(r"^(\d*\.\d+)$")
        if is_int.match(string):
            self.var_pool['int'].append(int(string))
        elif is_float.match(string):
            self.var_pool['float'].append(float(string))
        else:
            self.var_pool['string'].append(string)

    def process_string(self, concat):
        concat = concat.strip()
        all_str = [concat]
        regex = re.compile(r'\((.*)\)')
        assign = re.compile(r'(.*)=(.*)$')
        if regex.search(concat):
            all_str += [regex.search(concat).group(1)]
        if assign.match(concat):
            mapping = {}
            all_str += list(map(lambda x: x.strip(), concat.split('=')))
            matching = re.search(r'([a-z0-9]+)[ -/+*]+([a-z0-9]+)$', concat)
            if matching:
                for group in matching.groups():
                    mapping[group] = f'x["{group}"]'
                for key in mapping:
                    concat = concat.replace(key, mapping[key])
                all_str += [concat]
        return all_str
