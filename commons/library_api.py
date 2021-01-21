import json
import pathlib
from re import finditer
from typing import List
from functools import reduce
from commons.z3_utils import create_var, analyze_type
from utils.logger import get_logger
from z3 import Optimize, Solver, set_param, set_option

logger = get_logger('library')
logger.setLevel("DEBUG")

def camel_case_split(identifier: str) -> List[str]:
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0).lower() for m in matches]


def get_tokens_from_code(code: str) -> List[str]:
    keywords = []

    word = ''
    for i in range(len(code)):
        c = code[i]

        if c.isalpha():
            word += c
        elif len(word) > 0:
            keywords.append(word)
            word = ''
            if c.isnumeric():
                keywords.append(c)
        elif len(word) == 0 and c.isnumeric():
            keywords.append(c)

    if len(word) > 0:
        keywords.append(word)

    if len(keywords) == 0:
        return []

    result = reduce(lambda x, y: x + y, map(camel_case_split, keywords))
    return result


class Argument:
    def __init__(self):
        self.name: str = ''
        self.is_optional: bool = True
        self.type: str = ''
        self.default_value: str = ''
        self.description: str = ''


class LibraryAPI:
    def __init__(self):
        self.id: str = ''  # e.g. tf.reshape
        self.name: str = ''  # e.g. reshape
        self.library: str = ''  # e.g. tf or torch
        self.description: str = ''  # e.g. reshape the tensor

        self.raw_code: str = ''  # e.g. tf.reshape(tensor, dims)
        self.raw_shape: str = ''  # e.g. Input: ... Output: ... where ...

        self.arguments: List[Argument] = []
        self.returns: Argument
        self.knowledge_base: Solver = Solver()
        self.knowledge_base.set('auto_config', False)
        self.knowledge_base.set('seed', 0)
        self.knowledge_base.set('sat.random_seed', 0)
        self.permanent_constraints = []
        # self.knowledge_base.set('smt.phase_selection', 0)

        self.argument_vars = []

    def __str__(self):
        return self.id

    def keep_basic_info_only(self):
        self.knowledge_base = None
        self.argument_vars = None

    def add_perm_constraint(self, idx, values):
        self.permanent_constraints += [self.argument_vars[idx] != values[idx]]

    def get_api_call_code(self, args: dict) -> str:
        result = self.id + '('
        for arg in self.arguments:
            if (arg not in args) and (not arg.is_optional):
                raise ValueError('Arg {0} is not filled in'.format(arg.name))

            if arg.is_optional:
                result += arg.name + '=' + arg.default_value
            else:
                result += args[arg.name]

            result += ' ,'

        result = result[:-1] + ')'
        return result

    def get_keywords(self) -> List[str]:
        return get_tokens_from_code(self.id)

    @staticmethod
    def from_json_dict(json_obj: dict) -> 'LibraryAPI':
        result = LibraryAPI()

        # set api fields
        result.id = json_obj['id']
        result.raw_code = json_obj['code']
        if 'shape' in json_obj:
            result.raw_shape = json_obj['shape']
        # result.raw_shape = json_obj['shape']
        result.name = json_obj['id'].split('.')[-1]
        result.library = json_obj['id'].split('.')[0]
        # we use the first sentence (summary) as the description for now TODO: use better description
        result.description = json_obj['summary']

        # set argument fields
        first_optional = False
        for json_arg in json_obj['code-info']['parameters']:
            arg = Argument()
            arg.name = json_arg['name']
            arg.is_optional = json_arg['is_optional']
            if arg.is_optional or json_arg['type'].find('Optional') != -1:
                first_optional = True
            if 'default_value' in json_arg and json_arg['default_value'] != '':
                arg.default_value = json_arg['default_value']
                first_optional = True
            arg.is_optional = first_optional
            arg.type = json_arg['type']
            arg.description = json_arg['description']
            if json_arg['type'].find('Union') != -1 or json_arg['type'].find('Optional') != -1:
                type_set = analyze_type(result.id, json_arg['type'])
                if len(type_set) > 1 and next(filter(lambda x: isinstance(x, tuple), type_set), None):
                    arg.type = next(filter(lambda x: isinstance(x, tuple), type_set), None)
                    if len(arg.type) == 1: arg.type = arg.type[0]
                else: arg.type = type_set.pop()
            elif json_arg['type'].find('List[str]') != -1 or json_arg['type'].find('Dict[str,str]') != -1:
                arg.type = analyze_type(result.id, json_arg['type']).pop()
            elif arg.type not in ['int', 'bool', 'float', 'string', 'str', 'tensor', 'others']:
                arg.type = 'others'
            try:
                result.arguments.append(arg)
                result.argument_vars.append(create_var(arg.type, arg.name))
            except:
                logger.error(f'Could not load {arg}')

        return result


class Library:

    def __init__(self):
        self.name = ''
        self.apis: List[LibraryAPI] = []

    def __len__(self):
        return self.apis.__len__()

    @staticmethod
    def from_json_list(json_list: List[dict]) -> 'Library':
        results = [LibraryAPI.from_json_dict(api) for api in json_list]

        # deal with duplicates ids
        dedup_results = []
        api_dict = dict()
        for api in results:
            if api.id in api_dict:
                logger.debug('Duplicates found: {0} and {1}'.format(api_dict[api.id].raw_code, api.raw_code))
            else:
                api_dict[api.id] = api
                dedup_results.append(api)

        library = Library()
        library.name = results[0].library
        library.apis = dedup_results

        return library


def load_apis(library_name: str, basic_info=False) -> List[LibraryAPI]:
    return load_library(library_name, basic_info).apis


def load_library(library_name: str, basic_info=False) -> Library:
    # first get the path to the preprocessed data
    data_file = pathlib.Path(__file__).parent.parent.absolute() \
        .joinpath('crawler/preprocessed_{}_docs.json'.format(library_name))

    with open(data_file) as f:
        apis = json.load(f)
        library = Library.from_json_list(apis)

        if basic_info:
            for api in library.apis:
                api.keep_basic_info_only()

        return library
