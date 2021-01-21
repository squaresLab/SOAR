from commons.interfaces import ApiMatching
from synthesis.synthesizer.synthesizer import *
from synthesis.synthesizer.dplyr_to_pd.pd_decider import *
from synthesis.synthesizer.dplyr_to_pd.pd_enumerator import *
from synthesis.synthesizer.dplyr_to_pd.pd_spec import *
from utils.logger import get_logger
from synthesis.synthesizer.dplyr_to_pd.code_analysis.visitor import DplyrTransformer, DependencyFinder, RWriter
from synthesis.synthesizer.dplyr_to_pd.code_analysis.graph import Graph
from lark import Lark
import concurrent.futures
import time
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from argparse import ArgumentParser
import pathlib
import ast

executor = concurrent.futures.ProcessPoolExecutor()

logger = get_logger('synthesizer')
logger.setLevel('INFO')


class PDSynthesizer(Synthesizer):
    """ One to one synthesizer based on Z3. """

    def __init__(self, source_program: Program, source_library: str, target_library: str, inputs: dict, ground: str):
        self.source_program = source_program
        self.source_library = source_library
        self.target_library = target_library
        self.api_matcher = ApiMatching.get_matcher(source_library, target_library,
                                                   use_embedding=True, use_description=False, k=300)
        self.api_matcher_tf = ApiMatching.get_matcher(source_library, target_library,
                                                   use_embedding=False, use_description=False, k=300)
        self.inputs = inputs
        self.outputs = {}
        self.real_mapping = ast.literal_eval(ground)

    def synthesize(self) -> Optional[Program]:
        ranks = []
        synthesized_code = []
        n_mappings = len(self.source_program.code)
        # calculate glove
        for i in range(n_mappings):
            # first we load the src_api
            src_api_call = self.source_program.code[i]
            src_api_name = src_api_call.split('<-')[1].split('(')[0].strip()
            src_api_args = re.search(r'\((.+)\)', src_api_call.split('<-')[1])
            src_api = self.api_matcher.get_api(src_api_name)


            # then we get the top-10 matching apis in the target library
            matching_apis = self.calculate_api_ranking(src_api, src_api_args, src_api_name)
            ground_truth = self.real_mapping[src_api_name]
            idxs = [i+1 for i in range(len(matching_apis)) if matching_apis[i].id.find(ground_truth) != -1]
            ranks += [idxs[0]]
            
        self.api_matcher = self.api_matcher_tf
        ranks_tfidf = []
        # calculate tf-idf
        for i in range(n_mappings):
            # first we load the src_api
            src_api_call = self.source_program.code[i]
            src_api_name = src_api_call.split('<-')[1].split('(')[0].strip()
            src_api_args = re.search(r'\((.+)\)', src_api_call.split('<-')[1])
            src_api = self.api_matcher.get_api(src_api_name)


            # then we get the top-10 matching apis in the target library
            matching_apis = self.calculate_api_ranking(src_api, src_api_args, src_api_name)
            ground_truth = self.real_mapping[src_api_name]
            idxs = [i+1 for i in range(len(matching_apis)) if matching_apis[i].id.find(ground_truth) != -1]
            ranks_tfidf += [idxs[0]]

        import sys
        logger.info(f'Avg Rank GLOVE {sum(ranks)/n_mappings}')
        logger.info(f'Avg Rank TFIDF {sum(ranks_tfidf)/n_mappings}')
        sys.exit(0)

    def calculate_api_ranking(self, src_api, src_api_args, src_api_name):
        matching_apis = self.api_matcher.api_matching(src_api)
        if src_api_args:
            src_api_args = src_api_args.group(1).split(',')
            sub_api = next(filter(lambda x: re.search(r'[a-z]+\(.*\)', x), src_api_args), None)
            if sub_api:
                sub_api = re.search(r'([a-z]+)\((.*)\)', sub_api).group(1).strip()
                query = f'{src_api_name} {sub_api}'
                matching_apis = self.api_matcher.query_for_new_api(query)
        # we don't need the probabilities for now
        matching_apis = list(map(lambda x: x[0], matching_apis))
        return matching_apis

    def create_test_case(self, src_api, src_api_call):
        l_val = src_api_call.split('<-')[0].strip()
        self.outputs['r'] = ro.r(src_api_call)
        with localconverter(ro.default_converter + pandas2ri.converter):
            self.outputs['pandas'] = ro.conversion.rpy2py(self.outputs['r'])
            self.outputs['pandas_count'] = ro.conversion.rpy2py(ro.r(f'{l_val} %>% count'))
        test = TestCase(self.inputs, self.outputs, src_api, src_api_call)
        return test


def load_dplyr(path):

    f = open(pathlib.Path(__file__).parent.absolute().joinpath('code_analysis/grammar.lark'))
    parser = Lark(f.read(), start='lines', parser='lalr', transformer=DplyrTransformer())
    with open(path, 'r') as f:
        code = f.read()
        tree = parser.parse(code)
    finder = DependencyFinder(n_inputs=1)
    function_dep = tree.accept(finder)
    g = Graph()
    for function in function_dep:
        for dep in function_dep[function]:
            g.edge(function, dep)
    return g.dfs()


def load_csv(path):
    inputs = {}
    ro.globalenv['input1'] = ro.DataFrame.from_csvfile(str(path))
    inputs['r'] = ro.globalenv['input1']
    with localconverter(ro.default_converter + pandas2ri.converter):
        inputs['pandas'] = ro.conversion.rpy2py(ro.globalenv['input1'])
    return inputs


def run_benchmark(source_path, input_path):
    lines = load_dplyr(pathlib.Path(__file__).parent.absolute().joinpath(f'dplyr/{source_path}'))
    with open(pathlib.Path(__file__).parent.absolute().joinpath(f'dplyr_ground/{source_path}'), 'r+') as f:
        ground = f.read()
    s_program = Program(lines)
    inputs = load_csv(pathlib.Path(__file__).parent.absolute().joinpath(f'dplyr/inputs/{input_path}'))
    synthesizer = PDSynthesizer(s_program, 'dplyr', 'pd', inputs, ground)
    start = time.time()
    result = synthesizer.synthesize()
    return result, time.time() - start


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-b", "--benchmark", type=str)
    cmd_args = arg_parser.parse_args()

    res = run_benchmark(f'{cmd_args.benchmark}.R', f'{cmd_args.benchmark}.csv')
    logger.info(f'Synthesis time: {res[1]}')
    logger.info(res[0])

