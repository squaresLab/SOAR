from abc import ABC, abstractmethod
from typing import Any
from synthesis.synthesizer.torch_to_tf.tf_enumerator import *
from synthesis.synthesizer.torch_to_tf.tf_decider import *
from synthesis.synthesizer.synthesizer import *
from synthesis.synthesizer.torch_to_tf.tf_spec import *
from autotesting.auto_test_generation.single_api_test_generation import *
from mapping.representations import *
from synthesis.search_structure import *
from commons.interfaces import ApiMatching
from commons.test_utils import Interpreter, extract_api_arguments
from commons.library_api import load_apis
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
from utils.logger import get_logger
import re
import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import TimeoutError
from argparse import ArgumentParser


executor = concurrent.futures.ProcessPoolExecutor()

logger = get_logger('synthesizer')
logger.setLevel('DEBUG')


class TFSynthesizer(Synthesizer):
    """ One to one synthesizer based on Z3. """

    def __init__(self, source_program: Program, source_library: str,
                 target_library: str, input: np.ndarray, err_enabled: bool = False):
        self.source_program = source_program
        self.source_library = source_library
        self.target_library = target_library
        self.api_matcher = ApiMatching.get_matcher(source_library, target_library, use_embedding=False, k=3000)
        self.input = input
        self.tf_input = input
        self.err_enabled = err_enabled
        self.rx = re.compile(r'\(.*\)')
        self.interpreter = Interpreter()

    def synthesize(self) -> Optional[Program]:
        rank = 0
        synthesized_code = []
        n_mappings = len(self.source_program.code)
        # one-to-one mapping, api-wise
        for i in range(n_mappings):
            # first we load the src_api
            src_api_call = self.source_program.code[i]
            src_api_name = src_api_call.split('=')[1].split('(')[0].strip()
            src_api = self.api_matcher.get_api(src_api_name)

            # now we build a test case
            test_case = self.create_test_case(src_api, src_api_call)

            # then we get the top-10 matching apis in the target library
            matching_apis = self.api_matcher.api_matching(src_api)
            # we don't need the probabilities for now
            matching_apis = list(filter(lambda x: x.id.find('keras.layers') != -1, map(lambda x: x[0], matching_apis)))

            # create a unit program
            unit_program = Program([src_api_call])

            # create a enumerator and a decider
            spec = TFSpec(test_case)
            enumerator = TFEnumerator(unit_program, matching_apis, spec)
            decider = TFDecider([test_case], matching_apis, self.interpreter, self.api_matcher)
            f = open('errors.txt', 'a+')
            # one to one mapping
            while enumerator.has_next():
                program = enumerator.next()
                result = decider.analyze(program)
                if result.is_correct():
                    synthesized_code += program.code
                    enumerator.delete()
                    self.tf_input = self.input
                    break
                #elif self.err_enabled and result.error_message() is not None:
                #    try:
                #        constraint, var_names = decider.error_message_understanding(result.error_message(), program)
                #        enumerator.update(constraint, var_names)
                #    except Exception as e:
                #        logger.error(e)

        logger.info('Success!')
        for line in synthesized_code:
            logger.info(line)

    def create_test_case(self, src_api, src_api_call):
        if self.rx.search(src_api_call) is None:
            _, output = self.interpreter.execute_api_call_no_args(src_api, self.input)
        else:
            src_api_call = src_api_call[src_api_call.find("=") + 1:]
            layer = self.interpreter.create_layer_torch(src_api_call)
            _, output = self.interpreter.torch_fd_pass(layer, self.input)
        test_case = TestCase({'tf': self.tf_input, 'torch': self.input}, output, src_api)
        self.input = output
        return test_case


def run_benchmark(benchmark, test_cases):
    s_src_lines, fp_src_lines = load_example_by_name(f'../benchmarks_tf/{benchmark}')
    s_program = Program(s_src_lines)
    synth = TFSynthesizer(s_program, 'torch', 'tf', input=test_cases[benchmark], err_enabled=True)
    start = time.time()
    res = synth.synthesize()
    return res, time.time() - start


if __name__ == '__main__':
    test_cases = {'discriminator': np.random.rand(100, 1, 50, 40), 'basic': np.random.rand(100, 50),
                  'conv': np.random.rand(100, 1, 50, 40), 'conv_m': np.random.rand(100, 1, 50, 40),
                  'nlp_conv1d': np.random.randint(50, size=(32, 10)),
                  'word2vec': np.random.randint(1000, size=(32, 10)),  'nlp': np.random.randint(50, size=(32, 10)),
                  'densenet': np.random.rand(10, 3, 224, 224),
                  'auto_encoder': np.random.rand(10, 32, 7, 7),
                  'alexnet': np.random.rand(10, 3, 227, 227),
                  'generator': np.random.rand(10, 6272),
                  'lenet': np.random.rand(10, 1, 32, 32),
                  'vgg11': np.random.rand(10, 3, 224, 224), 'vgg16': np.random.rand(10, 3, 224, 224),
                  'vgg19': np.random.rand(10, 3, 224, 224),
                  'densenet_part1': np.random.rand(10, 3, 224, 224),
                  'densenet_part2': np.random.rand(10, 3, 224, 224), 'lstm': np.random.rand(10, 1, 28, 28),
                  'densenet_transition_block': np.random.rand(10, 3, 224, 224)
                  }
    run_benchmark('alexnet', test_cases)