from abc import ABC, abstractmethod
from typing import Any
from synthesis.synthesizer.tf_to_torch.torch_enumerator import *
from synthesis.synthesizer.tf_to_torch.torch_decider import *
from synthesis.synthesizer.synthesizer import *
from synthesis.synthesizer.tf_to_torch.torch_spec import *
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
logger.setLevel('INFO')


class TorchSynthesizer(Synthesizer):
    """ One to one synthesizer based on Z3. """

    def __init__(self, source_program: Program, source_library: str,
            target_library: str, input: np.ndarray, err_enabled: bool = False, glove_enabled: bool = False, spec_enabled: bool = False):
        self.source_program = source_program
        self.source_library = source_library
        self.target_library = target_library
        self.api_matcher = ApiMatching.get_matcher(source_library, target_library, use_embedding=glove_enabled)
        self.input = input
        self.torch_input = input
        self.err_enabled = err_enabled
        self.spec_enabled = spec_enabled
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
            matching_apis = list(map(lambda x: x[0], matching_apis))
            matching_apis = list(map(lambda x: self.api_matcher.get_api('torch.Tensor') if x.id.find('reshape') != -1 else x, matching_apis))

            # create a unit program
            unit_program = Program([src_api_call])

            # create a enumerator and a decider
            surrounding_apis = [self.api_matcher.get_api('torch.Tensor.long'),
                                self.api_matcher.get_api('torch.Tensor.permute'),
                                self.api_matcher.get_api('torch.Tensor.view')
                                ]
            spec = TorchSpec(test_case, self.spec_enabled)
            enumerator = TorchEnumerator(unit_program, matching_apis, surrounding_apis, spec, depth=2)
            decider = Z3Decider([test_case], matching_apis, self.interpreter, self.api_matcher)
            
            # one to one mapping
            while enumerator.has_next():
                new_unit_prog = enumerator.next()
                result = decider.analyze(new_unit_prog)

                if result.is_correct():
                    synthesized_code += new_unit_prog.print()
                    enumerator.delete()
                    self.torch_input = result.torch_output()
                    rank += 200 - len(enumerator.matching_apis)
                    break

                elif self.err_enabled and result.error_message() is not None:
                    try:
                        constraint, var_names = decider.error_message_understanding(result.error_message(), new_unit_prog)
                        enumerator.update(constraint, var_names)
                    except Exception as e:
                        pass #logger.error(e)

        logger.info('Success!')
        logger.info(f'Avg Rank {rank/n_mappings}')
        return Program(synthesized_code)

    def create_test_case(self, src_api, src_api_call):
        if self.rx.search(src_api_call) is None:
            _, output = self.interpreter.execute_api_call_no_args(src_api, self.input)
        else:
            src_api_call = src_api_call[src_api_call.find("=") + 1:]
            layer = self.interpreter.create_layer_tf(src_api_call)
            _, output = self.interpreter.tf_forward_pass(layer, self.input)
        test_case = TestCase({'tf': self.input, 'torch': self.torch_input}, output, src_api)
        self.input = output
        return test_case


def run_benchmark(args, test_cases):
    s_src_lines, fp_src_lines = load_example_by_name(f'../benchmarks/{args.benchmark}')
    s_program = Program(s_src_lines)
    synth = TorchSynthesizer(s_program, 'tf', 'torch', input=test_cases[args.benchmark], err_enabled=args.errmsg, glove_enabled=args.glove, spec_enabled = args.spec)
    start = time.time()
    res = synth.synthesize()
    return res, time.time() - start


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-b", "--benchmark", type=str)
    arg_parser.add_argument("-glove", action="store_true")
    arg_parser.add_argument("-errmsg", action="store_true")
    arg_parser.add_argument("-spec", action="store_true")
    cmd_args = arg_parser.parse_args()
    results = []
    test_cases = {'conv_pool_softmax': np.random.rand(100, 50, 40, 1), 
                  'img_classifier': np.random.rand(100, 50, 40, 1),
                  'three_linear': np.random.rand(100, 50),
                  'embed_conv1d_linear': np.random.randint(50, size=(32, 10)),
                  'word_autoencoder': np.random.randint(1000, size=(32, 10)), 
                  'gan_discriminator': np.random.rand(100, 50, 40, 1), 
                  'two_conv': np.random.rand(10, 224, 224, 3),
                  'img_autoencoder': np.random.rand(10, 7, 7, 32),   
                  'alexnet': np.random.rand(10, 227, 227, 3),
                  'gan_generator': np.random.rand(10, 6272),                                  
                  'lenet': np.random.rand(10, 32, 32, 1),
                  'tutorial': np.random.rand(10, 28, 28, 1),
                  'conv_for_text': np.random.randint(50, size=(32, 10)),
                  'vgg11': np.random.rand(10, 224, 224, 3), 
                  'vgg16': np.random.rand(10, 224, 224, 3),
                  'vgg19': np.random.rand(10, 224, 224, 3),
                  'densenet_part1': np.random.rand(10, 224, 224, 3),
                  'densenet_part2': np.random.rand(10, 224, 224, 3), 
                  'densenet_conv_block': np.random.rand(10, 224, 224, 3),
                  'densenet_trans_block': np.random.rand(10, 224, 224, 3)
                  }

    res = run_benchmark(cmd_args, test_cases)
    logger.info(f'Synthesis time: {res[1]}')
    logger.info(res[0])
