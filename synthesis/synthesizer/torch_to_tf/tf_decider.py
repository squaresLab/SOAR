from synthesis.synthesizer.decider import *
from synthesis.synthesizer.torch_to_tf.tf_result import *
from synthesis.synthesizer.synthesizer import *
from synthesis.search_structure import *
from commons.test_utils import Interpreter, extract_api_arguments_torch, code_to_params
import numpy as np
import nltk
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import concurrent.futures
from z3 import *
from utils.logger import get_logger
from fuzzywuzzy import fuzz
from transformers import *
from commons.interfaces import ApiMatching
import re
import random

logger = get_logger('synthesizer.decider')


class TFDecider(Decider):

    def __init__(self, test_cases: List[TestCase], matching_apis: List[LibraryAPI],
                 interpreter: Interpreter, api_matcher: ApiMatching):
        super().__init__(test_cases)
        self.matching_apis = matching_apis
        self.interpreter = interpreter
        self.api_matcher = api_matcher

    def error_message_understanding(self, raw_error_message: List[str], program: Program) -> (
            Constraint, List[str]):
        return None, None

        for test in self.test_cases:
            # extract the arg values
            mutated_program = program
            idx = mutated_program.code[0].find('=') + 2
            code = mutated_program.code[0][idx:]
            regex = re.search('([^\(]*)\((.*)\)', code)
            api_name = regex.group(1)
            arg_values = regex.group(2).split(',')
            api = self.api_matcher.get_api(api_name)
            count = 0
            for arg, arg_value in zip(api.arguments, arg_values):
                if arg.type == 'int':
                    arg_values[count] = int(arg_value)
                count += 1

            # mutate the program
            n_trials = 10
            for i in range(n_trials):
                mutated_args, idx = self.mutate_args(arg_values, VarPool.get_preset_vals())
                mutated_program.code[0] = api_name + f"({','.join(list(map(str,mutated_args)))})"
                success, output = self.interpreter.tensor_forward_pass(mutated_program, test.input['tf'])
                if success or raw_error_message != output:
                    api.add_perm_constraint(idx, arg_values)
                    return None

        return None

    def mutate_args(self, arg_values, value_pool: VarPool):
        idx = random.randint(0, len(arg_values) - 1)
        arg_values[idx] = random.choice(value_pool['int'])
        return arg_values, idx

    def analyze(self, program: Program) -> TFResult:
        target_call = program.code[0]

        # try to create layer
        logger.debug(f'Evaluating... {target_call}')
        if target_call.find("stride=0") != -1 or target_call.find("stride=(0,0)") != -1:
            return TFResult(False)

        # test cases
        output = None
        for test in self.test_cases:
            success, output = self.interpreter.tensor_forward_pass(program, test.input['tf'])
            if not success:  # runtime error
                return TFResult(False, error_msg=output)
            elif sorted(test.output.shape) != sorted(output.shape):  # wrong shape
                return TFResult(False, [f'Wrong shape {test.output.shape} vs {output.shape}'])
            else:
                transformed_output = output[:]
                idx = []
                for i in range(len(test.output.shape)):
                    for j in range(len(test.output.shape)):
                        if test.output.shape[i] == output.shape[j] and j not in idx:
                            idx += [j]
                transformed_output = transformed_output.transpose(tuple(idx))
                if not np.allclose(test.output, transformed_output, rtol=1e-04, atol=1e-07):  # we don't know why it failed
                    return TFResult(False)
                program.after = ['{input}.' + f'permute({",".join(map(str,idx))})']

        return TFResult(True, output=output)
