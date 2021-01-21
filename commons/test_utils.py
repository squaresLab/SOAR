import tensorflow as tf
import numpy as np
import torch
import os
import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from typing import Dict
from commons.library_api import LibraryAPI
from itertools import permutations
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
from concurrent.futures import TimeoutError
from commons.synthesis_program import TorchProgram, Program
from typing import List
from multiprocessing.pool import Pool


class Interpreter:

    def __init__(self, constant_init: float = 0.0001):
        self.executor = Pool(maxtasksperchild=1)
        self.constant_init = constant_init
        self.jobs = []

    def tf_forward_pass(self, tf_layer, input_tensor: np.ndarray):
        try:
            self.tf_init_layer(tf_layer)
            tf_input = self.np_tensor_to_tf(input_tensor)
            tf_output = tf_layer(tf_input)
            tf_result = self.tf_tensor_to_np(tf_output)
            return True, tf_result
        except Exception as e:
            return False, [str(e)]

    def torch_fd_pass(self, torch_layer, input_tensor: np.ndarray):
        try:
            self.torch_init_layer(torch_layer, self.constant_init)
            torch_input = self.np_tensor_to_torch(input_tensor)
            torch_output = torch_layer(torch_input)
            torch_result = self.torch_tensor_to_np(torch_output)
            return True, torch_result
        except Exception as e:
            return False, [str(e)]

    def tf_init_layer(self, tf_layer):
        # Normal Layers
        try:
            tf_layer.kernel_initializer = tf.initializers.Constant(self.constant_init)
            tf_layer.bias_initializer = tf.initializers.Constant(self.constant_init)
        except:
            pass

        # Embeddings
        try:
            tf_layer.embeddings_initializer = tf.initializers.Constant(self.constant_init)
        except:
            pass

    def torch_forward_pass(self, program: TorchProgram, input_tensor: np.ndarray):
        try:
            code = program.linearize('input_tensor')
            out = Interpreter.torch_forward_pass_aux(code,
                                       self.constant_init, input_tensor)
            return True, out
        except BrokenProcessPool as e:
            self.executor = concurrent.futures.ProcessPoolExecutor()
            return False, [str(e)]
        except Exception as e:
            return False, [str(e)]

    def tensor_forward_pass(self, program: Program, input_tensor: np.ndarray):
        try:
            code = program.code
            if program.code[0].find("2D") != -1:
                input_tensor = input_tensor.transpose(0, 2, 3, 1)
            # out = Interpreter.tensor_forward_pass_aux(code, self.constant_init, input_tensor)
            out = self.executor.apply_async(Interpreter.tensor_forward_pass_aux, args=[code, self.constant_init, input_tensor]).get(timeout=1)
            return True, out
        except BrokenProcessPool as e:
            self.executor = concurrent.futures.ProcessPoolExecutor()
            return False, [str(e)]
        except Exception as e:
            return False, [str(e)]

    @staticmethod
    def create_layer(api: LibraryAPI, args: Dict[str, object]):
        # Alias
        return Interpreter.execute_api_call(api, args)

    @staticmethod
    def create_layer_torch(layer: str):
        # Alias
        return eval(layer)

    @staticmethod
    def create_layer_tf(layer: str):
        # Alias
        return eval(layer)


    @staticmethod
    def tf_init_layer_static(tf_layer, constant_init):
        # Normal Layers
        try:
            tf_layer.kernel_initializer = tf.initializers.Constant(constant_init)
            tf_layer.bias_initializer = tf.initializers.Constant(constant_init)
        except:
            pass

        # Embeddings
        try:
            tf_layer.embeddings_initializer = tf.initializers.Constant(constant_init)
        except:
            pass

    @staticmethod
    def torch_init_layer(torch_layer, constant_init):
        params = []
        try:
            params = torch_layer.parameters()
        except:
            pass
        for param in params:
            try:
                torch.nn.init.constant_(param.data, constant_init)
            except Exception:
                pass

    @staticmethod
    def torch_forward_pass_aux(code: List[str], constant_init, input_tensor: np.ndarray):
        torch_layer = eval(code[0])
        Interpreter.torch_init_layer(torch_layer, constant_init)
        input_tensor = Interpreter.np_tensor_to_torch(input_tensor)
        for line in code[1:]:
            input_tensor = eval(line)
        return Interpreter.torch_tensor_to_np(input_tensor)

    @staticmethod
    def tensor_forward_pass_aux(code: List[str], constant_init, input_tensor: np.ndarray):
        tf_layer = eval(code[0][code[0].find('=')+1:].strip())
        Interpreter.tf_init_layer_static(tf_layer, constant_init)
        input_tensor = Interpreter.np_tensor_to_tf(input_tensor)
        input_tensor = tf_layer(input_tensor)
        return Interpreter.tf_tensor_to_np(input_tensor)

    @staticmethod
    def execute_api_call(api: LibraryAPI, args: Dict[str, object]) -> (bool, object):
        """
        execute the api call with the filled in arguments to get its expected output
        :param api: the api call to be called
        :param args: a dictionary of key (argument name) and value (instantiated argument value)
        :return: bool (executable or not) and object (return values)
        """

        # construct the api call with string concat
        api_call_str = Interpreter.create_api_call(api, args)

        try:
            exec_result = eval(api_call_str)
            return True, exec_result
        except Exception as e:
            return False, str(e)

    @staticmethod
    def create_api_call(api, args):
        if args != dict():
            api_call_str = api.id + '('
            for arg in api.arguments:
                if arg.name.find('kwargs') != -1: continue
                if arg.is_optional and arg.default_value is None: continue
                api_call_str += f'{arg.name}={args.get(arg.name, arg.default_value)}, '
            api_call_str = api_call_str[:-2] + ')'
        else:
            api_call_str = api.id + '()'
        return api_call_str

    @staticmethod
    def execute_api_call_no_args(api: LibraryAPI, input):
        try:
            tf_input = Interpreter.np_tensor_to_tf(input)
            exec_result = eval(api.id + '(tf_input)')
            return True, exec_result.numpy()
        except Exception as e:
            return False, str(e)

    @staticmethod
    def np_tensor_to_tf(tensor: np.ndarray) -> tf.Tensor:
        return tf.convert_to_tensor(tensor)

    @staticmethod
    def tf_tensor_to_np(tensor: tf.Tensor) -> np.ndarray:
        return tensor.numpy()

    @staticmethod
    def np_tensor_to_torch(tensor: np.ndarray) -> torch.Tensor:
        return torch.Tensor(tensor)

    @staticmethod
    def np_tensor_to_longtorch(tensor: np.ndarray) -> torch.Tensor:
        return torch.LongTensor(tensor)

    @staticmethod
    def torch_tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().numpy()


def extract_api_arguments(api: LibraryAPI, api_call: str) -> Dict[str, str]:
    # all type of brackets
    round_ctr = 0
    square_ctr = 0
    curly_ctr = 0

    raw_args = api_call[api_call.find(api.id) + len(api.id) + 1:-1]
    args = []
    partial_result = ''
    for i in range(len(raw_args)):
        c = raw_args[i]
        partial_result += c
        if c == '(':
            round_ctr += 1
        elif c == ')':
            round_ctr -= 1
        elif c == '[':
            square_ctr += 1
        elif c == ']':
            square_ctr -= 1
        elif c == '{':
            curly_ctr += 1
        elif c == '}':
            curly_ctr -= 1
        elif c == ',':
            if round_ctr == 0 and square_ctr == 0 and curly_ctr == 0:
                args.append(partial_result[:-1])
                partial_result = ''

        if i == len(raw_args) - 1:
            args.append(partial_result)

    arg_val_dict = dict()
    for arg_val, api_arg in zip(args, api.arguments):
        arg_val = arg_val.strip()
        if not api_arg.is_optional and arg_val.find('=') == -1:
            arg_val_dict[api_arg.name] = arg_val
        elif arg_val.split('=')[0] == api_arg.name:
            arg_val_dict[api_arg.name] = arg_val[arg_val.find('=') + 1:]

    return arg_val_dict


def extract_api_arguments_torch(api: LibraryAPI, api_call: str) -> Dict[str, str]:
    # all type of brackets
    arg_val_dict = dict()
    inside_brackets = 0
    args = []
    concat = ''
    for c in api_call:
        if inside_brackets > 0:
            concat += c
        if c == ',' and inside_brackets == 1:
            args += [concat[:-1]]
            concat = ''
            continue
        elif c == '(' or c == '[':
            inside_brackets += 1
        elif c == ')' or c == ']':
            inside_brackets -= 1
    args += [concat[:-1]]
    if args == ['']:
        return arg_val_dict
    for arg_val, api_arg in zip(args, api.arguments):
        if not api_arg.is_optional:
            arg_val_dict[api_arg.name] = arg_val
        else:
            # if arg_val.split('=')[0] != api_arg.name:
            #    assert (arg_val.split('=')[0] == api_arg.name)
            arg_val_dict[arg_val.split('=')[0]] = arg_val.split('=')[1]

    return arg_val_dict


def code_to_params(code_string):
    # all type of brackets
    square_ctr = 0
    round_ctr = 0
    code_params = code_string
    if '(' in code_string:
        code_params = code_string.split('(', 1)[1]
    split_code = list(code_params)
    if split_code[len(split_code) - 1] == ')':
        split_code.pop(len(split_code) - 1)
    for i, char in enumerate(split_code):
        if char == '[':
            square_ctr += 1
        elif char == ']':
            square_ctr -= 1
        elif char == '(':
            round_ctr += 1
        elif char == ')':
            round_ctr -= 1
        elif char == ',':
            if square_ctr != 0 or round_ctr != 0:
                # if not 0 then we are inside a bracket, replace with ... instead
                split_code[i] = '{.!.}'
    s = ''
    return_str = s.join(split_code)
    return_list = return_str.split(',')
    for i, char in enumerate(return_list):
        char = char.replace('{.!.}', ',')
        return_list[i] = char
    return return_list


if __name__ == '__main__':
    result = extract_api_arguments('tf.fill',
                                   'output = tf.fill(tf.convert_to_tensor(np.random.randint(1, 6, size=(3))),tf.convert_to_tensor(np.random.rand(3, 4, 5, 6)),name=None) ')
    print(result)
