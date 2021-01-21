import multiprocessing
import pickle
import time
import os
import pathlib
import ast

# have to have these to ensure that libraries are loaded
import tensorflow as tf
import numpy as np
import torch

from concurrent.futures import ProcessPoolExecutor as Pool
from typing import List, Dict, Tuple
from tqdm import tqdm
from commons.library_api import Library, LibraryAPI, load_library, load_apis
from commons.test_utils import extract_api_arguments, Interpreter
from mapping.representations import get_representation
from autotesting.run_tests import test_synthesized_network_structure, \
  test_synthesized_forward_pass, load_example_by_name, generate_result_file, load_ground_truth
from synthesis.search_structure import VarPool, get_values_from_code, get_tryout_name, get_tryout_combinations, \
  SearchableSyntaxTree
from commons.synthesis_program import TestCase
from synthesis.testcases_analysis import package_success_rate_analysis
execute_api_call = Interpreter.execute_api_call
# disable using GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def get_test_data_path(library_name: str):
  return pathlib.Path(__file__).parent.absolute() \
    .joinpath('{}_single_api_tests.bin'.format(library_name))


class NoDaemonProcess(multiprocessing.Process):
  """ this would allow creating more processes inside a child process """
  def _get_daemon(self):
    return False
  
  def _set_daemon(self, value):
    pass
  
  daemon = property(_get_daemon, _set_daemon)
  
  
class MyPool(multiprocessing.pool.Pool):
  """ A wrapper around previous NoDaemonProcess """
  Process = NoDaemonProcess


def try_executing_code(api: LibraryAPI, code: str, result_list: List[Tuple[bool, TestCase]]):
  """ save results in the result_list, first str in tuple is the code, second str is the error msg (can be '') """
  api_arguments = extract_api_arguments(api, code)

  # instantiate all the arguments (e.g. np.random.rand() -> 0.4893)
  instantiated_args = dict()
  for k in api_arguments.keys():
    instantiated_args[k] = eval(api_arguments[k])

  # execute the api call to get expected output
  success, exec_output = execute_api_call(api, instantiated_args)

  test_case = TestCase(instantiated_args, exec_output, api)
  
  # FIXME discard the result because bug may occur when trying to pass objects between processes
  if success:
    test_case.output = None
  result_list.append((success, test_case))


def collect_valid_tests(api: LibraryAPI) -> (str, List[Tuple[bool, TestCase]]):
  """generating the I/O tests"""
  input_values = VarPool.get_preset_pool()
  if api.library == 'tf':
    input_values.wrap_for_tf_tensor()
  else:
    input_values.wrap_for_torch_tensor()
  
  # build a searchable tree
  tree = SearchableSyntaxTree()
  tree.build_tree(api, input_values)

  # set return list for multiprocessing
  manager = multiprocessing.Manager()
  return_list = manager.list()

  # try out each of the var combinations
  all_var_combs = get_tryout_combinations(list(map(lambda node: len(node.value_candidates), tree.search_nodes)))
  start_time = time.time()
  # TODO: now we only try the first 1,000 combinations
  for var_comb in tqdm(all_var_combs[:1000]):
    linearized_code = 'output = ' + tree.linearize_tree(var_comb)
  
    p = multiprocessing.Process(target=try_executing_code, args=(api, linearized_code, return_list))
    p.start()
    p.join(timeout=2)
    p.terminate()
    
    if time.time() - start_time > 10:
      break
    
  print('{0} got {1} tests'.format(api.id, len(return_list)))
    
  return api.id, list(return_list)


def test_generation(library: Library, chunk=200) -> None:
  """ dumps a dictionary. Keys are api.id and value is a list of tuple of two strings, first string is the code that
   runs and the second string is the error message. if second is empty string, it means the code is executable """
  apis = library.apis
  
  result_list: List[Tuple[str, List[Tuple[bool, TestCase]]]] = []
  start = 0
  
  # FIXME: can't figure out why workers in the pools dies over time, use this alternative method instead
  while start < len(apis):
    end = min(len(apis), start + chunk)
    pool = Pool()
    result = pool.map(collect_valid_tests, apis[start:end])
    
    result_list.extend(result)
    start = end
    print('######## {0}/{1} Done, list length {2}'.format(end, len(apis), len(result_list)))
    
  # print out the summary
  avg_tries = sum(list(map(lambda x: len(x[1]), result_list))) / len(result_list)
  num_at_least_one_test = len(list(filter(lambda x: any(y[0] is True for y in x[1]), result_list)))
  print('Auto test generation done:')
  print('For {0} examples, {1} avg. tries and {2} have at least one test generated.'
        .format(len(result_list), avg_tries, num_at_least_one_test))

  # dump it into a file
  result_dict: Dict[str, List[Tuple[bool, TestCase]]] = dict(result_list)
  pickle.dump(result_dict, open(get_test_data_path(library.name), 'wb'))
  
  
def load_all_single_api_tests(library_name: str) -> Dict[str, List[TestCase]]:
  saved_tests_dict = pickle.load(open(get_test_data_path(library_name), 'rb'))
  tests = dict(map(lambda x: (x[0], list(map(lambda z: z[1],
                                         list(filter(lambda y: y[0] is True, x[1]))))), saved_tests_dict.items()))
  return tests


def load_all_single_api_errors(library_name: str) -> Dict[str, List[TestCase]]:
  saved_tests_dict = pickle.load(open(get_test_data_path(library_name), 'rb'))
  errors = dict(map(lambda x: (x[0], list(map(lambda z: z[1],
                                         list(filter(lambda y: y[0] is False, x[1]))))), saved_tests_dict.items()))
  return errors


if __name__ == '__main__':
  saved_tests_dict = load_all_single_api_errors('tf')
  
  library = load_library('tf', basic_info=True)

  test_generation(library)
