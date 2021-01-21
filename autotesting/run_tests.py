import unittest
import py_compile
import io
import os
import tensorflow as tf

from os import listdir
from os.path import isfile, join
from os import walk
from typing import List
import json

# suppress all the tensorflow warning and error information for simpler output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def find_mark_in_lines(mark_str: str, src_lines: List[str]) -> (int, str):
  # find the insert mark and spaces
  mark_idx = -1
  for i in range(len(src_lines)):
    if mark_str in src_lines[i]:
      mark_idx = i
      break
  indent = src_lines[mark_idx].split('[')[0]
  
  return mark_idx, indent


def insert_lines_of_code(src_lines: List[str], insert_lines: List[str], mark_str: str, offset=2):
  # find the insert mark and spaces
  mark_idx, indent = find_mark_in_lines(mark_str, src_lines)
  
  lines = map(lambda line: indent + line, insert_lines)
  code_chunk = '\n'.join(lines) + '\n'
  src_lines.insert(mark_idx + offset, code_chunk)


def insert_structure_code_to_test(lines: List[str], test_file_path: str, trial_name: str) -> None:
  # first read in all the code lines
  f = open(test_file_path, 'r')
  contents = f.readlines()
  f.close()
  
  # insert the structure code
  insert_lines_of_code(contents, lines, '[GENERATED STRUCTURE CODE STARTS HERE]')
  
  # then save the test file with inserted code
  dir_split = list(os.path.split(test_file_path))
  dir_split.insert(-1, 'tmp')
  dir_split[-1] = '_' + dir_split[-1][:-3] + '_s_' + trial_name + '.py'
  new_test_file_path = os.path.join(*dir_split)
  
  f = open(new_test_file_path, 'w')
  contents = ''.join(contents)
  f.write(contents)
  f.flush()
  f.close()


def insert_forward_pass_code_to_test(structure_lines: List[str], forward_pass_lines: List[str]
                                     , test_file_path: str, trial_name: str) -> None:
  # first read in all the code lines
  f = open(test_file_path, 'r')
  contents = f.readlines()
  f.close()
  
  # insert the structure code and forward pass code
  insert_lines_of_code(contents, structure_lines, '[GENERATED STRUCTURE CODE STARTS HERE]')
  insert_lines_of_code(contents, forward_pass_lines, '[GENERATED FORWARD-PASS CODE STARTS HERE]')

  # then save the test file with inserted code
  dir_split = list(os.path.split(test_file_path))
  dir_split.insert(-1, 'tmp')
  dir_split[-1] = '_' + dir_split[-1][:-3] + '_fp_' + trial_name + '.py'
  new_test_file_path = os.path.join(*dir_split)
  
  f = open(new_test_file_path, 'w')
  contents = ''.join(contents)
  f.write(contents)
  f.flush()
  f.close()


def test_synthesized_forward_pass(structure_lines: List[str], forward_pass_lines: List[str],
                                  test_file: str, trial_name: str, no_print: bool=True) -> bool:
  
  # first insert the synthesized code
  test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testcases', test_file + '.py')
  insert_forward_pass_code_to_test(structure_lines, forward_pass_lines, test_file_path, trial_name)
  
  # then get the test cases from file
  suite = unittest.TestLoader().loadTestsFromName(
    'autotesting.testcases.tmp._'+test_file+'_fp_'+trial_name+'.TestConsistency.test_forward_pass')
  f = io.StringIO(initial_value='')
  runner = unittest.TextTestRunner(stream=f, verbosity=0)
  result = runner.run(suite)
  
  # capture the output of the stack-trace
  f.flush()
  output = f.getvalue()
  if not no_print:
    print("###################ERROR INFO STARTS#####################")
    print(output)
    print("###################ERROR INFO ENDS#####################")
  f.close()
  
  # see if every test passes
  if len(result.errors) + len(result.failures) + len(result.skipped) == 0:
    if not no_print:
      print("all test cases passed!!!")
    return True
  else:
    if not no_print:
      print("at least one test failed")
    return False


def test_synthesized_network_structure(structure_lines: List[str],
                                       test_file: str, trial_name: str, no_print: bool = True) -> bool:
  # first insert the synthesized code
  test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testcases', test_file + '.py')
  insert_structure_code_to_test(structure_lines, test_file_path, trial_name)
  
  # try to compile the test file
  compile_result = py_compile.compile(test_file_path)
  if compile_result is None:
    return False
  
  # then get the test cases from file
  suite = unittest.TestLoader().loadTestsFromName(
    'autotesting.testcases.tmp._' + test_file + '_s_' + trial_name + '.TestConsistency.test_structure')
  f = io.StringIO(initial_value='')
  runner = unittest.TextTestRunner(stream=f, verbosity=0)
  result = runner.run(suite)
  
  # capture the output of the stack-trace
  f.flush()
  output = f.getvalue()
  if not no_print:
    print("###################ERROR INFO STARTS#####################")
    print(output)
    print("###################ERROR INFO ENDS#####################")
  f.close()
  
  # see if every test passes
  if len(result.errors) + len(result.failures) + len(result.skipped) == 0:
    if not no_print:
      print("all test cases passed!!!")
    return True
  else:
    if not no_print:
      print("at least one test failed")
    return False
  
  
def load_all_examples() -> List[str]:
  test_cases_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testcases')
  test_files = []
  for (_, _, filenames) in walk(test_cases_dir):
    test_files.extend(filenames)
    break
  test_names = list(map(lambda x: x.split('.')[0], test_files))
  return test_names


def trim_lines(src_lines: List[str]) -> List[str]:
  result = []
  for line in src_lines:
    line = line.strip()
    if line != '':
      result.append(line)
      
  return result

def load_ground_truth(name: str):
  import ast
  test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testcases', name+'.py')
  f = open(test_file_path, 'r')
  data = ast.literal_eval(f.read())
  f.close()
  return data

def load_example_by_name(name: str) -> (List[str], List[str]):
  # first read in all the code lines
  test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testcases', name+'.py')
  f = open(test_file_path, 'r')
  src_lines = f.readlines()
  f.close()

  # find the src content for the structure
  s_start_idx, _ = find_mark_in_lines('[SOURCE STRUCTURE CODE STARTS HERE]', src_lines)
  s_end_idx, _ = find_mark_in_lines('[SOURCE STRUCTURE CODE ENDS HERE]', src_lines)

  # find the src content for the forward pass
  fp_start_idx, _ = find_mark_in_lines('[SOURCE FORWARD-PASS CODE STARTS HERE]', src_lines)
  fp_end_idx, _ = find_mark_in_lines('[SOURCE FORWARD-PASS CODE ENDS HERE]', src_lines)

  src_s_code = trim_lines(src_lines[s_start_idx+1:s_end_idx])
  src_fp_code = trim_lines(src_lines[fp_start_idx+1:fp_end_idx])

  return src_s_code, src_fp_code


def generate_result_file(structure_lines: List[str], forward_pass_lines: List[str],
                         test_name: str) -> None:
  # delete all the tmp files
  tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testcases', 'tmp')
  test_files = []
  for (_, _, filenames) in walk(tmp_dir):
    test_files.extend(filenames)
    break
    
  for tmp_file in test_files:
    if tmp_file.startswith('_'+test_name):
      os.remove(os.path.join(tmp_dir, tmp_file))
  
  # first read in all the code lines
  test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testcases', test_name+'.py')
  f = open(test_file_path, 'r')
  contents = f.readlines()
  f.close()
  
  # insert the structure code and forward pass code
  insert_lines_of_code(contents, structure_lines, '[GENERATED STRUCTURE CODE STARTS HERE]')
  insert_lines_of_code(contents, forward_pass_lines, '[GENERATED FORWARD-PASS CODE STARTS HERE]')
  
  # then save the test file with inserted code
  dir_split = list(os.path.split(test_file_path))
  dir_split.insert(-1, 'tmp')
  dir_split[-1] = dir_split[-1][:-3] + '_sol.py'
  new_test_file_path = os.path.join(*dir_split)
  
  f = open(new_test_file_path, 'w')
  contents = ''.join(contents)
  f.write(contents)
  f.flush()
  f.close()


def print_curdir():
  print(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testcases'))


if __name__ == '__main__':
  print(load_all_examples())
  
  # conv_test_file_name = 'test_conv'
  #
  # ground_truth_structure_code = ['self.conv1 = nn.Conv2d(1, 32, 3, stride=2, bias=True)']
  # test_synthesized_network_structure(ground_truth_structure_code, conv_test_file_name, trial_name='debug_test')

  # ground_truth_forward_pass_code = ['x = x.permute(0, 3, 1, 2)', 'x = self.conv1(x)', 'x = x.permute(0, 2, 3, 1)']
  # test_synthesized_forward_pass(ground_truth_structure_code, ground_truth_forward_pass_code, conv_test_file_name, trial_name='debug_test')
