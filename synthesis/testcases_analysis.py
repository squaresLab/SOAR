import pickle
import numpy as np
import tensorflow as tf
import torch

from functools import reduce
from tqdm import tqdm
from typing import Dict, List
from synthesis.plot import plot_packages, pie_plot
from commons.library_api import load_apis

output = None


def failure_case_analysis(filename: str):
  api_tests_dict: Dict[str, List[str]] = pickle.load(open(filename, 'rb'))
  apis = load_apis('tf')
  
  failed_api_names = set(map(lambda k, v: k, filter(lambda k, v: len(v) > 0, api_tests_dict.items())))
  failed_apis = list(filter(lambda x: x.id in failed_api_names, apis))
  np.random.shuffle(failed_apis)
  
  pass
  

def output_type_analysis(filename: str):
  api_tests_dict: Dict[str, List[str]] = pickle.load(open(filename, 'rb'))
  
  type_dict = dict()
  
  for api_name, executable_tests in tqdm(api_tests_dict.items()):
    for test in executable_tests:
      try:
        exec('global output; ' + test)
        if output is None:
          output_type = 'NoneType'
        else:
          output_type = str(type(output))#.split('\'')[1]
      except:
        # TODO: find out why sometimes the test fails here
        continue
      
      if output_type in type_dict:
        type_dict[output_type] += 1.0 / len(executable_tests)
      else:
        type_dict[output_type] = 1.0 / len(executable_tests)
        
  # show as pie plot
  pie_plot(list(type_dict.values()), list(type_dict.keys()), show_percentage=True)


def package_success_rate_analysis(api_tests_dict: Dict[str, List[str]], api_freq_dict: Dict[str, int]=None) -> None:
  '''
  paint a nested pie chart for the success rate on package and sub-package levels
  
  :param api_tests_dict: api name and the list of successful generated tests
  :param api_freq_dict: api freq to weight the api_tests_dict
  '''
  
  # the data structure here FOR THE RESULT is Dict[str, Dict[str, List[int]]]
  # first 'str' -> package name; second 'str' -> subpackage name;
  # List[int] -> actually two integers here: total number and success number
  api_pkg_name_dict: Dict[str, Dict[str, List[int]]] = dict()
  
  # uniformly weighted if the freq dict is not given
  if api_freq_dict is None:
    api_freq_dict = dict(map(lambda x: (x, 1), api_tests_dict.keys()))
  total_usage_count = sum(api_freq_dict.values())
  
  # count for each package and sub-package
  for k, v in api_tests_dict.items():
    weight = api_freq_dict.get(k, -1) * 100 / total_usage_count
    if weight == -1:
      continue
    
    name_split = k.split('.')
    
    # TODO: temporarily solution, find out why this happens
    if len(name_split) < 2:
      continue
    
    pkg_name = name_split[1]
    sub_pkg_name = '.' if len(name_split) <= 2 else name_split[2]
    # accommondate for tensorflow
    if pkg_name == 'compat' and sub_pkg_name == 'v1':
      pkg_name = 'compat.v1'
      sub_pkg_name = name_split[3]
    
    auto_gen_success = weight if len(v) > 0 else 0
    
    if pkg_name in api_pkg_name_dict:
      sub_pkg_name_dict = api_pkg_name_dict[pkg_name]
      if sub_pkg_name in sub_pkg_name_dict:
        sub_pkg_name_dict[sub_pkg_name][0] += weight
        sub_pkg_name_dict[sub_pkg_name][1] += auto_gen_success
      else:
        sub_pkg_name_dict[sub_pkg_name] = [weight, auto_gen_success]
    else:
      sub_pkg_name_dict = {sub_pkg_name: [weight, auto_gen_success]}
      api_pkg_name_dict[pkg_name] = sub_pkg_name_dict
  
  # get the information in 2-D lists for plot
  package_sizes = []
  large_group_names = []
  small_group_names = []
  package_success_rate = []
  
  for k, v in api_pkg_name_dict.items():
    large_group_names.append(k)
    small_group_names.append(list(v.keys()))
    package_sizes.append(list(map(lambda x: x[0], v.values())))
    package_success_rate.append(list(map(lambda x: x[1] / x[0], v.values())))
  
  plot_packages(package_sizes, large_group_names, small_group_names, package_success_rate)

if __name__ == '__main__':
  output = 5
  exec('output = 6')
  print(output)
