import os
import pickle

from typing import List, Set, Dict
from commons.library_api import load_apis
from tqdm import tqdm

def get_python_files_from_dir(rootdir: str):
  result = []
  for subdir, dirs, files in os.walk(rootdir):
    for file in files:
      if str(file).endswith('.py'):
        result.append(os.path.join(subdir, file))
        
  return result
        
        
def get_api_usage_from_loc(lines_of_code: List[str], tf_api_names: Set[str],
                                   api_freq_dict: Dict[str, int]) -> None:
  # TODO: deal with import with different names (e.g. import xxx.yy as yy)
  # search for the keyword tf or tensorflow
  for line in lines_of_code:
    subline = line
    while subline.find('tf') != -1:
      api_starts = subline.find('tf')
      api_ends = subline[api_starts:].find('(')
      
      if api_ends == -1:
        break
        
      api_str = None
      potential_api_str = subline[api_starts:][:api_ends]
      compat_v1_api_str = 'tf.compat.v1' + potential_api_str[2:]
      # TODO: remove this once we have the alias information
      math_api_str = 'tf.math' + potential_api_str[2:]
      
      if potential_api_str in tf_api_names:
        api_str = potential_api_str
      elif compat_v1_api_str in tf_api_names:
        api_str = compat_v1_api_str
      elif math_api_str in tf_api_names:
        api_str = math_api_str
        
      if api_str is not None:
        if api_str in api_freq_dict:
          api_freq_dict[api_str] += 1
        else:
          api_freq_dict[api_str] = 1
        
      subline = subline[api_starts+2:]
      
      
def save_api_freq_dict(freq_dict: Dict[str, int], file_name: str):
  pickle.dump(freq_dict, open(file_name, 'wb'))
      
      
def get_api_usages_from_files_in_dir(dir_path: str):
  result_dict = dict()
  
  # load all the tensorflow apis to know what are valid api by name
  apis = load_apis('tf')
  api_names_set = set(map(lambda x: x.id, apis))
  
  # find all the python files in the local dir and start checking code line by line
  file_paths = get_python_files_from_dir(dir_path)
  
  for file in tqdm(file_paths):
    with open(file, 'r') as f:
      lines = f.readlines()
      get_api_usage_from_loc(lines, api_names_set, result_dict)
    
  # print the result
  sorted_items = sorted(result_dict.items(), key=lambda item: item[1], reverse=True)
  list(map(print, sorted_items))
  
  # dump this into a pickle file
  pickle.dump(result_dict, open('api_freq_dict.bin', 'wb'))

  
if __name__ == '__main__':
  rootdir = '../../models'
  get_api_usages_from_files_in_dir(rootdir)
