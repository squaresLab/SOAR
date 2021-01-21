from tqdm import tqdm

from typing import List, Dict
from mapping.representations import get_representation
from autotesting.run_tests import test_synthesized_network_structure, \
  test_synthesized_forward_pass, load_example_by_name, generate_result_file
from synthesis.search_structure import VarPool, get_values_from_code, get_tryout_name, get_tryout_combinations, \
  SearchableSyntaxTree


def synthesize_structure(src_lines: List[str]):
  # TODO: now we assume there is only one line
  line = src_lines[0]
  print('Start migrating for api call: ' + line)
  src_api_name = line.split('=')[1].split('(')[0].strip()

  # first get the top k similar apis from the other library
  representation = get_representation()
  top_k_apis = representation.query_top_k(src_api_name)
  
  # then get the variable pool from the source library api call and preset values
  var_pool = VarPool.combine_pool(get_values_from_code(line), VarPool.get_preset_vals())
  
  # test for each of these apis
  for tgt_api in top_k_apis:
    print('trying out api ' + tgt_api.id)
    found_match = False
    
    # build a searchable tree
    tree = SearchableSyntaxTree()
    tree.build_tree(tgt_api, var_pool)
    
    # try out each of the var combinations
    all_var_combs = get_tryout_combinations(list(map(lambda node: len(node.value_candidates), tree.search_nodes)))
    for var_comb in tqdm(all_var_combs):
      
      linearized_code = 'self.a = ' + tree.linearize_tree(var_comb)
      # FIXME: hard code, fix this
      test_result = test_synthesized_network_structure([linearized_code], 'test_conv', trial_name=get_tryout_name(var_comb))
      if test_result:
        print('migration success: ')
        print('From: '+line)
        print('To: '+linearized_code)
        found_match = True
        break
        
    if found_match:
      break


def synthesize_forward_pass(src_lines: List[str], structure_code: List[str]):
  # TODO: now we only have simple one layer pass
  fp_stub_code = ['x = x.permute(0, 3, 1, 2)', 'x = self.a(x)', 'x = x.permute(0, 2, 3, 1)']
  test_synthesized_forward_pass(structure_code, fp_stub_code, 'test_conv', 'debug')
  pass


def synthesize_for_example(name: str) -> None:
  # load the structure and forward-pass lines for the example
  s_src_lines, fp_src_lines = load_example_by_name(name)
  
  # TODO: now we assume there is only one line in structure and we use a stub for the forward-pass
  s_line = s_src_lines[0]
  print('Start migrating for api call: ' + s_line)
  src_api_name = s_line.split('=')[1].split('(')[0].strip()
  
  # first get the top k similar apis from the other library
  representation = get_representation()
  top_k_apis = representation.query_top_k(src_api_name)
  
  # then get the variable pool from the source library api call and preset values
  var_pool = VarPool.combine_pool(get_values_from_code(s_line), VarPool.get_preset_vals())
  
  # test for each of these apis
  for tgt_api in top_k_apis:
    print('trying out api ' + tgt_api.id)
    found_match = False
    
    # build a searchable tree
    tree = SearchableSyntaxTree()
    tree.build_tree(tgt_api, var_pool)
    
    # try out each of the var combinations
    all_var_combs = get_tryout_combinations(list(map(lambda node: len(node.value_candidates), tree.search_nodes)))
    for var_comb in tqdm(all_var_combs):
      trial_name = get_tryout_name(var_comb)
      
      linearized_code = 'self.var'+trial_name+' = ' + tree.linearize_tree(var_comb)
      # FIXME: hard code, fix this
      test_result = test_synthesized_network_structure([linearized_code], name, trial_name)
      if test_result:
        # print('Code: '+linearized_code+' passed the structure test')
        
        # check if the forward-pass test is successful
        fp_stub_code = ['x = x.permute(0, 3, 1, 2)', 'x = self.var'+trial_name+'(x)', 'x = x.permute(0, 2, 3, 1)']
        fp_result = test_synthesized_forward_pass([linearized_code], fp_stub_code, name, trial_name)
        
        if fp_result:
          print('migration success: ')
          print('From: ' + s_line)
          print('To: ' + linearized_code)
          
          generate_result_file([linearized_code], fp_stub_code, name)
          found_match = True
          break
    
    if found_match:
      break
  pass



if __name__ == '__main__':
  synthesize_for_example('test_conv')
  # synthesize_structure(['self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, use_bias=True)'])

  # result = get_tryout_combinations([3,4,5])
  # print('')
  
  # result = get_values_from_code('self.conv1 = Conv2D(32, 3, strides=2, use_bias=True)')
  # result = VarPool.combine_pool(result, VarPool.get_preset_vals())
  # print('')
  
  # with open('../crawler/preprocessed_torch_docs.json') as f:
  #   apis = json.load(f)
  #   library = Library.from_json_list(apis)
  #   apis = library.apis
  #
  #   tree = SearchableSyntaxTree()
  #   tree.build_tree(apis[92], get_default_vars())
  #
  #   s = tree.linearize_tree([0]*len(tree.search_nodes))
  #   print(s)
