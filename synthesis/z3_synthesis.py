from synthesis.network_synthesizer import *
from synthesis.search_structure import *
from z3 import *
from commons.z3_utils import *

class Z3Tree:

  def __init__(self):
    self.root: Node = None
    self.search_nodes: List[Node] = []
    self.api: LibraryAPI = None
    self.relevant_vars = []
    self.relevant_types = []
    self.relevant_names = []
    self.template = None

  def build_tree(self,  api: LibraryAPI, variables: VarPool):
    self.api = api
    self.root = Node(api.id)
    api.knowledge_base.push()
    for i in range(len(api.arguments)):
      # we dont enumerate optional vars
      if api.arguments[i].is_optional and api.arguments[i].name != "stride":
        continue


      arg_type = api.arguments[i].type
      values = variables[arg_type]
      ctr = []

      self.relevant_vars += [api.argument_vars[i]]
      self.relevant_names += [api.arguments[i].name]
      self.relevant_types += [arg_type]
      for value in values:
        if arg_type == 'int':
          ctr.append(api.argument_vars[i] == eval(value))
        elif arg_type == 'bool':
          mapping = {False: 0, True: 1}
          ctr.append(api.argument_vars[i] == mapping[eval(value)])
        elif arg_type == 'string':
          ctr.append(api.argument_vars[i] == StringVal(value))
      api.knowledge_base.append(Or(ctr))

      # hack to understand the impact of the error understanding model.
      # we block arguments with negative values
      # start
      if arg_type == 'int':
        api.knowledge_base.append(api.argument_vars[i] >= 0)
      # end

    self.template = blocking_template(self.relevant_vars, self.relevant_types)

  def enumerate(self):
    model = self.api.knowledge_base.model()
    values = [model[v] for v in self.relevant_vars]
    block_model(self.api.knowledge_base, self.template, values)


    #now we have a model with the args, we just have to build the actual program
    function = self.root.value + '({})'
    arguments = []
    for i in range(len(self.relevant_vars)):
      if self.relevant_names[i] == 'stride':
        arguments += [f'{self.relevant_names[i]} = {model[self.relevant_vars[i]]}']
      else:
        arguments += [f'{model[self.relevant_vars[i]]}']


    for i in range(len(self.api.arguments)):
      if self.api.arguments[i].is_optional and self.api.arguments[i].name != 'stride':
        arguments += [f'{self.api.arguments[i].name} = {self.api.arguments[i].default_value}']

    function = function.format(','.join(arguments))

    return function

  def more(self):
    return self.api.knowledge_base.check() == sat

  def empty(self):
    return False


def z3_synthesize_for_example(name: str) -> None:
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
    tree = Z3Tree()
    tree.build_tree(tgt_api, var_pool)

    # try out each of the var combinations

    i = -1
    while tree.more():
      i = i + 1
      trial_name = str(i)

      linearized_code = 'self.var' + trial_name + ' = ' + tree.enumerate()
      # FIXME: hard code, fix this
      test_result = test_synthesized_network_structure([linearized_code], name, trial_name)
      if test_result:
        # print('Code: '+linearized_code+' passed the structure test')

        # check if the forward-pass test is successful
        fp_stub_code = ['x = x.permute(0, 3, 1, 2)', 'x = self.var' + trial_name + '(x)', 'x = x.permute(0, 2, 3, 1)']
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
  z3_synthesize_for_example('test_conv')
