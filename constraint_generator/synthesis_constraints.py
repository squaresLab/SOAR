from z3 import *
import re
import numpy as np
import pickle
import autotesting
from autotesting.auto_test_generation.single_api_test_generation import load_all_single_api_errors

def block_model(z3_solver, model, vars):
    # m = self.z3_solver.model()
    block = []
    # block the model using only the variables that correspond to productions
    for x in vars:
        block.append(x == model[x])
    ctr = And(block)
    z3_solver.add(Not(ctr))

knowledge_base = Solver()


# suppose these are the arguments of function:   foo(x,y)
x = Int('x')
y = Int('y')
vars = [x, y]

# we know that:
#   x should be between 10 and 15,
#   y should be constrained to either 1 or 20
def generate_ctr(api, input, err_msg):
    ctr_list = []

    ## make ctr's based on some err_msg NLP algorithm
    if 'dimension' in err_msg:
        if 'negative dimension' in err_msg:
            params = err_msg[err_msg.find("[") + 1:err_msg.find("]")]
            err_words = err_msg.split(" ")
            for word in err_words:
                if ':' in word and 'Error' not in word:
                    target_arg = int(word.replace(':', ''))
            bracket_words = np.array(params.replace(' ', '').split(',')).astype(int)
            result = np.where(bracket_words == target_arg)[0][0]
            ctr1 = 'arg' + str(result) + ' >= 0'
    elif 'negative integer power' in err_msg:
        ctr1 = 'arg2 >= 0'
    elif 'Expected' in err_msg and '>= 0' in err_msg:
        ctr1 = 'arg1'
        params = err_msg[err_msg.find("[") + 1:err_msg.find("]")]

    print(err_msg)
    print('z3: "' + ctr1 + '"')

    ctr1 = And(x > 10, x < 15)
    ctr2 = Or(y == 1, y == 20)

    ctr_list.append(ctr1)
    ctr_list.append(ctr2)
    return ctr_list

def append_knowledge(api, input, err_msg):
    # err_msg = 'RuntimeError: Trying to create tensor with negative dimension -4: [20, 32, 21, -4]'
    ctr_list = generate_ctr(api, input, err_msg)
    # we add this to the knowledge base
    for i in ctr_list:
        knowledge_base.append(ctr_list[i])

    # now to enumerate the possible arguments we do:
    while knowledge_base.check() == sat:
        model = knowledge_base.model()
        block_model(knowledge_base, model, vars)
        print(model)

if __name__ == '__main__':

    api_calls = load_all_single_api_errors('tf')
    for api, output in api_calls.items():
        for error in output:
            if 'tf.roll' in str(error.api):
                print(
                    'call: ' + str(error.api) + ' >>> input: ' + str(error.input) + ' >>> error: ' + str(error.output))
            # print('call: ' + str(error.api) + ' >>> input: ' + str(error.input) + ' >>> error: ' + str(error.output))
            # print('call: ' + str(error.api) + ' >>> error: ' + str(error.output))
            # append_knowledge(error.api, error.input, error.output)

