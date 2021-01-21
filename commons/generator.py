import re
import os, sys
import pandas as pd
from commons.library_api import *
import random
from commons.test_utils import *
from commons.synthesis_program import TorchProgram, Program

lib = load_library('tf')
lib.apis = list(filter(lambda x: x.id.find('keras.layers') != -1, lib.apis))
interpreter = Interpreter()
df = pd.DataFrame(columns=['api_id', 'api_call', 'test_shape', 'success', 'error_message'])

arguments = {}
for api in lib.apis:
    print(api.id)
    api_calls = []
    for i in range(1000):
        args = []
        for argument in api.arguments:
            if argument.type == 'int':
                args += [f'{argument.name}={random.randint(-256, 256)}']
            elif argument.type == 'bool':
                args += [f'{argument.name}={random.randint(0, 1)}']
            elif argument.type == 'float':
                args += [f'{argument.name}={random.random() * 256 * (random.randint(-1, 1))}']
        api_calls += [api.id + f'({",".join(args)})']

    for i in range(1000):
        shape = [20, random.randint(0, 5)] + [random.randint(1, 256) for _ in range(random.randint(0, 2))]
        shape = [el for el in shape if el > 0]
        test = np.random.rand(*shape)
        success, out = interpreter.tensor_forward_pass(program=Program([api_calls[i]]), input_tensor=test)
        if success: out = ''
        df = df.append({'api_id': api.id, 'api_call': api_calls[i], 'success': success,
                   'test_shape': shape, 'error_message': out}, ignore_index=True)

df.to_csv(path_or_buf='./errors.csv')