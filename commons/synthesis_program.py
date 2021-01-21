from typing import Dict, List, Any
from z3 import ExprRef
from commons.library_api import LibraryAPI
import os
import numpy as np


class TestCase:
    """
    Test case that have inputs and expected outputs
    """

    def __init__(self,
                 input: Dict,
                 output: Any,
                 api: LibraryAPI,
                 executable_code: str = None):
        """
        :param input: a dictionary, key is the name of the variable, value is the value of that variable
        :param output: return values
        :param api: the api that this test case is on
        :param executable_code: use this to save the one-liner of code to simply things
        """
        self.input = input
        self.output = output
        self.api = api
        self.executable_code = executable_code

    def __eq__(self, other):
        if type(other) != type(self):
            return False

        if not self.executable_code and not other.executable_code:
            if self.executable_code == other.executable_code:
                return True

        return False


class Program:
    """
    Currently it is an alias for List[str] type to include lines of code, but we can make it more complicated it
      we need to.
    """

    def __init__(self, code: List[str]):
        self.code = code

    def __getitem__(self, item):
        self.code.__getitem__(item)

    def __setitem__(self, key, value):
        self.code.__setitem__(key, value)

    def __str__(self):
        prog = ''
        for line in self.code:
            prog += line + os.linesep
        return prog

    def __eq__(self, other):
        if isinstance(other, Program):
            if len(other.code) == len(self.code):
                for i in range(len(self.code)):
                    if self.code[i].split('=')[1] != other.code[i].split('=')[1]:
                        return False
                return True
        return False


class TorchProgram(Program):
    n = 0
    def __init__(self, code: List[str], argument_vars: List[str], before=(), after=()):
        super().__init__(code)
        if before is None:
            before = []
        self.argument_vars = argument_vars
        self.before = before
        self.after = after

    def __getitem__(self, item):
        self.argument_vars.__getitem__(item)

    def __setitem__(self, key, value):
        self.argument_vars.__setitem__(key, value)

    def print(self):
        out = []
        if self.before:
            out += [f'self.before_{TorchProgram.n} = lambda x: ' + self.before[0].format(input='x').strip()]
            TorchProgram.n += 1
        out += self.code
        if self.after:
            #out += [f'self.after_{TorchProgram.n} = lambda x: ' + self.after[0].format(input='x').strip()]
            TorchProgram.n += 1
        return out

    def linearize(self, var_name):
        lines = []
        for line in self.code:
            lines += [line[line.find('=')+1:].strip()]
        for line in self.before:
            lines += [line.format(input=var_name).strip()]
        lines += [f'torch_layer({var_name})']
        for line in self.after:
            lines += [line.format(input=var_name).strip()]
        return lines

"""
We use constraint as an alias for a Z3 expression.

Example: 
x = z3.Int('x')
z3.And( x >= 0, x <= 20) is a constraint
"""
Constraint = ExprRef
