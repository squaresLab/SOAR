from commons.synthesis_program import *
from synthesis.search_structure import VarPool
from itertools import combinations
from z3 import *
import numpy as np


class PDSpec:

    def __init__(self, test_case: TestCase):
        self.test_case = test_case
        self.input = self.test_case.input['pandas']
        self.output = self.test_case.output['pandas']