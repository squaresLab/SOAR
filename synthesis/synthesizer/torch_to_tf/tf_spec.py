from commons.synthesis_program import *
from synthesis.search_structure import VarPool
from itertools import combinations
from z3 import *
import numpy as np


class TFSpec:

    def __init__(self, test_case: TestCase):
        self.test_case = test_case
        self.input_array: np.ndarray = self.test_case.input['tf']
        self.output_array: np.ndarray = self.test_case.output

    def infer_ctr(self, api):
        ctr = []


        return None, None

    def values_from_test(self):
        pool = VarPool.get_empty_pool()
        values = set()
        for value in self.input_array.shape:
            values.add(str(value))
        # for val1, val2 in combinations(self.input_array.shape, 2):
        #     values.add(str(val1 * val2))
        pool['int'] = sorted(list(values))
        return pool
