from commons.synthesis_program import *
from synthesis.search_structure import VarPool
from itertools import combinations
from z3 import *
import numpy as np


class TorchSpec:

    def __init__(self, test_case: TestCase, spec_enabled: bool = False):
        self.test_case = test_case
        self.input_array: np.ndarray = self.test_case.input['tf']
        self.output_array: np.ndarray = self.test_case.output
        self.spec_enabled = spec_enabled

    def infer_ctr(self, api):
        ctr = []
        if not self.spec_enabled: return None, None

        if api.name == 'Linear':
            h_in = self.input_array.shape[1]
            h_out = self.output_array.shape[1]

            in_features = Var(0, IntSort())
            out_features = Var(1, IntSort())

            ctr.append(in_features == h_in)
            ctr.append(out_features == h_out)
            return And(ctr), ['in_features', 'out_features']

        if api.name == 'Conv1d' or api.name == 'MaxPool1d':
            if len(self.input_array.shape) < 3:
                return Not(True), []
            l_in = self.input_array.shape[1]
            l_out = self.output_array.shape[1]

            dilation = int(next(filter(lambda x: x.name == 'dilation', api.arguments)).default_value)
            kernel_size = Var(0, IntSort())
            stride = Var(1, IntSort())
            padding = Var(2, IntSort())

            ctr.append(l_out == (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            ctr.append(stride != 0)

            return And(ctr), ['kernel_size', 'stride', 'padding']

        if api.name == 'Conv2d' or api.name == 'MaxPool2d':
            if len(self.input_array.shape) < 4:
                return Not(True), []
            h_in = self.input_array.shape[1]
            h_out = self.output_array.shape[1]
            w_in = self.input_array.shape[2]
            w_out = self.output_array.shape[2]

            dilation = int(next(filter(lambda x: x.name == 'dilation', api.arguments)).default_value)
            kernel_size = [Var(0, IntSort()), Var(1, IntSort())]
            stride = [Var(2, IntSort()), Var(3, IntSort())]
            padding = [Var(4, IntSort()), Var(5, IntSort())]

            ctr.append(h_out == (h_in + 2 * padding[0] - dilation * (kernel_size[0] - 1) - 1) / stride[0] + 1)
            ctr.append(w_out == (w_in + 2 * padding[1] - dilation * (kernel_size[1] - 1) - 1) / stride[1] + 1)
            ctr.append(And([stride[i] != 0 for i in range(2)]))

            return And(ctr), ['kernel_size_0', 'kernel_size_1', 'stride_0', 'stride_1', 'padding_0', 'padding_1']

        if api.name == 'Conv3d' or api.name == 'MaxPool3d':
            if len(self.input_array.shape) < 5:
                return Not(True), []
            d_in = self.input_array.shape[1]
            d_out = self.output_array.shape[1]
            h_in = self.input_array.shape[2]
            h_out = self.output_array.shape[2]
            w_in = self.input_array.shape[3]
            w_out = self.output_array.shape[3]

            dilation = int(next(filter(lambda x: x.name == 'dilation', api.arguments)).default_value)
            kernel_size = [Var(0, IntSort()), Var(1, IntSort()), Var(2, IntSort())]
            stride = [Var(3, IntSort()), Var(4, IntSort()), Var(5, IntSort())]
            padding = [Var(6, IntSort()), Var(7, IntSort()), Var(8, IntSort())]
            ctr.append(d_out == (d_in + 2 * padding[0] - dilation * (kernel_size[0] - 1) - 1) / stride[0] + 1)
            ctr.append(h_out == (h_in + 2 * padding[1] - dilation * (kernel_size[1] - 1) - 1) / stride[1] + 1)
            ctr.append(w_out == (w_in + 2 * padding[2] - dilation * (kernel_size[2] - 1) - 1) / stride[2] + 1)
            ctr.append(And([stride[i] != 0 for i in range(3)]))

            return And(ctr), [f'kernel_size_{i}' for i in range(3)] + \
                             [f'stride_{i}' for i in range(3)] + \
                             [f'padding_{i}' for i in range(3)]

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
