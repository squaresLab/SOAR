from abc import ABC, abstractmethod
from typing import Any
from synthesis.synthesizer.enumerator import *
from synthesis.synthesizer.decider import *
from synthesis.synthesizer.synthesizer import *
from autotesting.run_tests import test_synthesized_network_structure, \
    test_synthesized_forward_pass, load_example_by_name, generate_result_file
from mapping.representations import *
from synthesis.search_structure import *


# TODO
class TorchResult(Result):

    def __init__(self, success: bool, error_msg: List[str] = None, output: np.ndarray = None):
        self.success = success
        self.error_msg = error_msg
        self.output = output

    def is_correct(self) -> bool:
        return self.success

    def error_message(self) -> List[str]:
        return self.error_msg

    def torch_output(self):
        return self.output
