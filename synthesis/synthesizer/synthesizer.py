from abc import ABC, abstractmethod
from typing import Any
from synthesis.synthesizer.enumerator import *
from synthesis.synthesizer.decider import *
from autotesting.run_tests import test_synthesized_network_structure, \
    test_synthesized_forward_pass, load_example_by_name, generate_result_file
from mapping.representations import *
from synthesis.search_structure import *


class Synthesizer(ABC):

    @abstractmethod
    def synthesize(self) -> Optional[Program]:
        """ Main method of this class. It should use the Enumerator and Decider classes"""
