from abc import ABC, abstractmethod
from typing import Any, List, Optional
from commons.z3_utils import *
from commons.library_api import *
from commons.synthesis_program import *
from synthesis.synthesizer.result import *


class Decider(ABC):

    def __init__(self, test_cases: List[TestCase]):
        self._test_cases = test_cases

    @property
    def test_cases(self):
        return self._test_cases

    @abstractmethod
    def analyze(self, program: Program) -> Result:
        """ Evaluates the program and decides whether is complies
        with the test cases or not.
        :param program: the program to analyze
        :param tests: the list of test cases used to check the validity of program
        :return: a data object containing information of the analysis"""

    @abstractmethod
    def error_message_understanding(self, raw_error_message: List[str],
                                    program: Program) -> (Constraint, List[str]):
        """ From the error message produced from running the program,
        find a list of constraints that program should satisfy to
        reduce the search space.

        :param raw_error_message: a list of strings, each string is a line from the raw error message
        :param program: the program that generates error message
        :return: a list of constraints extracted from the error message """
        raise NotImplementedError

