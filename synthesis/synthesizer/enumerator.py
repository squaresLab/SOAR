from abc import ABC, abstractmethod
from typing import Any
from commons.z3_utils import *
from commons.library_api import *
from commons.synthesis_program import *


class Enumerator(ABC):

    def __init__(self, source_program: Program):
        self._source_program = source_program

    @property
    def source_program(self):
        return self._source_program

    @abstractmethod
    def next(self) -> Program:
        """ Enumerates the next program. """
        raise NotImplementedError

    @abstractmethod
    def has_next(self) -> bool:
        """ Verifies if the search space has been exhausted. """
        raise NotImplementedError

    @abstractmethod
    def update(self, constraint: Constraint, var_names: List[str]) -> None:
        """ Updates the synthesizer with a new set of constraints obtained
        from the error message understanding model. """
        raise NotImplementedError
