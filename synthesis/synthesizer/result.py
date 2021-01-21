from abc import ABC, abstractmethod
from typing import Any, List, Optional
from commons.z3_utils import *
from commons.library_api import *
from commons.synthesis_program import *

class Result(ABC):

    @abstractmethod
    def is_correct(self) -> bool:
        """ Returns information regarding the correctness of the program """
        raise NotImplementedError

    @abstractmethod
    def error_message(self) -> List[str]:
        """ Error message generated throughout the execution of the program. """
        raise NotImplementedError
