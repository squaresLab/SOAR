from synthesis.synthesizer.result import *


class PDResult(Result):

    def __init__(self, correct=False, err_msg=None, out=None):
        self.correct = correct
        self.err_msg = err_msg
        self.out = out

    def is_correct(self) -> bool:
        return self.correct

    def error_message(self) -> List[str]:
        return self.err_msg

    def output(self):
        return self.out
