from synthesis.synthesizer.decider import *
from synthesis.synthesizer.dplyr_to_pd.pd_result import *
from synthesis.search_structure import *
from utils.logger import get_logger
import pandas
import copy
from io import StringIO

logger = get_logger('synthesizer.decider')


class PDDecider(Decider):

    def __init__(self, test_cases: List[TestCase], matching_apis: List[LibraryAPI]):
        super().__init__(test_cases)
        self.matching_apis = matching_apis

    def error_message_understanding(self, raw_error_message: List[str], program: Program) -> (Constraint, List[str]):
        pass

    def analyze(self, program: Program) -> Result:
        target_call = program.code[0]

        # try to create layer
        logger.debug(f'Evaluating... {target_call}')

        # test cases
        output = None
        for test in self.test_cases:
            success, output = self.pandas_eval(program, test.input['pandas'])
            try:
                if success:
                    if not isinstance(output, pandas.core.groupby.DataFrameGroupBy):
                        expect = test.output['pandas'].to_numpy()
                        result = output.reset_index()
                        other_result = copy.deepcopy(output).reset_index(drop=True)
                        if np.array_equal(result.to_numpy(), expect):
                            return PDResult(True, out=pandas.read_csv(StringIO(result.to_csv(index=False))))
                        elif np.array_equal(other_result.to_numpy(), expect):
                            return PDResult(True, out=pandas.read_csv(StringIO(other_result.to_csv(index=False))))

                    else:
                        tmp_result = output.size().to_frame('size').query('size > 0')
                        if np.array_equal(tmp_result.reset_index().to_numpy(), test.output['pandas_count'].to_numpy()):
                            return PDResult(True, out=output)
            except Exception as e:
                logger.error(e)

        return PDResult(False)

    def pandas_eval(self, program, df: pandas.DataFrame):
        code = program.code[0]
        try:
            fn = eval(code)
            df1 = copy.deepcopy(df)
            result = fn(df1)
            if not isinstance(result, pandas.DataFrame) and \
               not isinstance(result, pandas.core.groupby.DataFrameGroupBy) and \
               not isinstance(result, pandas.Series):
                result = df1
            return True, result
        except:
            return False, None
