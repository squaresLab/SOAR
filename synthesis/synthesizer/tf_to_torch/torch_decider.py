from synthesis.synthesizer.decider import *
from synthesis.synthesizer.tf_to_torch.torch_result import *
from synthesis.synthesizer.synthesizer import *
from synthesis.search_structure import *
from commons.test_utils import Interpreter, extract_api_arguments_torch, code_to_params
import numpy as np
import nltk
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import concurrent.futures
from z3 import *
from utils.logger import get_logger
from fuzzywuzzy import fuzz
from transformers import *
from commons.interfaces import ApiMatching

logger = get_logger('synthesizer.decider')


class Z3Decider(Decider):

    def __init__(self, test_cases: List[TestCase], matching_apis: List[LibraryAPI],
                 interpreter: Interpreter, api_matcher: ApiMatching):
        super().__init__(test_cases)
        self.matching_apis = matching_apis
        self.interpreter = interpreter
        self.api_matcher = api_matcher

    def is_number(self, text):
        if 'zero' in text:
            # For some reason zero is not CD in nltk
            return True
        text = nltk.word_tokenize(text)
        pos = nltk.pos_tag(text)
        for i in range(len(pos)):
            word, pos_tag = pos[i]
            if pos_tag == 'CD':
                return True
        return False

    def preprocess(self, msg):
        sent = nltk.word_tokenize(msg)
        sent = nltk.pos_tag(sent)
        return sent

    def nlp_tagger(self, err_msg, pattern):
        msg = self.preprocess(err_msg)
        cp = nltk.RegexpParser(pattern)
        cs = cp.parse(msg)
        iob_tagged = tree2conlltags(cs)
        return iob_tagged

    def iob_only(self, iob_tree):
        iob_list = []
        for word in iob_tree:
            if 'NP' in word[2]:
                iob_list.append(word)
        return iob_list

    def mutate_constraint(self, program, arg):
        mutated_args = []
        test_program = program.code[0].split('(')
        # 'self.var5 = torch.nn.Conv2d(1,3,0,stride=0,padding=0,dilation=1,groups=1,bias=True,padding_mode=\'zeros\')'

        test_list = (test_program[1].replace(')', '')).split(',')

        for idx, change_arg in enumerate(test_list):
            if change_arg == arg and idx not in mutated_args:

                ## if number is neg flip to positive
                if '-' in change_arg:
                    test_list[idx] = change_arg.replace('-', '')
                ## else change to a different number based on constraint
                else:

                    test_list[idx] = change_arg.replace(arg, '1')

                target_mute = idx
                mutated_args.append(target_mute)
                break
        # done change list
        mutated_code = test_program[0] + '(' + ','.join(test_list) + ')'
        program.code[0] = mutated_code
        z3res = self.analyze(program)
        return [z3res.error_message(), target_mute]

    # def multi_constraints(self, params, ):

    def param_not_supported(self, iob_tagged, params, program):
        for idx, word in enumerate(iob_tagged):
            if word[1] == 'JJ':
                target_arg = iob_tagged[idx + 1][0]
                break
        for idx, param in enumerate(params):
            if target_arg in param:
                target_idx = idx + 1
                param_tuple_list = []
                var_tuple_list = []
                param_list = code_to_params(param)
                if len(param_list) > 1:
                    for pidx, p in enumerate(param_list):
                        p = int(p)
                        if p < 0:
                            if 'padding' in str(program.argument_vars[idx]):
                                var_tuple_list.append(z3.Var(pidx, IntSort()) >= 0)
                            elif 'out_channels' in str(program.argument_vars[idx]):
                                var_tuple_list.append(z3.Var(pidx, IntSort()) > 0)
                            else:
                                var_tuple_list.append(z3.Var(pidx, IntSort()) > 0)
                            param_tuple_list.append(str(program.argument_vars[idx]) + '_' + str(pidx))
                    z3_constr = And(var_tuple_list), param_tuple_list
                else:
                    var = z3.Var(0, IntSort())
                    z3_constr = var >= 0, [program.argument_vars[idx]]
                    if 'stride' in program.argument_vars[idx]:
                        z3_constr = var > 0, [program.argument_vars[idx]]
                break
        return z3_constr

    def faulty_matrix_idx(self, iob, params, program):
        bad_arg = None
        main_arg = None
        arg_constr = ''
        var_tuple_list = []
        param_tuple_list = []
        for idx, word in enumerate(iob):
            if 'B-NP' in word[2] and '[' not in word[0]:
                main_arg = word[0]
            if 'NP' in word[2] and 'JJ' in word[1]:
                arg_constr = word[0]
            if '[' in word[0]:
                matrix_start = idx
        bad_arg_iob = iob[(matrix_start - 2)]
        if 'N' in bad_arg_iob[1]:
            bad_arg = bad_arg_iob[0]
        constraint = ''
        if main_arg and arg_constr:
            if arg_constr and 'neg' in arg_constr:
                for idx, arg in enumerate(params):
                    if arg == bad_arg:
                        target_idx = idx + 1
                        break
                    elif '(' in arg:
                        arg_list = arg[arg.find("(") + 1:arg.find(")")].split(',')
                        for tup_idx, a in enumerate(arg_list):
                            if int(a) == int(bad_arg):
                                target_idx = idx + 1
                                var_constr = z3.Var(tup_idx, IntSort()) > 0
                                var_tuple_list.append(var_constr)
                                param_tuple_list.append(program.argument_vars[target_idx] + '_' + str(tup_idx))
                        constraint = var_tuple_list, param_tuple_list
                        return constraint

                var0 = Var(0, IntSort())
                constraint = var0 > 0, [program.argument_vars[target_idx - 1]]
        return constraint

    def zero_param(self, err_msg, iob_tagged, program, params):
        constraint = ''
        for idx, word in enumerate(iob_tagged):
            if 'I-NP' in word[2]:
                if idx == (len(iob_tagged) - 1) or 'O' in iob_tagged[idx + 1][2]:
                    constraint = word[0]
        if self.is_number(constraint):
            if 'zero' in constraint:
                constraint = '0'
            arg_count = 1
            for arg in params:
                if constraint in arg:
                    # TODO this is where I add mutation
                    mutated_msg = self.mutate_constraint(program, arg)
                    if type(mutated_msg[0]) == list:
                        mutated_msg[0] = str(mutated_msg[0]).replace('[', '').replace(']', '').replace('\'', '')
                    if mutated_msg[0] != err_msg:
                        target_idx = mutated_msg[1] + 1
                        var0 = Var(0, IntSort())
                        constraint = var0 > 0, [program.argument_vars[mutated_msg[1]]]
                        break
                else:
                    arg_count += 1
        return constraint

    def faulty_weight(self, err_msg, iob_tagged, program, params):
        constraint = ''
        # 'Given groups=1, weight of size [50, 50, 3, 3], expected input[100, 1, 50, 40] to have 50 channels, but got 1 channels instead'
        # ["self.var4 = torch.nn.Conv2d(50,50,3,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')"]
        num_channels_iob = self.nlp_tagger(err_msg, 'NP: {<TO><VB><CD>}')
        target_idx = None
        for word in num_channels_iob:
            if 'NP' in word[2] and 'CD' in word[1]:
                for idx, word_exp in enumerate(num_channels_iob):
                    if 'expect' in word_exp[0]:
                        target_argname = num_channels_iob[idx + 1][0]
                        break
                if 'in' in target_argname:
                    target_idx = 0
                elif 'out' in target_argname:
                    target_idx = 1
                if target_idx is not None:
                    var0 = Var(0, IntSort())
                    constraint = var0 == word[0], [program.argument_vars[target_idx]]
                    break
        return constraint

    def idx_obounds(self, iob_numbers, params, program):
        constraint = ''
        for idx, word in enumerate(iob_numbers):
            if 'B-NP' in word[2] and 'index' in word[0]:
                target_idx = int(iob_numbers[idx + 1][0])
                break
        for idx, word in enumerate(iob_numbers):
            if 'B-NP' in word[2] and 'dimension' in word[0]:
                input_dim = int(iob_numbers[idx + 1][0])
                break
        var0 = Var(0, IntSort())
        constraint = var0 > input_dim, [program.argument_vars[target_idx]]
        return constraint

    def wrong_shape(self, iob_tagged, iob_refinement, program, params):
        faulty_arg = ''
        target_idx = None
        for idx, word in enumerate(iob_tagged):
            if word[2] == 'I-NP' and iob_tagged[idx + 1] != 'I-NP':
                matrix_words = iob_tagged[idx + 1:len(iob_refinement) - 1]
                for idx, matrix_word in enumerate(matrix_words):
                    if matrix_word[0] == ']':
                        matrix_words = matrix_words[0:idx - 1]
                for word in matrix_words:
                    if word[1] == 'CD':
                        for idx, arg in enumerate(params):
                            if str(word[0]) in arg:
                                faulty_arg = word[0]
                                target_idx = idx + 1

        for idx, word in enumerate(iob_refinement):
            if word[2] == 'I-NP' and iob_refinement[idx + 1] != 'I-NP':
                matrix_words = iob_refinement[idx + 1:len(iob_refinement) - 1]
                for matrix_word in matrix_words and faulty_arg:
                    if matrix_word[1] == 'CD' and matrix_word[0] != faulty_arg:
                        target_constraint = int(matrix_word[0])
        if target_idx is not None:
            var0 = Var(0, IntSort())
            constraint = var0 == target_constraint, [program.argument_vars[target_idx - 1]]
            return constraint

    def ofr_self(self, err_msg, iob_tagged, program, params):
        max_input = self.test_cases[0].input[0].max()
        max_input = int(max_input)
        for idx, arg in enumerate(program.argument_vars):
            if 'num' in arg:
                target_idx = idx + 1
                break
        # var0 = Var(target_idx, IntSort())
        var0 = Var(0, IntSort())
        constraint = var0 > max_input, [program.argument_vars[target_idx - 1]]
        if constraint:
            return constraint
        else:
            return None, None

    def enumerative_mutation(self, iob_tagged, program, err_matrices):
        constraint = ''
        for word in iob_tagged:
            for idx, arg in enumerate(program.argument_vars):
                if idx == 0 and len(arg) < 3:
                    arg = 'input'
                elif idx == 1 and len(arg) < 3:
                    arg = 'output'
                fuzz_ratio = fuzz.ratio(word[0], arg)

                if fuzz_ratio > 50:
                    target_param = arg
                    target_idx = idx

        return None, None

    def expect_bgot(self, err_msg, program, params):
        constraint = None, None
        split_msg = code_to_params(err_msg)
        e_num = ''
        target_p = ''
        target_idx = 'NA'
        var_tuple_list = []
        param_tuple_list = []
        expect_m = ''
        param_m = ''

        if "to have scalar type" in err_msg:
            var0 = Var(0, IntSort())
            return var0 != var0, []

        for msg in split_msg:
            eachword = msg.split(' ')
            if 'but got' in msg:
                param_m = msg[msg.find("[") + 1:msg.find("]")].split(',')
            elif 'expected' in msg:
                expect_m = msg[msg.find("[") + 1:msg.find("]")].split(',')
                for idx, m in enumerate(eachword):
                    if 'channels' in m:
                        try:
                            e_num = int(eachword[idx - 1])
                        except:
                            continue

        if 'but got' in param_m[0]:
            param_m = param_m[0].split(' ')
            for p_num in param_m:
                try:
                    target_p = int(p_num)
                    break
                except:
                    continue
            if target_p and e_num:
                for param_idx, param in enumerate(params):
                    try:
                        param = int(param)
                    except:
                        continue
                    if e_num == param:
                        target_idx = param_idx
                        break

                if isinstance(target_idx, int):
                    if program.before and program.before[0].find('permute') != -1:
                        permute = program.before[0]
                        permute_list = permute[permute.find("(") + 1:permute.find(")")].split(',')
                        var_perm_list = []
                        param_tuple_list = []
                        counter = 0
                        for perm_idx, perm in enumerate(permute_list):
                            var_perm_list.append(z3.Var(perm_idx, IntSort()) == int(perm))
                            param_tuple_list.append('permute_dim_' + str(perm_idx))
                            counter = perm_idx
                        perm_constr = And(var_perm_list)

                        var_constr = z3.Var(counter + 1, IntSort()) == target_p
                        final_constr = Implies(perm_constr, var_constr)
                        param_tuple_list.append(program.argument_vars[target_idx])
                        constraint = final_constr, param_tuple_list
                        return constraint
                    else:
                        return z3.Var(0, IntSort()) == e_num, [program.argument_vars[target_idx]]

        elif param_m and expect_m:
            for pidx, p_num in enumerate(param_m):
                for edix, e_num in enumerate(expect_m):
                    if '*' in e_num:
                        continue
                    if pidx == edix and int(p_num) != int(e_num):
                        for idx, param in enumerate(params):
                            try:
                                param = int(param)
                            except:
                                continue
                            if param == int(e_num):
                                var0 = Var(0, IntSort())
                                constraint = var0 == int(e_num), [program.argument_vars[idx]]
                                break
        return constraint

    def vectorize_compare(self, api_matcher, err_msg, program, params):
        constraint = None, None
        err_msg_list = err_msg.split(' ')

        # prep remove unnecessary square brackets and initialize inner square bracket as list
        if '[' in err_msg_list[0] and ']' in err_msg_list[1]:
            err_msg_list = err_msg_list[1:len(err_msg_list) - 2]
        err_matrices = err_msg[err_msg.find("[") + 1:err_msg.find("]")].split(',')
        for idx, err in enumerate(err_matrices):
            try:
                err_matrices[idx] = int(err.replace(' ', ''))
            except:
                continue
                err = float(err.replace(' ', ''))
                err_matrices[idx] = int(err)

        ##### Output: err_matrices

        # This block of code prepares the x: before a matrix
        colon_num = ''
        if err_matrices:
            for target_num in err_msg_list:
                if ':' in target_num:
                    colon_num = colon_num.replace(':', '')
                    if not colon_num.isnumeric():
                        colon_num = ''
        ##### Output: colon_num

        var0 = Var(0, IntSort())
        for word in err_msg_list:
            if 'dimension' in word or 'Dimension' in word:
                word = 'dim'
            word = word.replace('[', '').replace(']', '').replace(',', '').replace('(', '').replace(')', '')

            try:
                vectorized_w = ApiMatching.word_matching(api_matcher, word, arg)
            except:
                return None, None

            for idx, arg in enumerate(program.argument_vars):
                if abs(vectorized_w) > 0.7:
                    # match between arg and err msg
                    if not word.isnumeric():
                        # err msg word is not a number
                        print(word)
                        if err_matrices and 'range' in err_msg:
                            max_r = max(err_matrices)
                            min_r = min(err_matrices)
                            And(var0 > -3, var0 < 5)
                            constraint = And(var0 > min_r, var0 < max_r), [program.argument_vars[idx]]
                            return constraint

            for idx, arg in enumerate(params):
                if abs(vectorized_w) > 0.7:
                    if 'negative' in err_msg:
                        # if negative shows up at all we say constraint > 0
                        # var0 = Var(idx + 1, IntSort())
                        try:
                            constraint = var0 >= 0, program.argument_vars[idx]
                            return constraint
                        except:
                            continue
                    if 'out of range' in err_msg:
                        # if out of range we just find max matrix value and make dimension higher than that
                        try:
                            max_input = self.test_cases[0].input[0].max()
                            max_input = int(max_input)
                            constraint = var0 > max_input, [program.argument_vars[idx]]
                        except:
                            continue

        return constraint

    def error_message_understanding(self, raw_error_message: List[str], program: Program) -> (
            Constraint, List[str]):
        code = program.code[0]
        params = code_to_params(code)
        err_msg = str(raw_error_message)
        err_msg = err_msg.replace("[\'", "").replace("\']", "")
        if not err_msg:
            return None, None
        if 'Wrong shape ' in err_msg and 'vs' in err_msg:
            return None, None
        if 'missing' in err_msg:
            var0 = Var(0, IntSort())
            return var0 != var0, [program.argument_vars[0]]
        if re.findall(r'takes(.*)but(.*)were given', err_msg):
            var0 = Var(0, IntSort())
            return var0 != var0, [program.argument_vars[0]]
        if 'Could not run' in err_msg:
            var0 = Var(0, IntSort())
            return var0 != var0, [program.argument_vars[0]]
        if "axes don't match" in err_msg:
            return None, None
        if "must be Tensor," in err_msg:
            var0 = Var(0, IntSort())
            return var0 != var0, [program.argument_vars[0]]
        if "requires" in err_msg and "but received a" in err_msg:
            var0 = Var(0, IntSort())
            return var0 != var0, []

        # if "non-positive stride" in err_msg:
        #     print("here")

        # iob_tagged example: for 3-dimensional weight...
        iob_tagged = self.nlp_tagger(err_msg, 'NP: {<NN>*<IN><JJ>?<NN>}')
        # iob_numbers example: index 5 ...
        iob_cd = self.nlp_tagger(err_msg, 'NP: {<NN><CD>}')
        # iob_refinement example: but got 4-dimensional input...
        iob_refinement = self.nlp_tagger(err_msg, 'NP: {<CC><VBD><JJ>?<NN>}')
        # iob_adv example: stride is not supported
        iob_adv = self.nlp_tagger(err_msg, 'NP: {<JJ>?<NN><VBZ><RB><VBN>}')

        tagged_list = self.iob_only(iob_tagged)
        cdnum_list = self.iob_only(iob_cd)
        refinement_list = self.iob_only(iob_refinement)
        adv_list = self.iob_only(iob_adv)

        constraint = ''
        logger.debug(f'Error message: {raw_error_message}')

        # 'Expected 3-dimensional input for 3-dimensional weight [2, 2, 3],
        # but got 4-dimensional input of size [100, 50, 40, 1] instead'

        if tagged_list:
            for iob_word in iob_tagged:
                if iob_word[2] == 'B-NP' and iob_word[0] == 'weight':
                    constraint = self.faulty_weight(err_msg, iob_tagged, program, params)
                    break
            if '[' in err_msg and ']' in err_msg:
                # Matrix provided in the error message
                constraint = self.faulty_matrix_idx(iob_tagged, params, program)

            if 'zero' in err_msg:
                constraint = self.zero_param(err_msg, iob_tagged, program, params)

            if 'self' in err_msg:
                constraint = self.ofr_self(err_msg, iob_tagged, program, params)
            if 'but got' in err_msg:
                constraint = self.expect_bgot(err_msg, program, params)

        if cdnum_list:
            if 'out of bounds' in err_msg:
                constraint = self.idx_obounds(iob_cd, params, program)

        if refinement_list:
            constraint = self.expect_bgot(err_msg, program, params)

        if refinement_list and tagged_list and not constraint:
            constraint = self.wrong_shape(iob_tagged, iob_refinement, program, params)

        if adv_list:
            if 'not supported' in err_msg:
                constraint = self.param_not_supported(adv_list, params, program)

        if not constraint:
            constraint = self.vectorize_compare(self.api_matcher, err_msg, program, params)

        if not isinstance(constraint, tuple):
            return None, None

        return constraint

    def analyze(self, program: TorchProgram) -> TorchResult:
        target_call = program.code[0]

        # try to create layer
        logger.debug(f'Evaluating... {target_call}')
        if target_call.find("stride=0") != -1 or target_call.find("stride=(0,0)") != -1:
            return TorchResult(False)

        # test cases
        output = None
        for test in self.test_cases:
            success, output = self.interpreter.torch_forward_pass(program, test.input['torch'])
            if not success:  # runtime error
                return TorchResult(False, error_msg=output)
            elif sorted(test.output.shape) != sorted(output.shape):  # wrong shape
                return TorchResult(False, [f'Wrong shape {test.output.shape} vs {output.shape}'])
            else:
                transformed_output = output[:]
                idx = []
                for i in range(len(test.output.shape)):
                    for j in range(len(test.output.shape)):
                        if test.output.shape[i] == output.shape[j] and j not in idx:
                            idx += [j]
                transformed_output = transformed_output.transpose(tuple(idx))
                if not np.allclose(test.output, transformed_output, rtol=1e-04, atol=1e-07):  # we don't know why it failed
                    return TorchResult(False)
                program.after = ['{input}.' + f'permute({",".join(map(str,idx))})']

        return TorchResult(True, output=output)
