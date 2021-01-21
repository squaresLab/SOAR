import numpy as np

from typing import List, Dict, Tuple

from commons.library_api import LibraryAPI, load_apis, get_tokens_from_code
from commons.synthesis_program import Program, Constraint, TestCase
from mapping.representations import CodeWordCountRepresentation, CodeWordEmbeddingRepresentation
from mapping.representations import SummaryWordCountRepresentation, SummaryWordEmbeddingRepresentation
from mapping.representations import EmbeddingBasedRepresentation, FREQ_SMOOTH, stemming
import re
from functools import lru_cache


class ApiMatching:

    def __init__(self, source_library: str, target_library: str, k: int = 200,
                 use_embedding: bool = False, use_description: bool = False):
        """ Given the api from source library, find the apis in the target library that have similar functionality

        :param source_library: the name of the library that querying api is from (e.g. tf)
        :param target_library: the name of the library that target api in in (e.g. torch)
        :param k: number of candidates returned, default value: 10 """

        self.src_apis = load_apis(source_library)
        self.tgt_apis = load_apis(target_library)
        self.k = k
        self.use_embedding = use_embedding

        if use_embedding:
            if use_description:
                representation = SummaryWordEmbeddingRepresentation()
            else:
                representation = CodeWordEmbeddingRepresentation()
        else:
            if use_description:
                representation = SummaryWordCountRepresentation()
            else:
                representation = CodeWordCountRepresentation()

        representation.learn_representation(self.src_apis + self.tgt_apis)

        self.src_indices = [i for i in range(len(self.src_apis))]
        self.tgt_indices = [i + len(self.src_apis) for i in range(len(self.tgt_apis))]
        representation.build_query_index(self.src_indices, self.tgt_indices)

        self.representation = representation

    def query_for_new_api(self, api_call: str, lib: str = 'tgt'):
        api_code_keywords = [get_tokens_from_code(api_call)]

        if self.use_embedding:
            self.representation.break_to_subwords(api_code_keywords)
        else:
            api_code_keywords = stemming(api_code_keywords)

        # generate frequency representation for api
        freq_matrix = np.full(len(self.representation.vocab), FREQ_SMOOTH)  # smoothing to avoid nan error
        for word in api_code_keywords[0]:
            freq_matrix[self.representation.vocab[word]] += 1

        # inverse document normalization
        freq_matrix = freq_matrix / self.representation.word_freqs

        if self.use_embedding:
            new_api_embedding = np.zeros(300)
            for i in range(len(self.representation.vocab)):
                word = self.representation.vocab.get_word(i)
                word_embedding = self.representation.embedding_dict.get(word)
                new_api_embedding += word_embedding * freq_matrix[i]
            new_api_embedding = new_api_embedding / np.linalg.norm(new_api_embedding)
        else:
            new_api_embedding = freq_matrix / np.linalg.norm(freq_matrix)

        # calculate the similarity to all apis in the lib
        if lib == 'tgt':
            sim_scores = new_api_embedding @ self.representation.representation_matrix[self.tgt_indices].T
            top_k = sim_scores.argsort()[::-1][:self.k]
            return [(self.representation.lib_b_apis[i], sim_scores[i]) for i in top_k]
        elif lib == 'src':
            sim_scores = new_api_embedding @ self.representation.representation_matrix[self.src_indices].T
            top_k = sim_scores.argsort()[::-1][:self.k]
            return [(self.representation.lib_a_apis[i], sim_scores[i]) for i in top_k]
        else:
            raise ValueError('Unknown library {}'.format(lib))

    def word_matching(self, word1: str, word2: str) -> float:
        if not isinstance(self.representation, EmbeddingBasedRepresentation):
            raise ValueError('Must use word embedding for the api matcher!')

        word_embedding_1 = self.representation.index_word(word1)
        word_embedding_2 = self.representation.index_word(word2)

        cos_sim = np.dot(word_embedding_1, word_embedding_2) / \
                  (np.linalg.norm(word_embedding_1) * np.linalg.norm(word_embedding_2))
        return cos_sim

    def argument_matching(self, src_api: LibraryAPI, tgt_api: LibraryAPI, src_arg: str) -> List[Tuple[str, float]]:
        """
        try to find the most matching argument in the target api
        :param src_api: the api call in the source library
        :param tgt_api: the api call in the target library
        :param src_arg: the argument name in the source api
        :return: the arg names in the target api and their probabilities
        """

        # TODO this is a stub, which is a uniform distribution
        result = list(map(lambda x: (x.name, 1.0 / len(tgt_api.arguments)), tgt_api.arguments))
        return result

    def api_matching(self, api: LibraryAPI) -> List[Tuple[LibraryAPI, float]]:
        """ API Matching
        :return: a list of tuples, first item is the LibraryAPI type which includes a lot of useful information we obtained
                 from the document of this api, second is the probability
                 (e.g. [LibraryAPI(torch.nn.conv2d), 0.8), (LibraryAPI(torch.nn.conv1d), 0.12), ...]"""
        top_k_apis_with_prob = self.representation.query_top_k_with_prob(api.id, k=self.k)
        return top_k_apis_with_prob

    def get_api(self, api_full_name: str) -> LibraryAPI:
        rx = re.compile(api_full_name + '$')
        src_match = list(filter(lambda x: rx.search(x.id) is not None, self.src_apis))
        tgt_match = list(filter(lambda x: rx.search(x.id) is not None, self.tgt_apis))
        return (src_match + tgt_match)[0]

    @staticmethod
    @lru_cache(maxsize=5)
    def get_matcher(source_library, target_library, use_embedding=False, use_description=False, k=200):
        return ApiMatching(source_library, target_library, use_embedding=use_embedding, use_description=use_description, k=k)


def specification_mining(api: LibraryAPI) -> List[str]:
    raw_shape_str = api.raw_shape

    # establish a dictionary for converting variable name to simple letter
    var_lookup_dict = dict()
    var_letter_list = [chr(ord('a') + i) for i in range(26)] + [chr(ord('A') + i) for i in range(26)]

    # get inputs and outputs
    try:
        inputs_list = [x.group() for x in
                       re.finditer('\((.+?)\)', re.search('Input: (.+?)\n\n', raw_shape_str).group(1))]
        outputs_list = [x.group() for x in
                        re.finditer('\((.+?)\)', re.search('Output: (.+?)\n\n', raw_shape_str).group(1))]

        input_shape = inputs_list
        print()
    except AttributeError:
        print('no specification generated for {}'.format(api.id))


def error_message_understanding(raw_error_message: List[str], test_case,
                                program: Program) -> List[Constraint]:
    """
    From the error message produced from running the program, find a list of constraints that program should satisfy to
      reduce the search space.

    :param raw_error_message: a list of strings, each string is a line from the raw error message
    :param program: the program that generates error message
    :return: a list of constraints extracted from the error message
    """

    # TODO: this a dummy stub
    return []


def program_synthesis(source_program: Program,
                      test_cases: List[TestCase]) -> Program:
    """
    This could be further decomposed to several steps:
    1. extract the api call from the source_program that belongs to the old library
    2. find their similar alternatives in the new library by calling api_matching(...) function
    3. generate the new program (work your synthesis magic :P )
    4. for the generated new program, use the test cases to test
    5. if an error message occurred in step 4, call error_message_understanding(...) to get constraints
    6. apply the constraints and go back to step 3

    :param source_program: a program that we are trying to migrate
    :param test_cases: the test cases that includes the input and output for the program above
    :return: a program that pass the test cases and use the new library
    """

    return Program(["pass"])

if __name__ == '__main__':
    apis = load_apis('torch')
    apis_with_shape = list(filter(lambda x: len(x.raw_shape) > 0 and x.id.startswith('torch.nn'), apis))

    results = [specification_mining(api) for api in apis_with_shape]



