import numpy as np
import json
import pathlib

from functools import reduce
from numpy import ndarray
from typing import Collection
from nltk.stem.snowball import SnowballStemmer
from typing import List, Tuple, Dict
from wordfreq import word_frequency

from commons.library_api import LibraryAPI, get_tokens_from_code, load_apis
from mapping.vocab import VocabEntry
from utils.logger import get_logger

logger = get_logger('library')
logger.setLevel("INFO")

FREQ_SMOOTH = 0.0001


def stemming(training_sentences: List[List[str]]) -> List[List[str]]:
  result = []
  
  stemmer = SnowballStemmer("english")
  for sentence in training_sentences:
    result.append([stemmer.stem(word) for word in sentence])
  
  return result


def stemming_with_vocab(training_sentences: List[List[str]], vocab: Collection[str]) -> List[List[str]]:
  result = []

  stemmer = SnowballStemmer("english")
  for sentence in training_sentences:

    stemmed_sentence = []
    for word in sentence:
      word_stem = stemmer.stem(word)
      if word_stem in vocab:
        stemmed_sentence.append(word_stem)
      else:
        stemmed_sentence.append(word)

    result.append(stemmed_sentence)

  return result


class Representation:
  """
  The master class for all kinds of representations
  """
  def __init__(self):
    super(Representation, self).__init__()

    self.representation_matrix: 'np.ndarray' = None
    self.apis: List[LibraryAPI] = []
    self.vocab: VocabEntry = None
    self.word_freqs = None

    # in the query mode
    self.lib_a_apis = None
    self.lib_b_apis = None
    self.lib_a_dict = None
    self.lib_b_dict = None
    self.similarity_matrix = None

  def build_query_index(self, lib_a_idx, lib_b_idx):
    # get those apis separate
    self.lib_a_apis = [self.apis[i] for i in lib_a_idx]
    self.lib_b_apis = [self.apis[i] for i in lib_b_idx]
    rep_a = self.representation_matrix[lib_a_idx]
    rep_b = self.representation_matrix[lib_b_idx]
    
    # first build two dictionaries for faster index finding
    self.lib_a_dict = dict(zip(map(lambda x: x.id, self.lib_a_apis), range(len(self.lib_a_apis))))
    self.lib_b_dict = dict(zip(map(lambda x: x.id, self.lib_b_apis), range(len(self.lib_b_apis))))
    
    # then build the similarity matrix
    self.similarity_matrix = rep_a @ rep_b.T
  
  def query_similarity(self, api_a_id: str, api_b_id: str):
    if api_a_id in self.lib_b_dict and api_b_id in self.lib_a_dict:
      tmp = api_a_id
      api_a_id = api_b_id
      api_b_id = tmp
    elif not (api_a_id in self.lib_a_dict and api_b_id in self.lib_b_dict):
      raise ValueError('{0} and {1} do not match two apis in the libraries'.format(api_a_id, api_b_id))
    
    api_a_idx = self.lib_a_dict[api_a_id]
    api_b_idx = self.lib_b_dict[api_b_id]
    
    return self.similarity_matrix[api_a_idx][api_b_idx]
  
  def query_top_k_with_prob(self, api_id: str, k: int = 10) -> List[Tuple[LibraryAPI, float]]:
    if api_id in self.lib_a_dict:
      api_idx = self.lib_a_dict[api_id]
      sims = self.similarity_matrix[api_idx, :]
      top_k = sims.argsort()[::-1][:k]
      return [(self.lib_b_apis[i], self.similarity_matrix[api_idx, i]) for i in top_k]
    elif api_id in self.lib_b_dict:
      api_idx = self.lib_b_dict[api_id]
      sims = self.similarity_matrix[:, api_idx]
      top_k = sims.argsort()[::-1][:k]
      return [(self.lib_a_apis[i], self.similarity_matrix[i, api_idx]) for i in top_k]
    else:
      raise ValueError('The api id {0} is not in any of the libraries'.format(api_id))

  def query_top_k(self, api_id: str, k: int = 10) -> List[LibraryAPI]:
    if api_id in self.lib_a_dict:
      api_idx = self.lib_a_dict[api_id]
      sims = self.similarity_matrix[api_idx, :]
      top_k = sims.argsort()[::-1][:k]
      return [self.lib_b_apis[i] for i in top_k]
    elif api_id in self.lib_b_dict:
      api_idx = self.lib_b_dict[api_id]
      sims = self.similarity_matrix[:, api_idx]
      top_k = sims.argsort()[::-1][:k]
      return [self.lib_a_apis[i] for i in top_k]
    else:
      raise ValueError('The api id {0} is not in any of the libraries'.format(api_id))
      
  def print_top_similar_apis(self, indices_a: List[int], indices_b: List[int],
                             k: int = 200, ignore_same_name: bool = False) -> None:
    rep_a = self.representation_matrix[indices_a]
    rep_b = self.representation_matrix[indices_b]
  
    similarity_matrix = rep_a @ rep_b.T
    linearized_similarity = similarity_matrix.reshape(-1)
    sorted_linearized_indices = linearized_similarity.argsort()[::-1]
  
    # calculate the real indices and print the results
    print_count = i = 0
    while print_count < k:
      l_idx = sorted_linearized_indices[i]
      i += 1
      
      idx_a = l_idx // int(similarity_matrix.shape[1])
      idx_b = l_idx % int(similarity_matrix.shape[1])
    
      api_name_a = self.apis[indices_a[idx_a]].id
      api_name_b = self.apis[indices_b[idx_b]].id
      similarity_score = similarity_matrix[idx_a][idx_b]
      
      # ignore the apis with the same name to find more interesting mappings
      if ignore_same_name and api_name_a.split('.')[-1].lower() == api_name_b.split('.')[-1].lower():
        # logger.debug('Skip {0} and {1}'.format(api_name_a.split('.')[-1].lower(), api_name_b.split('.')[-1].lower()))
        continue
      else:
        print_count += 1
        logger.debug('{0} and {1} with similarity score {2}'.format(api_name_a, api_name_b, similarity_score))

  def break_to_subwords(self, words_list: List[List[str]]):
    # manually break unknown words to subwords
    for i, api_keywords in enumerate(words_list):
      for j, keyword in enumerate(api_keywords):
        if keyword not in self.vocab:
          break_candidates = []
          for k in range(1, len(keyword)-1):
            if keyword[:k] in self.vocab and keyword[k:] in self.vocab:
              break_score = word_frequency(keyword[:k], 'en') + word_frequency(keyword[k:], 'en')
              break_candidates.append((keyword[:k], keyword[k:], break_score))

          if len(break_candidates) > 0:
            # get the best candidate based on the frequency scores
            best_candidate = sorted(break_candidates, key=lambda x: x[2], reverse=True)[0]
            words_list[i][j] = best_candidate[0]
            words_list[i].insert(j+1, best_candidate[1])
            logger.debug('Unknown word {}: break into {} and {}'.format(keyword, best_candidate[0], best_candidate[1]))

  def learn_representation(self, training_apis: List[LibraryAPI]) -> None:
    pass

  def get_api_embedding(self, api: LibraryAPI) -> ndarray:
    pass


class EmbeddingBasedRepresentation(Representation):
  """
  This representation is based on the weighted-average embedding of the words in the code
  """

  def __init__(self, embedding_dict: Dict = None):
    np.random.seed(0)

    super(EmbeddingBasedRepresentation, self).__init__()

    if embedding_dict:
      logger.debug("Using a reference of the loaded Glove embedding!")
      self.embedding_dict = embedding_dict
      self.vocab = VocabEntry.from_corpus([list(embedding_dict.keys())], size=10000, freq_cutoff=1)
      return

    # load the glove embedding
    logger.info("Loading Glove Model")

    glove_file = pathlib.Path(__file__).parent.absolute().joinpath('glove.6B.300d.txt')
    try:
      with open(glove_file, "rt", encoding="utf-8") as f:
        self.embedding_dict = {}
        for line in f:
          splitLine = line.split()
          word = splitLine[0]
          embedding = np.array([float(val) for val in splitLine[1:]])
          self.embedding_dict[word] = embedding
        self.vocab = VocabEntry.from_corpus([list(self.embedding_dict.keys())], size=10000, freq_cutoff=1)
        logger.debug(f"Done. {len(self.embedding_dict)} words loaded!")
    except FileNotFoundError as e:
      logger.debug(str(e))
      logger.debug(f"Download and unzip from link http://nlp.stanford.edu/data/glove.6B.zip "
                   f"and put the file glove.6B.300d.txt here: {str(glove_file)}")
      exit(1)

  def get_embedding_representation(self, training_sents: List[List[str]]) -> None:
    all_api_words = training_sents

    # create the vocab on the training corpus
    self.vocab = VocabEntry.from_corpus(all_api_words, 10000, freq_cutoff=1)

    # get random embedding for the vocab that not in glove
    for word in self.vocab.word2id.keys():
        if word not in self.embedding_dict:
          self.embedding_dict[word] = np.random.normal(0.0, 0.3, 300)

    # measure how many words in the vocab have a pretrained embedding
    embedding_dict_keys = set(self.embedding_dict.keys())
    in_count = sum(map(lambda k: 1 if k in embedding_dict_keys else 0, self.vocab.word2id.keys()))
    logger.debug('{0} out of {1} words in the vocab are also found in the pretrained embedding'
          .format(in_count, len(self.vocab)))

    # generate frequency representation for api
    freq_matrix = np.full((len(all_api_words), len(self.vocab)), FREQ_SMOOTH)  # smoothing to avoid nan error
    for i in range(len(all_api_words)):
      for word in all_api_words[i]:
        freq_matrix[i][self.vocab[word]] += 1
    logger.debug('Generating representation matrix complete!')

    # inverse document normalization
    self.word_freqs = np.sum(freq_matrix, axis=0)
    freq_matrix = freq_matrix / self.word_freqs

    # normalize over each example
    api_norm = np.linalg.norm(freq_matrix, axis=1)
    freq_matrix = freq_matrix / api_norm[:, None]

    # use the frequency matrix to derive the embedding matrix
    rep_matrix = np.zeros((freq_matrix.shape[0], 300))  # smoothing to avoid nan error
    for i in range(len(self.vocab)):
      word = self.vocab.get_word(i)
      word_embedding = self.embedding_dict.get(word)

      if word_embedding is None:
        logger.debug('Word "{0}" is not in the pretrained embedding'.format(word))
        continue

      tmp_matrix = freq_matrix[:, i][:, None] @ word_embedding[None, :]
      rep_matrix = rep_matrix + tmp_matrix

    # re-normalize the representation matrix
    rep_matrix = rep_matrix / np.linalg.norm(rep_matrix, axis=1)[:, None]
    self.representation_matrix = rep_matrix

  def index_word(self, word):
      if word not in self.embedding_dict:
        self.embedding_dict[word] = np.random.normal(0.0, 0.3, 300)

      return self.embedding_dict[word]


class CountBasedRepresentation(Representation):
  """
  This representation has one dimension for each word in the vocab
  """

  def __init__(self):
    super(CountBasedRepresentation, self).__init__()

  def get_count_representation(self, training_sents: List[List[str]]) -> None:
    # first get the words and get them stemmed
    all_api_words = stemming(training_sents)

    # create the vocab on the training corpus
    self.vocab = VocabEntry.from_corpus(all_api_words, 10000, freq_cutoff=1)

    # generate frequency representation for api
    rep_matrix = np.full((len(all_api_words), len(self.vocab)), FREQ_SMOOTH) # smoothing to avoid nan error
    for i in range(len(all_api_words)):
      for word in all_api_words[i]:
        rep_matrix[i][self.vocab[word]] += 1
    logger.debug('Generating representation matrix complete!')

    # inverse document normalization
    self.word_freqs = np.sum(rep_matrix, axis=0)
    rep_matrix = rep_matrix / self.word_freqs

    # normalize over each example
    api_norm = np.linalg.norm(rep_matrix, axis=1)
    rep_matrix = rep_matrix / api_norm[:, None]
    self.representation_matrix = rep_matrix

  def get_api_embedding(self, api: LibraryAPI) -> ndarray:
    pass


class CodeWordCountRepresentation(CountBasedRepresentation):
  def __int__(self):
    super(CodeWordCountRepresentation, self).__init__()

  def learn_representation(self, training_apis: List[LibraryAPI]) -> None:
    # first set the names of the apis
    self.apis = training_apis

    # then get all the keywords from the code
    all_api_code_keywords = [api.get_keywords() for api in training_apis]

    # get the count based representation
    self.get_count_representation(all_api_code_keywords)


class CodeWordEmbeddingRepresentation(EmbeddingBasedRepresentation):
  def __init__(self):
    super(CodeWordEmbeddingRepresentation, self).__init__()

  def learn_representation(self, training_apis: List[LibraryAPI]) -> None:
    # first set the names of the apis
    self.apis = training_apis

    # then get all the keywords from the code
    all_api_code_keywords = [api.get_keywords() for api in training_apis]

    # all_api_code_keywords = stemming_with_vocab(all_api_code_keywords, set(self.embedding_dict.keys()))

    # manually break unknown words to subwords
    self.break_to_subwords(all_api_code_keywords)

    # get the embedding based representation
    self.get_embedding_representation(all_api_code_keywords)


class SummaryWordCountRepresentation(CountBasedRepresentation):
  def __int__(self):
    super(SummaryWordCountRepresentation, self).__init__()

  def learn_representation(self, training_apis: List[LibraryAPI]) -> None:
    # first set the names of the apis
    self.apis = training_apis

    # then get all the words in the summary (with a little preprocessing)
    def process_summary(summary: str) -> List[str]:
      return reduce(lambda x, y: x+y, list(map(lambda word: get_tokens_from_code(word), summary.split(' '))))

    all_api_summary_words = [process_summary(api.description) for api in training_apis]

    # get the count based representation
    self.get_count_representation(all_api_summary_words)


class SummaryWordEmbeddingRepresentation(EmbeddingBasedRepresentation):
  def __int__(self):
    super(SummaryWordEmbeddingRepresentation, self).__init__()

  def learn_representation(self, training_apis: List[LibraryAPI]) -> None:
    # first set the names of the apis
    self.apis = training_apis

    # then get all the words in the summary (with a little preprocessing)
    def process_summary(summary: str) -> List[str]:
      return reduce(lambda x, y: x + y, list(map(lambda word: get_tokens_from_code(word), summary.split(' '))))

    all_api_summary_words = [process_summary(api.description) for api in training_apis]

    # manually break unknown words to subwords
    self.break_to_subwords(all_api_summary_words)

    # get the count based representation
    self.get_embedding_representation(all_api_summary_words)


def test_code_keyword_based_representation():
  tf_apis = load_apis('tf')
  torch_apis = load_apis('torch')

  representation = CodeWordCountRepresentation()
  representation.learn_representation(tf_apis + torch_apis)

  tf_indices = [i for i in range(len(tf_apis))]
  torch_indices = [i + len(tf_apis) for i in range(len(torch_apis))]
  # representation.print_top_similar_apis(tf_indices, torch_indices)

  test_query_representation(representation, tf_indices, torch_indices, 'tf.keras.layers.Dense')

  # logger.debug(representation.query_similarity('tf.keras.applications.inception_resnet_v2.preprocess_input', 'torch.nn.Conv2d'))
  # logger.debug(representation.query_similarity('tf.keras.layers.Conv2D', 'torch.nn.Conv2d'))


def test_summary_word_based_representation():
  tf_apis = list(filter(lambda api: len(api.description) > 10, load_apis('tf')))
  torch_apis = list(filter(lambda api: len(api.description) > 10, load_apis('torch')))

  representation = SummaryWordEmbeddingRepresentation()
  representation.learn_representation(tf_apis + torch_apis)

  tf_indices = [i for i in range(len(tf_apis))]
  torch_indices = [i + len(tf_apis) for i in range(len(torch_apis))]
  representation.print_top_similar_apis(tf_indices, torch_indices, ignore_same_name=True)


def test_query_representation(representation: Representation, tf_idx, torch_idx, query_api_id: str):
  representation.build_query_index(tf_idx, torch_idx)
  top_k_apis = representation.query_top_k(query_api_id)
  logger.debug('top similar api for {0} is:'.format(query_api_id))
  for api in top_k_apis:
    logger.debug(api.raw_code)
    
    
def get_representation():
  # TODO: hard code here, should add options
  tf_apis = load_apis('tf')
  torch_apis = load_apis('torch')
  
  representation = CodeWordCountRepresentation()
  representation.learn_representation(tf_apis + torch_apis)
  
  tf_indices = [i for i in range(len(tf_apis))]
  torch_indices = [i + len(tf_apis) for i in range(len(torch_apis))]
  representation.build_query_index(tf_indices, torch_indices)
  
  return representation


if __name__ == '__main__':
  # test_summary_word_based_representation()
  test_code_keyword_based_representation()
