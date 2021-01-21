from commons.interfaces import ApiMatching
from commons.library_api import LibraryAPI, load_apis, get_tokens_from_code

ground_truth_mappings = [
    ('tf.keras.layers.Conv2D', 'torch.nn.Conv2d'),
    ('tf.keras.layers.ReLU', 'torch.nn.ReLU'),
    ('tf.keras.layers.MaxPool2D', 'torch.nn.MaxPool2d'),
    ('tf.keras.layers.Flatten', 'torch.nn.Flatten'),
    ('tf.keras.layers.Dense', 'torch.nn.Linear'),
    ('tf.keras.layers.Softmax', 'torch.nn.Softmax'),
    ('tf.keras.layers.Conv2DTranspose', 'torch.nn.ConvTranspose2d'),
    ('tf.keras.layers.BatchNormalization', 'torch.nn.BatchNorm2d'),
    ('tf.keras.layers.ZeroPadding2D', 'torch.nn.ZeroPad2d'),
    ('tf.keras.layers.GlobalAveragePooling2D', 'torch.nn.AvgPool2d'),
    ('tf.keras.layers.LeakyReLU', 'torch.nn.LeakyReLU'),
    ('tf.keras.activations.tanh', 'torch.nn.Tanh'),
    ('tf.keras.layers.LSTM', 'torch.nn.LSTM'),
    ('tf.keras.layers.Embedding', 'torch.nn.Embedding'),
    ('tf.keras.layers.GRU', 'torch.nn.GRU'),
]

def mapping_ranking_eval():
    pass

def single_api_test(matcher: ApiMatching):
    pass


if __name__ == '__main__':
    # load the two libraries
    tf_apis = load_apis('tf')
    torch_apis = load_apis('torch')

    # build dictionary of api names
    tf_apis_dict = dict(map(lambda x: (x.id, x), tf_apis))
    torch_apis_dict = dict(map(lambda x: (x.id, x), torch_apis))

    for s, t in ground_truth_mappings:
        if s not in tf_apis_dict:
            print(s)
        if t not in torch_apis_dict:
            print(t)

    src_tgt_api_tuples = list(map(lambda x: (tf_apis_dict[x[0]], torch_apis_dict[x[1]]), ground_truth_mappings))

    tfidf_matcher = ApiMatching('tf', 'torch', k=1000, use_embedding=False, use_description=False)
    embedding_matcher = ApiMatching('tf', 'torch', k=1000, use_embedding=True, use_description=False)

    result1 = tfidf_matcher.query_for_new_api('tf.keras.layers.Conv2D()')
    result2 = embedding_matcher.query_for_new_api('tf.keras.layers.Conv2D()')

    rankings = []
    for src, tgt in src_tgt_api_tuples:

        tfidf_ranking = -1
        for i, api_prob in enumerate(tfidf_matcher.api_matching(src)):
            if api_prob[0].id == tgt.id:
                tfidf_ranking = i
                break

        embedding_ranking = -1
        for i, api_prob in enumerate(embedding_matcher.api_matching(src)):
            if api_prob[0].id == tgt.id:
                embedding_ranking = i
                break
        rankings.append((src.id, tgt.id, tfidf_ranking, embedding_ranking))

    for t in rankings:
        print(t)



