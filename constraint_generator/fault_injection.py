import json
import tensorflow as tf
import random
import numpy as np


# generate random numbers
random_dim = random.sample(range(1, 10), 3)
random_matrix = np.random.rand(random_dim[0], random_dim[1], random_dim[2])
tensor = tf.constant(random_matrix)

with open('C:\\Projects\\api-representation-learning\\crawler\\preprocessed_tf_docs.json') as f:
  data = json.load(f)


for api_call in data:
    tensorArg = False
    params = api_call['code-info']['parameters']
    for param in params:
        if 'tensor' in param['type']:
            tensorArg = True
    if tensorArg:
        print(params)
        # eval(api_call + '(' + )
