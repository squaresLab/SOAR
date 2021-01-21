import pandas as pd
import os
import json
import csv

result = pd.DataFrame()
with open('C:\\Projects\\api-representation-learning\\crawler\\preprocessed_tf_docs.json') as f:
    data = json.load(f)
id_list = []
for i in data:
    id = i['id']
    if ":" in id:
        id = id.split(':')[0]
    # if "." in id:
    #     id_split = id.split('.')
    #     id = id_split[len(id_split) - 1]
    #     id = '.' + id
    id = id.replace('tf.', '')
    if id not in id_list:
        id_list.append(id)
id_dict = {id: 0 for id in id_list}
print(id_list)

test_count = 0
directory = 'C:\\Projects\\api-representation-learning\\github_crawler\\tf_pyfiles'
api_count = 0
benchmark_counter = 0
total_counter = 0
benchmarks = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
totes = 0
file_counter = 0

for filename in os.listdir(directory):
    # test_count += 1
    # if test_count > 10:
    #     break
    file_bool = False
    if filename.endswith(".csv"):
        print(filename)
        df = pd.read_csv(directory + '\\' + filename)
        code_list = df['code'].to_list()
        for code in code_list:
            if file_bool:
                break
            split_code = code.split('\\n')
            for idx, nline in enumerate(split_code):
                if file_bool:
                    break
                totes += 1
                # for api in id_list:
                #     if api in nline:
                #         id_dict[api] += 1

                if 'tensorflow' in nline or 'tf' in nline:
                    total_counter += 1
                if 'Conv' in nline \
                        or 'ReLU' in nline \
                        or 'MaxPool2D' in nline \
                        or 'Flatten' in nline \
                        or 'Dense' in nline \
                        or 'Softmax' in nline \
                        or 'Reshape' in nline \
                        or 'Conv2DTranspose' in nline \
                        or 'BatchNormalization' in nline \
                        or 'Activation' in nline \
                        or 'ZeroPadding2D' in nline \
                        or 'MaxPooling2D' in nline \
                        or 'GlobalAveragePooling2D' in nline \
                        or 'leaky_relu' in nline \
                        or 'LeakyReLU()' in nline \
                        or 'tanh' in nline \
                        or 'LSTM' in nline \
                        or 'Embedding' in nline \
                        or 'GRU' in nline \
                        or 'Activation' in nline \
                        or 'ZeroPadding2D' in nline:
                    file_counter += 1
                    file_bool = True
                    benchmark_counter += 1
                # if 'Conv' in nline:
                #     benchmarks[0] += 1
                # if 'ReLU' in nline:
                #     benchmarks[1] += 1
                # if 'MaxPool2D' in nline:
                #     benchmarks[2] += 1
                # if 'Flatten' in nline:
                #     benchmarks[3] += 1
                # if 'Dense' in nline:
                #     benchmarks[4] += 1
                # if 'Softmax' in nline:
                #     benchmarks[5] += 1
                # if 'Reshape' in nline:
                #     benchmarks[6] += 1
                # if 'BatchNormalization' in nline:
                #     benchmarks[7] += 1
                # if 'Activation' in nline:
                #     benchmarks[8] += 1
                # if 'ZeroPadding2D' in nline:
                #     benchmarks[9] += 1
                # if 'MaxPooling2D' in nline:
                #     benchmarks[10] += 1
                # if 'GlobalAveragePooling2D' in nline:
                #     benchmarks[11] += 1
                # if 'leaky_relu' in nline:
                #     benchmarks[12] += 1
                # if 'LeakyReLU' in nline:
                #     benchmarks[13] += 1
                # if 'tanh' in nline:
                #     benchmarks[14] += 1
                # if 'LSTM' in nline:
                #     benchmarks[15] += 1
                # if 'Embedding' in nline:
                #     benchmarks[16] += 1
                # if 'GRU' in nline:
                #     benchmarks[17] += 1
                # if 'Activation' in nline:
                #     benchmarks[18] += 1
                # if 'ZeroPadding2D' in nline:
                #     benchmarks[19] += 1
                # if 'Conv2DTranspose' in nline:
                #     benchmarks[20] += 1

# print(benchmark_counter)
# print(total_counter)
print(file_counter)
# print(benchmarks)
#print(totes)

# with open('C:\\Projects\\api-representation-learning\\github_crawler\\api_freq.csv', 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     for key, value in id_dict.items():
#         key = key.replace('.', '')
#         writer.writerow([key, value])
