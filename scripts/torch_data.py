from argparse import *
import subprocess
import os
import re
import sys


parser = ArgumentParser(description='Create dplyr-to-pandas figures and tables.')
parser.add_argument('--clean', action='store_true', help='Clean dplyr-to-pandas figures and tables.') 
cmd_args = parser.parse_args()

if cmd_args.clean:
    try:
        os.remove('/home/soar/Documents/SOAR/scripts/tables/table2.csv')
        os.remove('/home/soar/Documents/SOAR/scripts/tables/table3.csv')
    except:
        print('Nothing to remove!')
    sys.exit(0)
    


times = {'tfidf': {},
         'glove': {},
         'nospec': {},
         'noerr': {}}
         
avg_rankings = {'tfidf': {},
         'glove': {},
         'nospec': {},
         'noerr': {}}

benchmarks = ['conv_pool_softmax',
              'img_classifier',
              'three_linear',
              'embed_conv1d_linear',
              'word_autoencoder',
              'gan_discriminator',
              'two_conv',
              'img_autoencoder',
              'alexnet',
              'gan_generator',
              'lenet',
              'tutorial', 
              'conv_for_text', 
              'vgg11',
              'vgg16',
              'vgg19',
              'densenet_part1',
              'densenet_part2',
              'densenet_conv_block',
              'densenet_trans_block']

         
for k, v in times.items():
    results_directory = f'./results/tf_to_torch/soar_{k}/'
    files_path = list(map(lambda x: results_directory + x, os.listdir(results_directory)))

    data = {}
    for path in files_path:
        with open(path, "r+") as f:
            data[path] = f.read()


    successful_bench = list(filter(lambda x: data[x].find('Success!') != -1, data.keys()))
    successful_bench = {bench: data[bench] for bench in successful_bench}
    avg_ranks_bench = list(filter(lambda x: data[x].find('Avg Rank') != -1, data.keys()))
    avg_ranks_bench = {bench: data[bench] for bench in avg_ranks_bench}


    for benchmark in benchmarks:
        # we only care about successful benchmarks for now...
        benchmark_data = successful_bench.get(results_directory + f'{benchmark}.out', None)
        benchmark_data_rank = avg_ranks_bench.get(results_directory + f'{benchmark}.out', None)
        if benchmark_data is not None:
            synthesis_time = float(re.search(r'Synthesis time: ([0-9]*\.*[0-9]*)', benchmark_data).group(1))
            times[k][benchmark] = f'{synthesis_time:.2f}'
        if benchmark_data_rank is not None:
            avg_ranking = float(re.search(r'Avg Rank ([0-9]*\.*[0-9]*)', benchmark_data_rank).group(1))
            avg_rankings[k][benchmark] = f'{avg_ranking:.1f}'


table_2 = 'Name,SOAR,SOAR w/o Specs,SOAR w/o Err. Msg' + os.linesep
for benchmark in benchmarks:
    table_2 += f'{benchmark},'
    for method in ['tfidf', 'nospec', 'noerr']:
        table_2 += f'{times[method].get(benchmark, "timeout")},'
    table_2 = table_2[:-1] + os.linesep
    
with open('/home/soar/Documents/SOAR/scripts/tables/table2.csv', 'a+') as f:
    f.write(table_2)


table_3 = 'Name,SOAR W/ TF-IDF,SOAR W/ TF-IDF,SOAR w/ Tfidf-GloVe, SOAR w/ Tfidf-GloVe' + os.linesep
table_3 += 'Name,Time(s),Avg. Ranking,Time(s),Avg. Ranking,' + os.linesep
for benchmark in benchmarks:
    table_3 += f'{benchmark},'
    for method in ['tfidf', 'glove']:
        table_3 += f'{times[method].get(benchmark, "timeout")},'
        table_3 += f'{avg_rankings[method].get(benchmark, "manual")},'
    table_3 = table_3[:-1] + os.linesep
    
with open('/home/soar/Documents/SOAR/scripts/tables/table3.csv', 'a+') as f:
    f.write(table_3)

print("Finished!")
print("Tables can be found in /home/soar/Documents/SOAR/scripts/tables/")
