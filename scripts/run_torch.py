from argparse import *
import subprocess
import sys
import os
from tqdm import tqdm

synthesizer_path = '../synthesis/synthesizer/tf_to_torch/torch_synthesizer.py'
ranking_path = '../synthesis/synthesizer/tf_to_torch/torch_ranking.py'

parser = ArgumentParser(description='Run DL Benchmarks.')
parser.add_argument('--clean', action='store_true', help='Clean results directory.') 
parser.add_argument('--all', action='store_true', help='Run all benchmarks.')
parser.add_argument('-t', '--timeout', type=int, default=120, help='Timeout in seconds for each benchmark.') 
parser.add_argument('-s', '--specific_benchmark', default='tfidf', choices=['glove', 'tfidf', 'nospec', 'noerr'], help='Run a specific benchmark set (options glove, tfidf, nospec, noerr).') 
parser.add_argument('-m', '--memory', type=int, default=8192, help='Memory limit in MB.') 
cmd_args = parser.parse_args()
memory = cmd_args.memory

benchmarks_to_run = {'tfidf': False, 'glove': False, 'noerr': False, 'nospec': False}

if cmd_args.clean:
    for k, v in benchmarks_to_run.items():
        results_directory = f'./results/tf_to_torch/soar_{k}/'
        files_path = list(map(lambda x: results_directory + x, os.listdir(results_directory)))
        for path in files_path: os.remove(path)
    sys.exit(0)
            
if cmd_args.all:
    benchmarks_to_run = {k: True for k, v in benchmarks_to_run.items()}
    

benchmarks_to_run[cmd_args.specific_benchmark] = True


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

              
option_set = {'tfidf': '-spec -errmsg',
              'glove': '-spec -errmsg -glove',
              'nospec': '-errmsg',
              'noerr': '-spec'}


for k, v in benchmarks_to_run.items():
    if v:
        results_directory = f'./results/tf_to_torch/soar_{k}'
        print(f'Running {k}')
        for i in tqdm(range(len(benchmarks))):
            benchmark = benchmarks[i]
            with open(f"{results_directory}/{benchmark}.out", "a+") as f:
                subprocess.run(f"runsolver -W {cmd_args.timeout} -w /dev/null --rss-swap-limit {memory} python3 {synthesizer_path} -b {benchmark} {option_set[k]}", shell=True, check=True, stdout=f, stderr=f)
                subprocess.run(f"runsolver -W {cmd_args.timeout} -w /dev/null --rss-swap-limit {memory} python3 {ranking_path} -b {benchmark} {option_set[k]}", shell=True, check=True, stdout=f, stderr=f)

print("Finished!") 
print("Logs can be found in /home/soar/Documents/SOAR/scripts/results/tf_to_torch")
print("To generate tables and plots run: python3 torch_data.py")

