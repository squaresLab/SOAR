from argparse import *
import subprocess
import sys
import os
from tqdm import tqdm


synthesizer_path = '../synthesis/synthesizer/dplyr_to_pd/pd_synthesizer.py'
ranker_path =  '../synthesis/synthesizer/dplyr_to_pd/pd_ranking.py'

parser = ArgumentParser(description='Run DL Benchmarks.')
parser.add_argument('--clean', action='store_true', help='Clean results directory.') 
parser.add_argument('-t', '--timeout', type=int, default=120, help='Timeout in seconds for each benchmark.') 
parser.add_argument('-m', '--memory', type=int, default=8192, help='Memory limit in MB.') 
cmd_args = parser.parse_args()
results_directory = f'./results/dplyr_to_pd/'
memory = cmd_args.memory

if cmd_args.clean:

    files_path = list(map(lambda x: results_directory + x, os.listdir(results_directory)))
    for path in files_path: os.remove(path)
    sys.exit(0)



benchmarks = ['Q2', 'Q4',
              'Q6', 'Q7',
              'Q9', 'Q54',
              'Q10', 'Q11',
              'Q12', 'Q13',
              'Q14', 'Q22',
              'Q23', 'Q28',
              'Q34', 'Q37',
              'Q40', 'Q46',
              'Q50', 'Q51']

print(f'Running dplyr-to-pd')
for i in tqdm(range(len(benchmarks))):
    benchmark = benchmarks[i]
    with open(f"{results_directory}/{benchmark}.out", "a+") as f:
        subprocess.run(f"runsolver -W {cmd_args.timeout} -w /dev/null --rss-swap-limit {memory} python3 {synthesizer_path} -b {benchmark}", shell=True, check=True, stdout=f, stderr=f)
        subprocess.run(f"runsolver -W {cmd_args.timeout} -w /dev/null --rss-swap-limit {memory} python3 {ranker_path} -b {benchmark}", shell=True, check=True, stdout=f, stderr=f)

print("Finished!") 
print("Logs can be found in /home/soar/Documents/SOAR/scripts/results/dplyr_to_pd")
print("To generate tables and plots run: python3 pd_data.py")
