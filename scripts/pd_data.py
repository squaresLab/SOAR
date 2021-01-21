from argparse import *
import subprocess
import os
import re
import sys
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

parser = ArgumentParser(description='Create dplyr-to-pandas figures and tables.')
parser.add_argument('--clean', action='store_true', help='Clean dplyr-to-pandas figures and tables.') 
cmd_args = parser.parse_args()

if cmd_args.clean:
    try:
        os.remove('/home/soar/Documents/SOAR/scripts/figures/figure6.png')
        os.remove('/home/soar/Documents/SOAR/scripts/figures/figure7.png')
    except:
        print('Nothing to remove!')
    sys.exit(0)

times = {}
         
avg_rankings = {'tfidf': {}, 'glove': {}}

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

results_directory = f'./results/dplyr_to_pd/'
files_path = list(map(lambda x: results_directory + x, os.listdir(results_directory)))

data = {}
for path in files_path:
    with open(path, "r+") as f:
        data[path] = f.read()


successful_bench = list(filter(lambda x: data[x].find('Success!') != -1, data.keys()))
successful_bench = {bench: data[bench] for bench in successful_bench}
ranked_bench = list(filter(lambda x: data[x].find('Avg Rank') != -1, data.keys()))
ranked_bench = {bench: data[bench] for bench in ranked_bench}

for benchmark in benchmarks:
    # we only care about successful benchmarks for now...
    benchmark_data = successful_bench.get(results_directory + f'{benchmark}.out', None)
    benchmark_data_rank = ranked_bench.get(results_directory + f'{benchmark}.out', None)
    if benchmark_data is not None:
        synthesis_time = float(re.search(r'Synthesis time: ([0-9]*\.*[0-9]*)', benchmark_data).group(1))
        times[benchmark] = float(f'{synthesis_time:.2f}')
    if benchmark_data_rank is not None:
        avg_ranking = float(re.search(r'Avg Rank GLOVE ([0-9]*\.*[0-9]*)', benchmark_data_rank).group(1))
        avg_rankings['glove'][benchmark] = float(f'{avg_ranking:.1f}')
        avg_ranking = float(re.search(r'Avg Rank TFIDF ([0-9]*\.*[0-9]*)', benchmark_data_rank).group(1))
        avg_rankings['tfidf'][benchmark] = float(f'{avg_ranking:.1f}')

y = sorted(list(times.values()))
x = [i+1 for i in range(len(y))]


# Figure 6
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_yticks([1, 100, 10000])
ax.set_ylim(bottom=1, top=10000)
ax.set_xticks([1,5,10,15,20])
ax.set_ylabel("Time (seconds)")
ax.set_xlim(left=1, right=20)
ax.set_xlabel("Instances")
ax.scatter(x, y, s=10)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig('/home/soar/Documents/SOAR/scripts/figures/figure6.png')
plt.clf()

# Figure 7


tf_bar = [0,0,0]
for el in list(avg_rankings['tfidf'].values()):
    if el < 10:
        tf_bar[0] += 1
    elif el < 100:
        tf_bar[1] += 1
    else:
        tf_bar[2] += 1  

glove_bar = [0,0,0]
for el in list(avg_rankings['glove'].values()):
    if el < 10:
        glove_bar[0] += 1
    elif el < 100:
        glove_bar[1] += 1
    else:
        glove_bar[2] += 1  
        
fig, ax = plt.subplots()
ind = np.arange(3)  # the x locations for the groups
width = 0.25
bar1=ax.bar(ind - width/2, tf_bar, width, label='tfidf')
bar2=ax.bar(ind + width/2, glove_bar, width, label='glove')
ax.set_xticklabels(('0-10','11-100', '101+'))
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(bar1)
autolabel(bar2)
ax.legend()
plt.savefig('/home/soar/Documents/SOAR/scripts/figures/figure7.png')

#with open('/home/soar/Documents/SOAR/scripts/figures/figure7.csv', 'a+') as f:
#    f.write(','.join(map(str,)))
#    f.write(os.linesep)
#    f.write(','.join(map(str,sorted(list(avg_rankings['glove'].values())))))
print("Finished!")
print("Tables can be found in /home/soar/Documents/SOAR/scripts/figures/")
