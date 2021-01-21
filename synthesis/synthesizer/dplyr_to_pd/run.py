import subprocess

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

for benchmark in benchmarks:
    print(benchmark)
    with open(f"results/{benchmark}.txt", "a+") as f:
        subprocess.run(f"runsolver -W 150 -w /dev/null python3 pd_synthesizer.py -b {benchmark}", shell=True, check=True, stdout=f, stderr=f)
