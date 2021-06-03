# Synthesis for Open-Source API Refactoring (SOAR)


## How to set up your own environment:

https://colab.research.google.com/drive/1b9Ipn4nya2Oxi9RgivomS1QLbyeu_nTU




## Benchmarks:

The DL benchmarks can be found at
`api-learning-representation/autotesting/benchmarks`, and the 
data wrangling benchmarks at
`api-learning-representation/synthesis/synthesizer/dplyr_to_pd/dplyr`.


## Reproducing all results:

There are two different scripts used to generate the results presented in the paper:

1. run_torch.py (runs the TensorFlow-to-PyTorch benchmarks)
2. run_pd.py (runs the dplyr-to-pandas benchmarks).


To reproduce all results start by opening a terminal and go to the *scripts* folder:

    cd /home/soar/Documents/SOAR/scripts

To execute all benchmarks with the different versions of SOAR, run the
following commands:

    python3 run_torch.py --all -t timeout -m memory_limit
    python3 run_pd.py -t timeout -m memory_limit

where `timeout` is the time limit for each benchmark (3600 in the paper), and
`memory_limit` is the memory
limit in MB for each benchmark (65536MB / 64GB in the paper). It is possible to
partially replicate most of the results using significantly lower time and
memory limits for each benchmark (for instance, 300 seconds and 8GB of RAM).

The previous two commands generate logs in the
`/home/soar/Documents/SOAR/scripts/results` that are used to generate the
tables and figures shown in the paper. To generate the tables and the data
depicted in the paper's figures, run the following commands:

    python3 torch_data.py
    python3 pd_data.py

The commands create files in
`/home/soar/Documents/SOAR/scripts/figures`  and
`/home/soar/Documents/SOAR/scripts/tables` with identifiers matching the
figures and tables in the paper (e.g., table1.png --> Table I)


We would like to note that we do not count the loading of the dependencies 
(including loading the GloVe model, TensorFlow, PyTorch, etc) towards the execution 
time of SOAR. In theory, this step could be done only once, as it is equal for all instances.
Therefore, the execution times presented in the paper's results only show the time that is 
effectively necessary to migrate the code (after SOAR's dependencies have been loaded).

## Preparing results directories

Before re-running results, it is necessary to delete the previous data from the
results, figures, and tables directories. To clean the results folder for the
dplyr-to-pandas task, run the following commands:

    python3 run_pd.py --clean
    python3 pd_data.py --clean

Similarly, to clean the results directory for tf-to-torch task, run the
following commands:

    python3 run_torch.py --clean
    python3 torch_data.py --clean

## Running a specific version of SOAR

It is possible to run a specific version of SOAR for the tf-to-torch
benchmarks. In the paper, we study four different versions of SOAR on the
tf-to-torch benchmarks: 

1. SOAR (tfidf)
2. SOAR with GloVe (glove)
3. SOAR without error message understanding (noerr)
4. SOAR without specifications (nospec)

To run a specific version of SOAR with the tf-to-torch benchmarks, run the
following commands:

    python3 run_torch.py -s soar_version -t timeout -m memory_limit

where `soar_version` is a string indicating the version of SOAR to be run
(either 'tfidf', or 'glove', or 'noerr', or 'nospec')

This command create logs inside the directory
`/home/soar/Documents/SOAR/scripts/results/tf_to_torch/soar_<soar_version>`

Instructions on how to interpret these logs are below.

## Running specific benchmarks

To execute specific benchmarks for the deep learning task use the following command:

    cd /home/soar/Documents/SOAR/synthesis/synthesizer/tf_to_torch/
    python3 torch_synthesizer.py -b <benchmark_name> -t <timeout> -b <memory_limit>

where <benchmark_name> is the benchmark name (found in Table III of the paper),
<timeout> is the timeout in seconds and <memory_limit> is the memory limit in MB. 
The source code of the benchmarks can be found in `/home/soar/Documents/SOAR/autotesting/benchmarks/`. 
It is recommended to use at least 8GB of RAM to produce reasonable results.


For instance, in the benchmark *conv_pool_softmax.py*, we have the following source code:

```
(...)
class TFConv(Model):
    def __init__(self, **kwargs):
        super(TFConv, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.conv = tf.keras.layers.Conv2D(3, 3, strides=(1, 1))
        self.pool = tf.keras.layers.MaxPool2D(2, 2)
        self.flatten = tf.keras.layers.Flatten()
        self.softmax = tf.keras.layers.Softmax()
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def call(self, x):
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.softmax(x)
        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
(...)
```
Running the commands:

    cd /home/soar/Documents/SOAR/synthesis/synthesizer/tf_to_torch/
    python3 torch_synthesizer.py -b conv_pool_softmax -t 60 -b 8192

Yields the following output:

```
[info] [2021-01-15 00:37:10.947203] Success!
[info] [2021-01-15 00:37:10.947732] Avg Rank 1.0
[info] [2021-01-15 00:37:10.985732] Synthesis time: 2.0532901287078857
[info] [2021-01-15 00:37:10.985955] self.before_0 = lambda x: x.permute(0,3,2,1)
[info] [2021-01-15 00:37:10.985955] self.var70 = torch.nn.Conv2d(1,3,(3,3),stride=(1,1),padding=(0,0))
[info] [2021-01-15 00:37:10.985955] self.var74 = torch.nn.MaxPool2d((2,2),stride=(2,2),padding=(0,0))
[info] [2021-01-15 00:37:10.985955] self.before_3 = lambda x: x.permute(0,3,2,1)
[info] [2021-01-15 00:37:10.985955] self.var115 = lambda x: torch.flatten(x,1)
[info] [2021-01-15 00:37:10.985955] self.var116 = torch.nn.Softmax(dim=1)
```

Avg Rank indicates the average rank of the correct API matching (as described in the paper).
Synthesis time is the total time used to perform the migration except for the dependencies loading time.
The migrated code is printed after the synthesis time.

Conversely, to run a benchmark for the dplyr to pandas task execute the following command:

    cd /home/soar/Documents/SOAR/synthesis/synthesizer/dplyr_to_pd/
    python3 pd_synthesizer.py -b <benchmark_name>

The benchmark names can be found at \texttt{/Documents/SOAR/synthesis/synthesizer/dplyr\_to\_pd/dplyr}.



