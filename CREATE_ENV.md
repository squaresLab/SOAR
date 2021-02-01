
### Download and install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
sha256sum Anaconda3-2020.11-Linux-x86_64.sh

### Create a conda environment

    conda create --name SOAR python==3.8.5
    conda activate SOAR

### Install conda packages

    conda install tensorflow==2.2.0
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    conda install matplotlib
    conda install nltk

### Install pip packages

    pip3 install z3-solver
    pip3 install click
    pip3 install wordfreq
    pip3 install fuzzywuzzy
    pip3 install transformers

### Open python shell and install nltk dependencies

    python3
    import nltk
    
    nltk.download('punkt')
    ltk.download('averaged_perceptron_tagger')
    exit()

### Download and unzip glove model

    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    mv glove.6B.300d.txt (PATH_TO_SOAR)/mapping/

### Include SOAR in your python path

    export PYTHONPATH=PATH_TO_SOAR   (e.g. export PYTHONPATH=/home/blue/Documents/SOAR)

### Run SOAR!

    python3 synthesis/synthesizer/tf_to_torch/torch_synthesizer.py -b conv_for_text -debug -errmsg        






