# Tree-Structured LSTMs for Semantic Similarity

### Based on this repo - https://github.com/dasguptar/treelstm.pytorch

This is a [PyTorch](http://pytorch.org/) implementation of Tree-LSTM as described in the paper [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075) by Kai Sheng Tai, Richard Socher, and Christopher Manning. On the semantic similarity task using the SICK dataset, this implementation reaches:
 - Pearson's coefficient: `0.8492` and MSE: `0.2842` using hyperparameters `--lr 0.010 --wd 0.0001 --optim adagrad --batchsize 25`
 - Pearson's coefficient: `0.8674` and MSE: `0.2536` using hyperparameters `--lr 0.025 --wd 0.0001 --optim adagrad --batchsize 25 --freeze_embed`
 - Pearson's coefficient: `0.8676` and MSE: `0.2532` are the numbers reported in the original paper.
 - Known differences include the way the gradients are accumulated (normalized by batchsize or not).

### Requirements
- Python (tested on **3.6.5**, should work on **>=2.7**)
- Java >= 8 (for Stanford CoreNLP utilities)
- Other dependencies are in `requirements.txt`
Note: Currently works with PyTorch 0.4.0. Switch to the `pytorch-v0.3.1` branch if you want to use PyTorch 0.3.1.

### Usage
Before delving into how to run the code, here is a quick overview of the contents:
 - Use the script `fetch_and_preprocess.sh` to download the [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools), [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) and [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml), and [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840) -- **Warning:** this is a 2GB download!), and additionally preprocees the data, i.e. generate dependency parses using [Stanford Neural Network Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml).
 - `main.py`does the actual heavy lifting of training the model and testing it on the SICK dataset. For a list of all command-line arguments, have a look at `config.py`.
     - The first run caches GLOVE embeddings for words in the SICK vocabulary. In later runs, only the cache is read in during later runs.
     - Logs and model checkpoints are saved to the `checkpoints/` directory with the name specified by the command line argument `--expname`.

Next, these are the different ways to run the code here to train a TreeLSTM model.
#### Local Python Environment
If you have a working Python3 environment, simply run the following sequence of steps:
```
- bash fetch_and_preprocess.sh
- pip install -r requirements.txt
- python main.py
```
#### Pure Docker Environment
If you want to use a Docker container, simply follow these steps:
```
- docker build -t treelstm .
- docker run -it treelstm bash
- bash fetch_and_preprocess.sh
- python main.py
```
#### Local Filesystem + Docker Environment
If you want to use a Docker container, but want to persist data and checkpoints in your local filesystem, simply follow these steps:
```
- bash fetch_and_preprocess.sh
- docker build -t treelstm .
- docker run -it --mount type=bind,source="$(pwd)",target="/root/treelstm.pytorch" treelstm bash
- python main.py
```
**NOTE**: Setting the environment variable OMP_NUM_THREADS=1 usually gives a speedup on the CPU. Use it like `OMP_NUM_THREADS=1 python main.py`. To run on a GPU, set the CUDA_VISIBLE_DEVICES instead. Usually, CUDA does not give much speedup here, since we are operating at a batchsize of `1`.


### Acknowledgements
Thanks to Riddhiman Dasgupta

### License
MIT
