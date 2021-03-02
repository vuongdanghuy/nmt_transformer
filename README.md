# English-Vietnamese Machine Translation using Transformer

## Data:
This project uses 2 dataset:
1. IWSLT'15 English-Vietnamese from Stanford NLP group. You can download it [here](https://nlp.stanford.edu/projects/nmt/). This dataset contains 133317 sentence pairs in training set and 1268 sentence pair in testset.
2. VLSP 2020 dataset contain English-Vietnamese sentence pair in domain NEWS. You can download it [here](https://slp.vinbigdata.org/). Training data consists of two corpora: Parallel corpora, which are in UTF-8 plaintexts, 1-to-1 sentence aligned, one sentence per line, and include in-domain NEWS dataset of size 20k samples with 80% in the training set, 10% in the dev set and 10% in the test set; and out-of-domain parallel datasets roughly of size 4M samples, such as openSub (3.5M), ted-like (55k), evbcorpus (45k), wiki-alt (20k), and basic (8.8k) datasets. In this project we only use evbcorpus, in-domain NEWS and wiki-alt dataset.

Data from VLSP 2020 dataset will be appended with IWSLT training data to create a larger training set for training model. In testing phase, BLEU score on IWSLT testing set will be used to benchmark.

## Model:
In this project we build a transformer based model. Original implementation of baseline model can be found [here](https://pbcquoc.github.io/transformer/). Baseline has 6 layers for encoder and decoder. We also train and test a new model with 8 layers.

## Training and Testing:
1.  Tokenize with spaCy: model 'en' for English, spacy_vi_model (from https://github.com/trungtv/vivi_spacy)
2.  Vectorize using torchtext
3.  Positional Embedding: Concatenate positional embedding matrix to input
4.  Label Smoothing: Distribute correct class in one-hot vector into the remaining class to reduce overfit
5.  Train each version for 30 epochs
6.  Testing using beam search and nltk wordnet

## Result:
| Model | Training set | BLEU Score (%) |
| :---: | :---: | :---: |
| Baseline, 6 layers | IWSLT | 25.16 |
| Modified, 8 layers | IWSLT | 25.95 |
| Modified, 8 layers | IWSLT + VLSP 2020 | 27.69 |
| Tensor2tensor | IWSLT | 29.44 |
| Google API | | 31.69 |
