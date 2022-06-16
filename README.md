# Memorizing All for Implicit Discourse Relation Recognition

This is the code for paper:

Memorizing All for Implicit Discourse Relation Recognition (https://dl.acm.org/doi/fullHtml/10.1145/3485016)

## Usage

We use the processed data from https://github.com/cgpotts/pdtb2.

Put the `pdtb2.csv` to `./data/raw/` first.

Edit the paths of pre-trained word embedding file and ELMo files in `config.py`.

Then prepare the data:

        bash ./prepare_data.sh

For training and evaluating:

        python main.py func splitting

`func` can be `train` or `eval`, and `splitting` is 1 or 2 or 3,
1 for PDTB-Lin 11-way classification, 2 for PDTB-Ji 11-way classification and 3 for 4-way classification.

For example:

        python main.py train 1

means training for PDTB-Lin 11-way classification.

        python main.py eval 2

means evaluating with pre-trained parameters for PDTB-Ji 11-way classification.

## Requirements

        python == 3.6.7
        nltk == 3.3.0
        numpy == 1.15.4
        gensim == 3.4.0
        scikit-learn == 0.20.1
        pytorch == 1.0.1
        allennlp == 0.8.2
        tensorboardX == 1.2
