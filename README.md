# NGS
Source code for AACL-IJCNLP 2020 paper "Neural Gibbs Sampling for Joint Event Argument Extraction".

## Requirements

- python == 3.6.9
- pytorch == 0.6.1
- numpy == 1.15.2
- sklearn == 0.20.2
- pytorch-pretrained-bert == 0.4.0
- nltk
- tqdm

We use the ACE2005 (LDC2006T06) and TAC KBP 2016 (LDC2017E05) as our benchmarks. Due to the LDC license limitation, we cannot  share the datasets.

For NGS (CNN), the 100-dim Glove word vectors pre-trained with Wikipedia 2014+Gigaword 5 is used. 

## Usage

### NGS (CNN)

The codes are in the `NGS-DMCNN` folder.

1. run `input.py` to preprocess the data.
2. run `trigger.py` and `argument.py` to train and test the prior models for the ED and EAE.
3. run `conditional.py` to train and test the conditional neural model.
4. run `Gibbs_an.py` to run the Gibbs sampling + Simulated annealing.
5. hyper-parameters and data paths are specified in `constant.py`.

### NGS (BERT)

The codes are in the `NGS-DMBERT` folder. 

1. run `input_bert_role.py` to preprocess the data.
2. run `trigger_bert.py` and `argument_bert.py` to train and test the prior models for the ED and EAE.
3. run `conditional_bert.py` to train and test the conditional neural model.
4. run `Gibbs_an_bert.py` to run the Gibbs sampling + Simulated annealing.
5. hyper-parameters and data paths are specified in `constant.py`.

## Cite

If the codes help you, please cite the following paper:

**Neural Gibbs Sampling for Joint Event Argument Extraction.** *Xiaozhi Wang, Shengyu Jia, Xu Han, Zhiyuan Liu, Juanzi Li, Peng Li, Jie Zhou.* AACL-IJCNLP 2020.
