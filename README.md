## NaturalProofs: Mathematical Theorem Proving in Natural Language

[NaturalProofs: Mathematical Theorem Proving in Natural Language](https://cs.nyu.edu/~welleck/welleck2021naturalproofs.pdf)\
Sean Welleck, Jiacheng Liu, Ronan Le Bras, Hannaneh Hajishirzi, Yejin Choi, Kyunghyun Cho

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4632538.svg)](https://doi.org/10.5281/zenodo.4632538)


This repo contains:

- The **NaturalProofs Dataset**
- **Tokenized task data** for mathematical reference retrieval and generation.
- **Preprocessing** NaturalProofs and the task data.
- **Training** and **evaluation** for mathematical reference retrieval and generation.
- **Pretrained models** for mathematical reference retrieval and generation.

Please cite our work if you found the resources in this repository useful:
```
@article{welleck2021naturalproofs,
  title={NaturalProofs: Mathematical Theorem Proving in Natural Language}, 
  author={Sean Welleck and Jiacheng Liu and Ronan Le Bras and Hannaneh Hajishirzi and Yejin Choi and Kyunghyun Cho},
  year={2021},
  eprint={2104.01112},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}
```

## Quick download
To download and unpack NaturalProofs, use:
```
pip install gdown
python download.py --naturalproofs --savedir /path/to/savedir
```

To download and unpack all files that we describe below, use:
```
python download.py --naturalproofs --tokenized --checkpoint --other --savedir /path/to/savedir
```
This creates the following file structure:
```
{savedir}/data   # contains NaturalProofs base data (.json files) and tokenized task data (.pkl files)
{savedir}/ckpt   # contains pretrained model checkpoints
{savedir}/other  # contains precomputed files for evaluation (ref encodings, etc.)
```


## NaturalProofs Dataset
We provide the NaturalProofs Dataset (JSON per domain):

| NaturalProofs Dataset [[zenodo](https://doi.org/10.5281/zenodo.4632538)]| Domain|
|-|-|
|[naturalproofs_proofwiki.json](https://zenodo.org/record/4902289/files/naturalproofs_proofwiki.json?download=1)|ProofWiki|
|[naturalproofs_stacks.json](https://zenodo.org/record/4902289/files/naturalproofs_stacks.json?download=1)|Stacks|
|[naturalproofs_trench.json](https://zenodo.org/record/4902202/files/naturalproofs_trench.json?download=1)|Real Analysis textbook|
|[naturalproofs_stein.json](https://zenodo.org/record/4902289/files/naturalproofs_stein.py?download=1) (script)|Number Theory textbook|


To download NaturalProofs, use:
```
python download.py --naturalproofs --savedir /path/to/savedir
```

#### Combined ProofWiki+Stacks
The download includes an extra combined ProofWiki+Stacks file  made with [notebooks/merge.ipynb](notebooks/merge.ipynb).

#### Preprocessing
To see the steps used to create each domain of NaturalProofs from raw data, see the following notebooks.\
This preprocessing is **not needed** if you are using a preprocessed dataset provided above.
| Domain| |
|-|-|
|ProofWiki|[notebook](notebooks/parse_proofwiki.ipynb)|
|Stacks|[notebook](notebooks/parse_stacks.ipynb)|
|Real Analysis textbook|[notebook](notebooks/parse_textbooks.ipynb)|
|Number Theory textbook|[notebook](notebooks/parse_textbooks.ipynb)|




## Mathematical Reference Retrieval and Generation
To use NaturalProofs for the reference retrieval and generation tasks described in the paper, the first step is tokenization.


### Tokenized dataset
We tokenize the raw NaturalProofs Dataset into two different formats:
- **Pairwise**: `(x, r)`
    - `x` theorem (sequence of tokens)
    - `r` reference (sequence of tokens)
    - This version is used to train and evaluate the **pairwise model**.
- **Sequence**: `(x, [rid_1, ..., rid_Tx])`
    - `x` theorem (sequence of tokens)
    - `rid_i` reference id
    - This version is used to train and evaluate the **autoregressive** and **joint** models.
  
We provide the following versions used in the paper (`bert-based-cased` tokenizer):
| Type | Domain| Splits|
|-|-|-|
|Pairwise, `bert-base-cased`|Proofwiki | train,valid,test |
||Stacks| train,valid,test |
||Real Analysis (textbook))| test |
||Number Theory (textbook)| test |
|Sequence, `bert-base-cased`|Proofwiki | train,valid,test |
||Stacks| train,valid,test |
||Real Analysis (textbook)| test | 
||Number Theory (textbook)| test | 

To download and unpack them, use:
```
python download.py --tokenized --savedir /path/to/savedir
```
Or use [google drive link](https://drive.google.com/file/d/1OCIvcCyKTyRJeV7QiHdtQQhPJ6QknMpV/view?usp=sharing).

### Pretrained Models
We provide the following models used in the paper:
| Type | |Domain| 
|-|-|-|
|**Pairwise**|`bert-base-cased`|Proofwiki |  [link]()|
|**Pairwise**|`bert-base-cased`|Stacks|  [link]()|
|**Pairwise**|`bert-base-cased`|Proofwiki+Stacks |  [link]()|
|**Joint**|`bert-base-cased`|Proofwiki |  [link]()|
|**Joint**|`bert-base-cased`|Stacks|  [link]()|
|**Joint**|`bert-base-cased`|Proofwiki+Stacks |  [link]()|
|**Autoregressive**|`bert-base-cased`|Proofwiki | 
|**Autoregressive**|`bert-base-cased`|Stacks|  [link]()|


To download and unpack them, use:
```
python download.py --checkpoint --savedir /path/to/savedir
```
Or use [google drive link](https://drive.google.com/file/d/1uIBeI7fw5vJBhDOl2WL3SbXWmzHgfK3W/view?usp=sharing).

### Creating your own tokenized dataset
This step is **not needed** if you are using a tokenized dataset provided above.\
First, setup the code:
```bash
python setup.py develop
```

To create your own tokenized versions:
- **Pairwise**: `python naturalproofs/tokenize_pairwise.py`
- **Sequence**: `python naturalproofs/encoder_decoder/utils.py`



## Evaluation

We will show you how to run evaluation on the pretrained model checkpoints & associated files.
### Setup
We will assume the file structure given by using the download script.
```bash
python download.py --naturalproofs --tokenized --checkpoint --other --savedir <SAVE-DIR>
```



We provide a script which assembles an evaluation command for `(model type, domain, task)` combinations.\
We show example commands below.

```bash
python run_analysis.py \
--train-ds-names {proofwiki stacks}+ \              # one or more training domains to choose a model
--eval-ds-names {proofwiki stacks stein trench}+ \  # one or more evaluation domains
--model {pairwise, joint, autoregressive} \ 
--generation \                                      # for generation task (autoregressive or joint models only)
--split {valid, test} \
--gpu <integer GPU id> \
--codedir /path/to/naturalproofs_code \
--datadir <SAVE-DIR>/data \
--ckptdir <SAVE-DIR>/ckpt \
--outdir <SAVE-DIR>/output
```

To make sure your filepaths line up, please look inside `run_analysis.py` to see how the `--{}dir` arguments are used.

#### Example: pairwise retrieval
```
python run_analysis.py --train-ds-names proofwiki \
--eval-ds-names proofwiki stein trench \
--model pairwise \
--gpu 1 \
--split test
```

#### Example: joint retrieval
```
python run_analysis.py --train-ds-names proofwiki \
--eval-ds-names proofwiki \
--model joint \
--gpu 1 \
--split test
```

#### Example: joint retrieval OOD
For OOD evaluation on `stein` and `trench` textbooks, provide 
reference embeddings from the pairwise model.\
These are the `__encs.pt` files from running the pairwise retrieval evaluation (we provide an example in `other/`).
```
python run_analysis.py --model joint \
--train-ds-names proofwiki \
--eval-ds-names stein trench \
--stein-rencs <SAVE-DIR>/other/pairwise__train_proofwiki__eval_stein__test__encs.pt \
--trench-rencs <SAVE-DIR>/other/pairwise__train_proofwiki__eval_trench__test__encs.pt \
--gpu 1 \
--split test
```

#### Example: joint retrieval proofwiki+stacks model
To align the model's combined output space with the individual dataset used for evaluation, give a `tok2tok.pkl` map (we provide an example in `other/`):
```
python run_analysis.py --model joint \
--train-ds-names both \
--eval-ds-names proofwiki stacks \
--modeltok2datatok <SAVE-DIR>/other/tok2tok.pkl \
--gpu 1 \
--split test
```
Note that OOD evaluation (`stein` or `trench`) is not implemented for the combined model.

#### Example: autoregressive retrieval
Without the `--generation` flag, adjusts settings for retrieval evaluation:
```
python run_analysis.py --model autoregressive \
--train-ds-names proofwiki \
--eval-ds-names proofwiki \
--gpu 1 \
--split valid
```

Note that OOD evaluation (`stein` or `trench`) is not implemented for the autoregressive model.

#### Example: autoregressive generation
```
python run_analysis.py --model autoregressive --generation \
--train-ds-names proofwiki \
--eval-ds-names proofwiki \
--gpu 1 \
--split valid
```



## Training

The provided code supports:
- Training a **pairwise** model
- Training an **autoregressive** or **joint** model, initialized with pairwise model components (parameters, reference embeddings)

#### Training a pairwise model

```bash
python naturalproofs/model.py --expr-name pairwise \
--datapath /path/to/<DOMAIN>_tokenized__bert-base-cased.pkl \
--default-root-dir /path/to/output
```


#### Training a joint model
The joint (and autoregressive) model uses a pairwise checkpoint, and reference encodings for initialization.

- The pairwise checkpoint is saved during pairwise training. 
- The reference encodings are saved in a `encs.pt` file during pairwise Evaluation.

```bash
python naturalproofs/encoder_decoder/model.py \
--datapath /path/to/sequence_<DOMAIN>_tokenized__bert-base-cased.pkl \
--default-root-dir /path/to/output
--pretrained-retrieval-checkpoint /path/to/pairwise__<DOMAIN>.ckpt \
--encs-file /path/to/train_<DOMAIN>__eval_<DOMAIN>__valid__encs.pt \  # obtained from running evaluation on trained pairwise model 
--parallel 1 \
--set-mode 1   # discard duplicates
```

Our implementation uses the same encoder-decoder architecture for the autoregressive and joint models,
considering the joint model as a one-step special case (with KL-div loss).
See the Appendix for a discussion on this design decision and technical details.


#### Training an autoregressive model

```bash
python naturalproofs/encoder_decoder/model.py \
--datapath /path/to/sequence_<DOMAIN>_tokenized__bert-base-cased.pkl \
--default-root-dir /path/to/output
--pretrained-retrieval-checkpoint /path/to/pairwise__<DOMAIN>.ckpt \
--encs-file /path/to/train_<DOMAIN>__eval_<DOMAIN>__valid__encs.pt \  # obtained from running evaluation on trained pairwise model 
--parallel 0 \
--set-mode 0   # keep duplicates
```



### Non-neural baselines
TF-IDF example:
```bash
python naturalproofs/baselines.py \
--method tfidf \
--datapath /path/to/<DOMAIN>_tokenized__bert-base-cased_200.pkl \
--datapath-base /path/to/naturalproofs_<DOMAIN>.json \
--savedir /path/to/out/

==> /path/to/out/tfidf__eval.pkl
```
Then use `analyze.py` to compute metrics:
```bash
python naturalproofs/analyze.py \
--method tfidf \
--eval-path /path/to/out/tfidf__eval.pkl \
--datapath-base /path/to/naturalproofs_<DOMAIN>.json 

==> /path/to/out/tfidf__analysis.pkl
```

