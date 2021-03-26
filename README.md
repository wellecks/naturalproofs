## NaturalProofs: Mathematical Theorem Proving in Natural Language

[NaturalProofs: Mathematical Theorem Proving in Natural Language](https://wellecks.github.io/naturalproofs/welleck2021naturalproofs.pdf)\
Sean Welleck, Jiacheng Liu, Ronan Le Bras, Hannaneh Hajishirzi, Yejin Choi, Kyunghyun Cho


This repo contains:

- The **NaturalProofs Dataset** and the **mathematical reference retrieval** task data.
- **Preprocessing** NaturalProofs and the retrieval task data.
- **Training** and **evaluation** for mathematical reference retrieval.
- **Pretrained models** for mathematical reference retrieval.

Please cite our work if you found the resources in this repository useful:
```
@article{welleck2021naturalproofs,
  title={NaturalProofs: Mathematical Theorem Proving in Natural Language},
  author={Welleck, Sean and Liu, Jiacheng and Le Bras, Ronan and Hajishirzi, Hannaneh and Choi, Yejin and Cho, Kyunghyun},
  year={2021}
}
```

| Section | Subsection |
|-|-|
| [NaturalProofs Dataset](#naturalproofs-dataset) | [Dataset](#naturalproofs-dataset) |
| | [Preprocessing](#preprocessing) |
| [Mathematical Reference Retrieval](#mathematical-reference-retrieval) | [Dataset](#dataset) |
| | [Setup](#setup) |
| | [Preprocessing](#generating-and-tokenizing) |
| | [Pretrained models](#pretrained-models) |
| | [Training](#training) |
| | [Evaluation](#evaluation) |


## NaturalProofs Dataset
We provide the preprocessed NaturalProofs Dataset (JSON):

| NaturalProofs Dataset|
|-|
|[dataset.json](https://zenodo.org/record/4632539/files/dataset.json?download=1) [[zenodo](https://zenodo.org/record/4632539)]|

#### Preprocessing
To see the steps used to create the NaturalProofs `dataset.json` from raw ProofWiki data:
1. Download the [ProofWiki XML](https://drive.google.com/file/d/1pg6ae7xt-PO0ot4F_iJv9uhTLJ8Mr6Gi/view?usp=sharing).
2. Preprocess the data using [notebooks/parse_proofwiki.ipynb](notebooks/parse_proofwiki.ipynb).
3. Form the data splits using [notebooks/dataset_splits.ipynb](notebooks/dataset_splits.ipynb).


## Mathematical Reference Retrieval

### Dataset
The Mathematical Reference Retrieval dataset contains `(x, r, y)` examples with theorem statements `x`, positive and negative references `r`, and 0/1 labels `y`,
derived from NaturalProofs.

We provide the version used in the paper (`bert-based-cased` tokenizer, 200 randomly sampled negatives):

| Reference Retrieval Dataset|
|-|
|[`bert-base-cased` 200 negatives](https://drive.google.com/file/d/1HlG36uZEM_EZ_J_C37bkivzrRYuil8OC/view?usp=sharing)|


### Pretrained Models
| Pretrained models|
|-|
|[`bert-base-cased`](https://drive.google.com/file/d/1-Jih8FhfpXecKGSnGomGmsavvL4lipjZ/view?usp=sharing)|
|[`lstm`](https://drive.google.com/file/d/1c1MGb3Y9YFcs5afWCaC1FXXQRHu89xat/view?usp=sharing)|

These models were trained with the "`bert-base-cased` 200 negatives" dataset provided above.

### Setup
```bash
python setup.py develop
```
You can see the [DockerFile](Dockerfile) for additional version info, etc.

### Generating and tokenizing
To create your own version of the retrieval dataset, use `python utils.py`. 

This step is **not needed** if you are using the reference retrieval dataset provided above.

Example:
```bash
python utils.py --filepath /path/to/dataset.json --output-path /path/to/out/ --model-type bert-base-cased
# => Writing dataset to /path/to/out/dataset_tokenized__bert-base-cased_200.pkl
```



### Evaluation
Using the retrieval dataset and a model provided above, we compute the test evaluation metrics in the paper:


1. Predict the rankings:
```bash
python naturalproofs/predict.py \
--method bert-base-cased \      # | lstm
--model-type bert-base-cased \  # | lstm
--datapath /path/to/dataset_tokenized__bert-base-cased_200.pkl \
--datapath-base /path/to/dataset.json \
--checkpoint-path /path/to/best.ckpt \
--output-dir /path/to/out/ \
--split test  # use valid during model development
```

2. Compute metrics over the rankings:
```bash
python naturalproofs/analyze.py \
--method bert-base-cased \      # | lstm
--eval-path /path/to/out/eval.pkl \
--analysis-path /path/to/out/analysis.pkl
```

### Training
```bash
python naturalproofs/model.py \
--datapath /path/to/dataset_tokenized__bert-base-cased_200.pkl \
--default-root-dir /path/to/out/
```

### Classical Retrieval Baselines
TF-IDF example:
```bash
python naturalproofs/baselines.py \
--method tfidf \
--datapath /path/to/dataset_tokenized__bert-base-cased_200.pkl \
--datapath-base /path/to/dataset.json \
--savedir /path/to/out/
```
Then use `analyze.py` as shown above to compute metrics.
