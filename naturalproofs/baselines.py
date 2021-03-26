from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
import naturalproofs.utils as utils
import pickle
import transformers
import os
import json
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--method', required=True, choices=['tfidf', 'bm25', 'random', 'distribution']
)
parser.add_argument(
    '--datapath', default='/path/to/dataset_tokenized__bert-base-cased_200.pkl'
)
parser.add_argument(
    '--savedir', default='/path/to/out'
)
parser.add_argument(
    '--tokenizer', default='bert-base-cased'
)
parser.add_argument(
    '--datapath-base',
    default='/data/dataset.json'
)
parser.add_argument(
    '--split', default='valid', choices=['valid', 'test']
)
args = parser.parse_args()


# Get set of tokenized reference set
print("== Loading data")
ds_base = json.load(open(args.datapath_base, 'rb'))

tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
ds_raw = pickle.load(open(args.datapath, 'rb'))
xdl, rdl, x2rs = utils.get_eval_dataloaders(
    ds_raw,
    pad=tokenizer.pad_token_id,
    token_limit=10000,
    workers=0,
    split_name=args.split
)

# References
references = []
rids = []
for (rs, rids_) in rdl.dataset:
    for r in rs:
        r_ = r[r != tokenizer.pad_token_id].tolist()
        references.append(r_)
    rids.extend(rids_.view(-1).tolist())
rids = np.array(rids)

# Validation examples
examples = []
for (xs, xids) in xdl.dataset:
    for x, xid in zip(xs, xids):
        x_ = x[x != tokenizer.pad_token_id].tolist()
        examples.append((x_, xid.item()))


if args.method == 'tfidf':
    print("== Computing tf-idf")
    # Compute tf-idf vectors for tokenized reference set
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    R = vectorizer.fit_transform(references)

    # Compute tf-idf vector for validation example
    # Produce ranked list of references for the validation example using cosine similarity
    x2ranked = {}
    for (x, xid) in examples:
        x_ = vectorizer.transform([x])
        scores = np.asarray(R.dot(x_.T).todense()).ravel()
        ranks = scores.argsort()[::-1]
        ranked_scores = scores[ranks].tolist()
        ranked_rids = rids[ranks].tolist()
        ranked = list(zip(ranked_scores, ranked_rids))
        x2ranked[xid] = ranked

if args.method == 'bm25':
    print("== Computing bm25")

    bm25 = BM25Okapi(references)
    x2ranked = {}
    for (x, xid) in tqdm(examples, total=len(examples)):
        scores = bm25.get_scores(x)
        ranks = scores.argsort()[::-1]
        ranked_scores = scores[ranks].tolist()
        ranked_rids = rids[ranks].tolist()
        ranked = list(zip(ranked_scores, ranked_rids))
        x2ranked[xid] = ranked

if args.method == 'random':
    print("== Computing random")

    x2ranked = {}
    for (x, xid) in examples:
        ranks = np.random.permutation(np.arange(len(rids)))
        ranked_scores = ((np.arange(len(rids), dtype=float)[::-1] / len(rids))).tolist()
        ranked_rids = rids[ranks].tolist()
        ranked = list(zip(ranked_scores, ranked_rids))
        x2ranked[xid] = ranked


if args.method == 'distribution':
    print("== Computing according to training reference distribution")
    train_xids = set(ds_base['splits']['train']['example_ids'])
    from collections import Counter
    counts = Counter()
    for item in ds_base['dataset']['retrieval_examples']:
        if item['example_id'] in train_xids:
            for p in item['proofs']:
                for r in p['ref_ids']:
                    counts[r] += 1

    xs = counts.most_common()
    ranked_rids = [x[0] for x in xs]
    ranked_scores = ((np.arange(len(counts), dtype=float)[::-1] / len(rids))).tolist()
    ranked = list(zip(ranked_scores, ranked_rids))

    x2ranked = {}
    for (x, xid) in examples:
        x2ranked[xid] = ranked


# Save
if not os.path.exists(args.savedir):
    os.makedirs(args.savedir, exist_ok=True)
outfile = os.path.join(os.path.dirname(args.savedir), 'eval_%s.pkl' % args.method)

pickle.dump({
    'x2ranked': x2ranked,
    'x2rs': x2rs,
    'rids': rids,
    'name': args.method
}, open(outfile, 'wb'))


print("\nWrote to %s" % outfile)
print("== done.")
