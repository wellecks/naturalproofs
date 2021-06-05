from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from naturalproofs.tokenize_pairwise import replace_links
import torch
import numpy as np
import json
import transformers


class SequenceRetrievalDataset(Dataset):
    def __init__(
            self, data, xpad, ypad, token_limit, buffer_size,
            sort_buffer=False,
            include_metadata=False,
            set_mode=False,
            order='ground-truth'
    ):
        super().__init__()
        self.xpad = xpad
        self.ypad = ypad
        self.data = data
        self._set = set_mode
        self._order = order
        self._tok_limit = token_limit
        self._buf_size = buffer_size
        self._sort_buffer = sort_buffer
        self._include_metadata = include_metadata
        self.reset()
        print("%d batches" % (len(self._batches)))

    def reset(self):
        self._perm = np.random.permutation(len(self.data))
        self._curr = 0
        self._example_ctr = 0
        self._buf = []
        self._batches = self._make_batches()

    def __len__(self):
        return len(self._batches)

    def _make_batch(self):
        batch = []
        x_longest, y_longest = 0, 0
        start_idx = 0
        while self._buf and (len(batch) * x_longest + len(batch) * y_longest) <= self._tok_limit:
            idx = min(start_idx, len(self._buf)-1)
            example = self._buf.pop(idx)
            x_longest = max(len(example['x']), x_longest)

            if self._set:
                example['y'] = self._to_set(example['y'])

            y_longest = max(len(example['y']), y_longest)
            batch.append(example)

        # remove the violation so that we're guaranteed to be under the limit
        if len(batch) > 1:
            ex_ = batch[-1]
            self._buf.append(ex_)
            batch = batch[:-1]
            x_longest = max(len(ex['x']) for ex in batch)
            y_longest = max(len(ex['y']) for ex in batch)

        x_batch = torch.zeros((len(batch), x_longest), dtype=torch.long).fill_(self.xpad)
        y_batch = torch.zeros((len(batch), y_longest), dtype=torch.long).fill_(self.ypad)

        for i, example in enumerate(batch):
            x_batch[i, :len(example['x'])] = torch.tensor(example['x'], dtype=torch.long)
            y_ordered = self._order_y(example['y'])
            y_batch[i, :len(example['y'])] = torch.tensor(y_ordered, dtype=torch.long)

        if self._include_metadata:
            metadata = [example['metadata'] for example in batch]
            return x_batch, y_batch, metadata

        return x_batch, y_batch

    @staticmethod
    def _to_set(y):
        # maintains the ground-truth order but removes duplicates
        seen = set()
        yset = []
        for yt in y:
            if yt not in seen:
                yset.append(yt)
                seen.add(yt)
        return yset

    def _order_y(self, y):
        if self._order == 'ground-truth':
            return y
        else:
            raise NotImplementedError("order %s" % self._order)

    def _make_batches(self):
        # continue until we've iterated through the whole dataset
        batches = []
        while self._example_ctr < len(self.data):
            # if the buffer is empty, fill it
            if len(self._buf) == 0:
                buf_idxs = self._perm[self._curr:self._curr + self._buf_size]
                self._buf = [self.data[i] for i in buf_idxs]
                self._curr += self._buf_size
                if self._sort_buffer:
                    self._buf = sorted(
                        self._buf,
                        key=lambda x: len(x['x'] + x['y'])
                    )

            batch = self._make_batch()
            self._example_ctr += batch[0].size(0)

            batches.append(batch)
        return batches

    def __getitem__(self, idx):
        return self._batches[idx]

    @classmethod
    def collate(cls, batch):
        return batch


def trim(model, seq):
    assert seq.ndim == 1 and len(seq) > 0
    # remove bos
    assert seq[0] == model.hparams.bos
    seq = seq[1:]
    # remove padding
    seq = seq[seq != model.hparams.ypad]
    # truncate at first <eos> (if available)
    seq = seq[((seq == model.hparams.eos).cumsum(0).cumsum(0) < 1)]
    return seq


def extract_rankings(model, xj, yhatj, use_first, use_generations, k=1000):
    logits = model(xj, yhatj).logits
    yhatj_ = yhatj[0].tolist()

    if model.hparams.eos in yhatj_:  # use the timestep at which <eos> was predicted.
        idx = yhatj_.index(model.hparams.eos) - 1
    else:
        idx = len(yhatj_) - 1

    if use_first:
        idx = 0

    def _allowed(token):
        return token not in {model.hparams.bos, model.hparams.eos, model.hparams.xpad, model.hparams.ypad}

    rankings = []
    if use_generations:
        for yh in yhatj_:
            if _allowed(yh) and yh not in rankings:
                rankings.append(yh)
            if yh == model.hparams.eos:
                break

    last_probs = logits[0, idx]
    topk_k = k + len(set(yhatj_))
    topk = (last_probs.topk(topk_k)[1]).tolist()
    for yh in topk:
        if len(rankings) == k:
            break
        if _allowed(yh) and yh not in rankings:
            rankings.append(yh)

    return rankings


# ---- data tokenization and loading
def _tokenize_examples(split, dataset, tokenizer, rid2tok, title_only=False, content_only=False):
    stats = defaultdict(int)

    examples = split['examples']
    id2thm = {thm['id']: thm for thm in dataset['theorems']}

    tokenized = []
    for eid, (tid, pix) in tqdm(enumerate(examples), total=len(examples)):
        ex = id2thm[tid]
        proof = ex['proofs'][pix]

        title_ = '' if content_only else ex['title']
        content_ = '' if title_only else ' '.join(replace_links(ex['contents']))
        contents = "%s%s%s" % (
            title_,
            tokenizer.sep_token,
            content_
        )

        ids = tokenizer.encode(contents)
        if len(ids) > tokenizer.model_max_length:
            ids = ids[:tokenizer.model_max_length-1] + [tokenizer.sep_token_id]
            stats['truncated'] += 1

        rids = proof['ref_ids']
        for rid in rids:
            if rid not in rid2tok:
                tok = len(rid2tok)
                rid2tok[rid] = tok

        tokenized.append({
            'x': ids,
            'y': [rid2tok['<bos>']] + [rid2tok[rid] for rid in rids] + [rid2tok['<eos>']],
            'metadata': {
                'eid': eid
            }
        })
        if tokenizer.unk_token_id in ids:
            stats['hasunk'] += 1

    print("%d examples\n\t%d truncated\n\t%d have [UNK]" % (
        len(tokenized), stats['truncated'], stats['hasunk']
    ))
    return tokenized, rid2tok


def tokenize_dataset(
    filepath,
    model_type='bert-base-cased',
    ex_title_only=False,
    ex_content_only=False
):
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
    base = json.load(open(filepath, 'r'))
    dataset, splits = base['dataset'], base['splits']
    tokenized = {}

    # We distinguish between the dataset's reference id, and the token used to encode the reference.
    rid2tok = {
        '<pad>': 0,
        '<bos>': 1,
        '<eos>': 2
    }
    for name, split in splits.items():
        print("== tokenizing %s" % name)
        tokenized_, rid2tok = _tokenize_examples(
            split, dataset, tokenizer,
            rid2tok=rid2tok,
            title_only=ex_title_only,
            content_only=ex_content_only
        )
        tokenized[name] = tokenized_

    return tokenized, rid2tok


def get_dataloaders(ds_raw, xpad, ypad, token_limit, buffer_size, workers, set_mode, order, include_metadata=False):
    dls = {}
    for name, data in ds_raw.items():
        if name == 'refs':
            continue
        ds = SequenceRetrievalDataset(
            data=data,
            xpad=xpad,
            ypad=ypad,
            token_limit=token_limit,
            buffer_size=buffer_size,
            include_metadata=include_metadata,
            set_mode=set_mode,
            order=order
        )
        dls[name] = DataLoader(
            ds,
            collate_fn=SequenceRetrievalDataset.collate,
            batch_size=None,
            batch_sampler=None,
            num_workers=workers,
        )
    return dls


def get_idx2tok_map(pretrained_model_idx2rid, dataset_rid2tok):
    idx2tok = {}
    for idx, rid in enumerate(pretrained_model_idx2rid.tolist()):
        if rid in dataset_rid2tok:
            idx2tok[idx] = dataset_rid2tok[rid]
    return idx2tok


if __name__ == '__main__':
    import argparse
    import pickle
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default='/path/to/dataset.json')
    parser.add_argument('--output-dir', default='/path/to/output/')
    parser.add_argument('--model-type', default='bert-base-cased')
    parser.add_argument('--ex-title-only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ex-content-only', type=int, default=0, choices=[0, 1])

    parser.add_argument('--dataset-name', required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenized, rid2tok = tokenize_dataset(
        args.filepath,
        model_type=args.model_type,
        ex_title_only=bool(args.ex_title_only),
        ex_content_only=bool(args.ex_content_only),
    )

    suffix = '' if not args.ex_title_only else '_extitle_only'
    suffix += '' if not args.ex_content_only else '_excontentonly'
    outfile = os.path.join(args.output_dir, 'sequence_%s_tokenized__%s%s.pkl' % (
        args.dataset_name, args.model_type.replace('/', '_'), suffix
    ))

    dataset = {
        'tokenized': tokenized,
        'rid2tok': rid2tok
    }
    print("Writing dataset to %s" % outfile)
    with open(outfile, 'wb') as f:
        pickle.dump(dataset, f)
    print("=== done.")







