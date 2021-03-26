from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm
from collections import defaultdict
from pytorch_lightning.callbacks.progress import ProgressBar
import torch
import numpy as np
import json
import transformers
import re


class IterablePairedDataset(IterableDataset):
    def __init__(self, data, pad, token_limit, buffer_size, sort_buffer=False, include_metadata=False):
        super().__init__()
        self.pad = pad
        self.data = data
        self._tok_limit = token_limit
        self._buf_size = buffer_size
        self._sort_buffer = sort_buffer
        self._include_metadata = include_metadata
        # WORKAROUND: prevent duplicate data with pytorch lightning.
        # Note that this is not ideal and introduces randomness into training.
        self._randomstate = np.random.RandomState(int(id(self))%(2**32))
        self.reset()

    def reset(self):
        self._perm = self._randomstate.permutation(len(self.data))
        self._curr = 0
        self._example_ctr = 0
        self._buf = []

    def _make_batch(self):
        batch = []
        x_longest, r_longest = 0, 0
        start_idx = 0
        while self._buf and (len(batch) * x_longest + len(batch) * r_longest) <= self._tok_limit:
            idx = min(start_idx, len(self._buf)-1)
            example = self._buf.pop(idx)
            x_longest = max(len(example['x']), x_longest)
            r_longest = max(len(example['r']), r_longest)
            batch.append(example)

        # remove the violation so that we're guaranteed to be under the limit
        if len(batch) > 1:
            ex_ = batch[-1]
            self._buf.append(ex_)
            batch = batch[:-1]
            x_longest = max(len(ex['x']) for ex in batch)
            r_longest = max(len(ex['r']) for ex in batch)

        x_batch = torch.zeros((len(batch), x_longest), dtype=torch.long).fill_(self.pad)
        r_batch = torch.zeros((len(batch), r_longest), dtype=torch.long).fill_(self.pad)
        y_batch = torch.zeros(len(batch), dtype=torch.float).fill_(self.pad)

        for i, example in enumerate(batch):
            x_batch[i, :len(example['x'])] = torch.tensor(example['x'], dtype=torch.long)
            r_batch[i, :len(example['r'])] = torch.tensor(example['r'], dtype=torch.long)
            y_batch[i] = example['y']

        if self._include_metadata:
            metadata = [example['metadata'] for example in batch]
            return x_batch, r_batch, y_batch, metadata

        return x_batch, r_batch, y_batch

    def __iter__(self):
        while True:
            # if the buffer is empty, fill it
            if len(self._buf) == 0:
                buf_idxs = self._perm[self._curr:self._curr + self._buf_size]
                self._buf = [self.data[i] for i in buf_idxs]
                self._curr += self._buf_size
                if self._sort_buffer:
                    self._buf = sorted(
                        self._buf,
                        key=lambda x: len(x['x'] + x['r'])
                    )

            batch = self._make_batch()
            self._example_ctr += batch[0].size(0)

            if self._example_ctr >= len(self.data):
                # we've iterated through the whole dataset
                self.reset()

            yield batch

    @classmethod
    def collate(cls, batch):
        return batch


class PairedDataset(Dataset):
    def __init__(self, data, pad, token_limit, buffer_size, sort_buffer=False, include_metadata=False):
        super().__init__()
        self.pad = pad
        self.data = data
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
        x_longest, r_longest = 0, 0
        start_idx = 0
        while self._buf and (len(batch) * x_longest + len(batch) * r_longest) <= self._tok_limit:
            idx = min(start_idx, len(self._buf)-1)
            example = self._buf.pop(idx)
            x_longest = max(len(example['x']), x_longest)
            r_longest = max(len(example['r']), r_longest)
            batch.append(example)

        # remove the violation so that we're guaranteed to be under the limit
        if len(batch) > 1:
            ex_ = batch[-1]
            self._buf.append(ex_)
            batch = batch[:-1]
            x_longest = max(len(ex['x']) for ex in batch)
            r_longest = max(len(ex['r']) for ex in batch)

        x_batch = torch.zeros((len(batch), x_longest), dtype=torch.long).fill_(self.pad)
        r_batch = torch.zeros((len(batch), r_longest), dtype=torch.long).fill_(self.pad)
        y_batch = torch.zeros(len(batch), dtype=torch.float).fill_(self.pad)

        for i, example in enumerate(batch):
            x_batch[i, :len(example['x'])] = torch.tensor(example['x'], dtype=torch.long)
            r_batch[i, :len(example['r'])] = torch.tensor(example['r'], dtype=torch.long)
            y_batch[i] = example['y']

        if self._include_metadata:
            metadata = [example['metadata'] for example in batch]
            return x_batch, r_batch, y_batch, metadata

        return x_batch, r_batch, y_batch

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
                        key=lambda x: len(x['x'] + x['r'])
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


class ReferenceDataset(Dataset):
    def __init__(self, data, pad, token_limit, key, id_key):
        super().__init__()
        self.pad = pad
        self.data = data
        self._tok_limit = token_limit
        self._key = key
        self._id_key = id_key
        self.reset()
        print("%d batches (%d %s)" % (len(self._batches), self._example_ctr, self._key))

    def reset(self):
        self._curr = 0
        self._example_ctr = 0
        self._batches = self._make_batches()

    def __len__(self):
        return len(self._batches)

    def _make_batch(self):
        batch = []
        longest = 0
        start_idx = 0
        while self._buf and ((len(batch) * longest) <= self._tok_limit):
            idx = min(start_idx, len(self._buf)-1)
            example = self._buf.pop(idx)
            longest = max(len(example[self._key]), longest)
            batch.append(example)

        # remove the violation so that we're guaranteed to be under the limit
        if len(batch) > 1:
            ex_ = batch[-1]
            self._buf.append(ex_)
            batch = batch[:-1]
            longest = max(len(ex[self._key]) for ex in batch)

        r_batch = torch.zeros((len(batch), longest), dtype=torch.long).fill_(self.pad)
        rid_batch = torch.zeros((len(batch)), dtype=torch.long)
        for i, example in enumerate(batch):
            r_batch[i, :len(example[self._key])] = torch.tensor(example[self._key], dtype=torch.long)
            rid_batch[i] = example['metadata'][self._id_key]

        return r_batch, rid_batch

    def _make_batches(self):
        # continue until we've iterated through the whole dataset
        batches = []
        self._buf = [self.data[i] for i in range(len(self.data))]  
        # eliminate duplicates
        seen = set()
        self._buf = []
        for i in range(len(self.data)):
            example = self.data[i]
            if example['metadata'][self._id_key] in seen:
                continue
            else:
                self._buf.append(example)
                seen.add(example['metadata'][self._id_key])

        while self._example_ctr < len(seen):
            batch = self._make_batch()
            self._example_ctr += batch[0].size(0)
            batches.append(batch)

        return batches

    def __getitem__(self, idx):
        return self._batches[idx]

    @classmethod
    def collate(cls, batch):
        return batch


class MaxStepsProgressBar(ProgressBar):
    def __init__(self, max_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.total = self.max_steps
        return bar

# ---- data tokenization and loading
def _tokenize_examples(split, dataset, tokenizer, title_only=False, include_proofs=False, content_only=False):
    stats = defaultdict(int)
    exs = dataset['retrieval_examples']
    eids = split['example_ids']

    id2proofcontent = {}
    for proof in dataset['proofs']:
        pc = '\n'.join(proof['contents'])
        pc_masked = re.sub(
            r'\[\[.*\]\]',
            tokenizer.mask_token,
            pc,
            re.MULTILINE
        )
        id2proofcontent[proof['proof_id']] = pc_masked

    tokenized = []
    for eid in tqdm(eids, total=len(eids)):
        ex = exs[eid]
        title_ = '' if content_only else ex['title']
        content_ = '' if title_only else ' '.join(ex['statement']['read_contents'])
        contents = "%s%s%s" % (
            title_,
            tokenizer.sep_token,
            content_
        )

        if include_proofs:
            proof_ids = [proof['proof_id'] for proof in ex['proofs']]
            pcs = tokenizer.sep_token.join(
                [id2proofcontent[pid] for pid in proof_ids]
            )
            contents += "%s%s" % (
                tokenizer.sep_token,
                pcs
            )

        ids = tokenizer.encode(contents)
        if len(ids) > tokenizer.model_max_length:
            ids = ids[:tokenizer.model_max_length-1] + [tokenizer.sep_token_id]
            stats['truncated'] += 1

        rids = sorted(set([r for proof in ex['proofs'] for r in proof['ref_ids']]))
        tokenized.append({
            'x': ids,
            'rids': rids,
            'metadata': {
                'x_tokens': tokenizer.convert_ids_to_tokens(ids),
                'eid': eid
            }
        })
        if tokenizer.unk_token_id in ids:
            stats['hasunk'] += 1

    print("%d examples\n\t%d truncated\n\t%d have [UNK]" % (
        len(tokenized), stats['truncated'], stats['hasunk']
    ))
    return tokenized


def _tokenize_refs(split, dataset, tokenizer, title_only=False, content_only=False):
    stats = defaultdict(int)
    rids = split['ref_ids']
    tokenized = {}
    exs = dataset['theorems'] + dataset['definitions'] + dataset['other']
    for ex in tqdm(exs, total=len(exs)):
        if ex['id'] in rids:
            title_ = '' if content_only else ex['title']
            content_ = '' if title_only else ' '.join(ex['read_contents'])
            contents = "%s%s%s" % (
                title_,
                tokenizer.sep_token,
                content_
            )
            ids = tokenizer.encode(contents)
            if len(ids) > tokenizer.model_max_length:
                ids = ids[:tokenizer.model_max_length-1] + [tokenizer.sep_token_id]
                stats['truncated'] += 1

            tokenized[ex['id']] = {
                'r': ids,
                'metadata': {
                    'r_tokens': tokenizer.convert_ids_to_tokens(ids),
                    'rid': ex['id']
                }
            }
            if tokenizer.unk_token_id in ids:
                stats['hasunk'] += 1

    print("%d refs\n\t%d truncated\n\t%d have [UNK]" % (
        len(tokenized), stats['truncated'], stats['hasunk']
    ))
    reflist = [tokenized[r] for r in rids]
    return tokenized, reflist


def _match_examples_refs(exs, refs, split, num_negative):
    data = []
    all_rids = split['ref_ids']
    for ex in tqdm(exs, total=len(exs)):
        # -- positive references
        x, rids = ex['x'], ex['rids']
        for rid in rids:
            r = refs[rid]
            example = {
                'x': ex['x'],
                'r': r['r'],
                'y': 1,
                'metadata': {}
            }
            for k, v in ex['metadata'].items():
                example['metadata'][k] = v
            for k, v in r['metadata'].items():
                example['metadata'][k] = v
            data.append(example)

        # -- negative references
        rset = sorted(list(set(all_rids).difference(set(rids))))
        num_negative_ = len(rids) if num_negative < 0 else num_negative
        rids = np.random.choice(rset, size=num_negative_, replace=False)
        for rid in rids:
            r = refs[rid]
            example = {
                'x': ex['x'],
                'r': r['r'],
                'y': 0,
                'metadata': {}
            }
            for k, v in ex['metadata'].items():
                example['metadata'][k] = v
            for k, v in r['metadata'].items():
                example['metadata'][k] = v
            data.append(example)
    print("%d (x, r, y) examples." % (len(data)))
    return data


def tokenize_dataset(
    filepath,
    model_type='bert-base-cased',
    num_negative=200,
    ref_title_only=False,
    ex_title_only=False,
    include_proofs=False,
    ref_content_only=False,
    ex_content_only=False
):
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
    base = json.load(open(filepath, 'r'))
    dataset, splits = base['dataset'], base['splits']
    tokenized = {}

    print("---- assuming all refs are present in `valid` `ref_ids`")
    print("== tokenizing refs")
    rid2r, reflist = _tokenize_refs(
        splits['valid'], dataset, tokenizer,
        title_only=ref_title_only,
        content_only=ref_content_only
    )
    tokenized['refs'] = reflist

    for name, split in splits.items():
        print("== tokenizing %s" % name)
        xs = _tokenize_examples(
            split, dataset, tokenizer,
            title_only=ex_title_only,
            include_proofs=include_proofs,
            content_only=ex_content_only
        )
        print("- matching %s" % name)
        data = _match_examples_refs(xs, rid2r, split, num_negative)
        tokenized[name] = data

    return tokenized


def get_dataloaders(ds_raw, pad, token_limit, buffer_size, workers, include_metadata=False, iterable_dataset=False):
    dls = {}
    for name, data in ds_raw.items():
        if name == 'refs':
            continue
        if iterable_dataset and name == 'train':
            ds = IterablePairedDataset(data, pad, token_limit, buffer_size, include_metadata=include_metadata)
        else:
            ds = PairedDataset(data, pad, token_limit, buffer_size, include_metadata=include_metadata)
        dls[name] = DataLoader(
            ds,
            collate_fn=PairedDataset.collate,
            batch_size=None,
            batch_sampler=None,
            num_workers=workers,
        )
    return dls


def get_eval_dataloaders(ds_raw, pad, token_limit, workers, split_name):
    # -- examples
    data = ds_raw[split_name]
    ds = ReferenceDataset(data, pad, token_limit, key='x', id_key='eid')
    xdl = DataLoader(
        ds,
        collate_fn=ReferenceDataset.collate,
        batch_size=None,
        batch_sampler=None,
        num_workers=workers
    )

    # -- refs
    data = ds_raw['refs']
    ds = ReferenceDataset(data, pad, token_limit, key='r', id_key='rid')
    rdl = DataLoader(
        ds,
        collate_fn=ReferenceDataset.collate,
        batch_size=None,
        batch_sampler=None,
        num_workers=workers
    )

    # -- example to ground-truth refs mapping
    x2rs = defaultdict(list)
    for ex in ds_raw[split_name]:
        if ex['y'] == 1:
            xid = ex['metadata']['eid']
            rid = ex['metadata']['rid']
            x2rs[xid].append(rid)

    return xdl, rdl, x2rs

if __name__ == '__main__':
    import argparse
    import pickle
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default='/path/to/dataset.json')
    parser.add_argument('--output-path', default='/path/to/out/')
    parser.add_argument('--model-type', default='bert-base-cased')
    parser.add_argument('--num-negative', type=int, default=200, help='-1 for equal')
    parser.add_argument('--ref-title-only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ex-title-only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--include-proofs', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ref-content-only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ex-content-only', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    tokenized = tokenize_dataset(
        args.filepath,
        model_type=args.model_type,
        num_negative=args.num_negative,
        ref_title_only=bool(args.ref_title_only),
        ex_title_only=bool(args.ex_title_only),
        include_proofs=bool(args.include_proofs),
        ref_content_only=bool(args.ref_content_only),
        ex_content_only=bool(args.ex_content_only),
    )

    suffix = '' if not args.ref_title_only else '_reftitleonly'
    suffix += '' if not args.ex_title_only else '_extitle_only'
    suffix += '' if not args.include_proofs else '_include_proofs'
    suffix += '' if not args.ref_content_only else '_refcontentonly'
    suffix += '' if not args.ex_content_only else '_excontentonly'
    outfile = os.path.join(args.output_path, 'dataset_tokenized__%s_%d%s.pkl' % (
        args.model_type.replace('/', '_'), args.num_negative, suffix
    ))
    print("Writing dataset to %s" % outfile)
    with open(outfile, 'wb') as f:
        pickle.dump(tokenized, f)
    print("=== done.")







