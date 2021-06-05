from torch.utils.data import Dataset, DataLoader, IterableDataset
from collections import defaultdict
from pytorch_lightning.callbacks.progress import ProgressBar
import torch
import numpy as np


def match(unmatched_data, rid2r):
    data = []
    for ex in unmatched_data:
        # -- positive references
        x, rids = ex['x'], ex['rids']
        for rid in rids:
            r = rid2r[rid]
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

    return data


class IterablePairedDataset(IterableDataset):
    def __init__(self, unmatched_data, rid2r, pad, token_limit, buffer_size, include_metadata=False):
        super().__init__()
        self.pad = pad
        self._rid2r = rid2r
        self._unmatched_data = unmatched_data
        self._tok_limit = token_limit
        self._buf_size = buffer_size
        self._include_metadata = include_metadata
        # WORKAROUND: prevent duplicate data with pytorch lightning.
        # Note that this is not ideal and introduces randomness into training.
        self._randomstate = np.random.RandomState(int(id(self))%(2**32))
        self.reset()

    def reset(self):
        self.data = match(self._unmatched_data, self._rid2r)
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
    def __init__(self, unmatched_data, rid2r, pad, token_limit, buffer_size, include_metadata=False):
        super().__init__()
        self.pad = pad
        self._rid2r = rid2r
        self._unmatched_data = unmatched_data
        self._tok_limit = token_limit
        self._buf_size = buffer_size
        self._include_metadata = include_metadata
        self.reset()
        print("%d batches" % (len(self._batches)))

    def reset(self):
        self.data = match(self._unmatched_data, self._rid2r)
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


def get_train_dataloaders(ds_raw, pad, token_limit, buffer_size, workers, include_metadata=False):
    data = ds_raw['train']
    ds = IterablePairedDataset(
        unmatched_data=data['unmatched_data'],
        rid2r=data['rid2r'],
        pad=pad,
        token_limit=token_limit,
        buffer_size=buffer_size,
        include_metadata=include_metadata,
    )
    return DataLoader(
        ds,
        collate_fn=PairedDataset.collate,
        batch_size=None,
        batch_sampler=None,
        num_workers=workers,
    )


def get_eval_dataloaders(ds_raw, pad, token_limit, workers, split_name):
    # -- examples
    data = ds_raw[split_name]
    ds = ReferenceDataset(data['unmatched_data'], pad, token_limit, key='x', id_key='eid')
    xdl = DataLoader(
        ds,
        collate_fn=ReferenceDataset.collate,
        batch_size=None,
        batch_sampler=None,
        num_workers=workers
    )

    # -- refs
    ds = ReferenceDataset(ds_raw['refs'], pad, token_limit, key='r', id_key='rid')
    rdl = DataLoader(
        ds,
        collate_fn=ReferenceDataset.collate,
        batch_size=None,
        batch_sampler=None,
        num_workers=workers
    )

    # -- example to ground-truth refs mapping
    x2rs = defaultdict(list)
    for ex in ds_raw[split_name]['unmatched_data']:
        xid = ex['metadata']['eid']
        rids = ex['rids']
        x2rs[xid] = rids

    return xdl, rdl, x2rs

