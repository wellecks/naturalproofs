import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import pickle
import naturalproofs.utils as utils
import torch.nn.functional as F
import torch.nn as nn
import torch
import transformers
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser


class Classifier(pl.LightningModule):
    def __init__(self, hidden_dim=128, lr=1e-3, dropout=0.0, log_every=50, pad_idx=0, model_type='bert-base-cased'):
        super().__init__()
        self.save_hyperparameters()
        self.metrics_train = Metrics('trn')
        self.metrics_valid = Metrics('val')

        self.x_encoder = transformers.AutoModel.from_pretrained(model_type)
        self.r_encoder = transformers.AutoModel.from_pretrained(model_type)
        enc_output_size = self.x_encoder.base_model.pooler.dense.out_features
        self.clf = nn.Sequential(
            nn.Linear(2*enc_output_size, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, r):
        x_enc = self.encode_x(x)
        r_enc = self.encode_r(r)
        logits = self.clf(torch.cat((x_enc, r_enc), 1)).view(-1)
        return logits

    def encode_x(self, x):
        xmask = x.ne(self.hparams.pad_idx).float()
        x_enc = self.x_encoder(x, attention_mask=xmask)[0][:, 0]
        return x_enc

    def encode_r(self, r):
        rmask = r.ne(self.hparams.pad_idx).float()
        r_enc = self.r_encoder(r, attention_mask=rmask)[0][:, 0]
        return r_enc

    def forward_clf(self, x_enc, r_enc):
        logits = self.clf(torch.cat((x_enc, r_enc), 1)).view(-1)
        return logits

    def training_step(self, batch, batch_idx):
        x, r, y = batch
        logits = self(x, r)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction='sum')
        avg_loss = loss / x.size(0)

        # -- logging
        self.metrics_train.update(loss.item(), logits, y)
        if batch_idx % self.hparams.log_every == 0:
            self.log_dict(self.metrics_train.report(), prog_bar=True)
            self.metrics_train.reset()

        return avg_loss

    def validation_step(self, batch, batch_idx):
        x, r, y = batch
        logits = self(x, r)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        # -- logging
        self.metrics_valid.update(loss.item(), logits, y)
        return loss

    def validation_epoch_end(self, outputs):
        self.log_dict(self.metrics_valid.report(), prog_bar=True)
        self.metrics_valid.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--lr', type=float, default=2e-5)
        return parser


class Metrics(object):
    def __init__(self, name):
        self._metrics = defaultdict(list)
        self.name = name

    def update(self, loss, logits, y):
        self._metrics['bsz'].append(logits.size(0))
        self._metrics['loss'].append(loss)

        yhat = (torch.sigmoid(logits.detach()) > 0.5).float()
        self._metrics['acc'].append(
            (yhat == y).float().sum().item()
        )
        self._metrics['pos'].append(
            (y == 1).float().sum().item()
        )
        self._metrics['rec'].append(
            ((y == 1).float()*(yhat == y).float()).sum().item()
        )
        self._metrics['prc'].append(
            ((yhat == 1).float()*(yhat == y).float()).sum().item()
        )
        self._metrics['yhat_pos'].append(
            (yhat == 1).float().sum().item()
        )

    def report(self):
        out = {}
        for k, vs in self._metrics.items():
            if k == 'yhat_pos':
                continue
            if k == 'bsz':
                normalizer = len(self._metrics['bsz'])
            elif k == 'rec':
                normalizer = max(1.0, np.sum(self._metrics['pos']))
            elif k == 'prc':
                normalizer = max(1.0, np.sum(self._metrics['yhat_pos']))
            else:
                normalizer = len(self._metrics['bsz']) if k == 'bsz' else np.sum(self._metrics['bsz'])
            out['%s/%s' % (self.name, k)] = (
                np.sum(vs) / normalizer
            )

        # F1
        prc = out['%s/prc' % self.name]
        rec = out['%s/rec' % self.name]
        normalizer = prc + rec
        out['%s/f1' % self.name] = 0.0 if normalizer == 0.0 else (2.0*prc*rec/normalizer)

        return out

    def reset(self):
        self._metrics = defaultdict(list)


def cli_main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath',
        default='/data/dataset_tokenized__bert-base-cased_200.pkl'
    )
    parser.add_argument(
        '--default-root-dir',
        default='/output'
    )
    parser.add_argument('--checkpoint-path', default=None)
    parser.add_argument('--token-limit', type=int, default=16384)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=500000)  
    parser.add_argument('--dataloader-workers', type=int, default=0)
    parser.add_argument('--model-type', default='bert-base-cased')
    parser.add_argument('--debug-mode', type=int, choices=[0,1], default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--iterable-dataset', type=int, default=1, choices=[0, 1])

    # --- Trainer/lightning
    parser.add_argument('--accumulate-grad-batches', type=int, default=1)
    parser.add_argument('--val-check-interval', type=int, default=5000)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--accelerator', default='ddp')
    parser.add_argument('--precision', type=int, default=16)

    parser = Classifier.add_model_specific_args(parser)
    args = parser.parse_args()

    assert args.dataloader_workers == 0  # others not supported by our iterable dataset

    print(args)

    pl.seed_everything(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)

    ds_raw = pickle.load(open(args.datapath, 'rb'))
    print("Loading data (%s)" % args.datapath)
    dls = utils.get_dataloaders(
        ds_raw,
        pad=tokenizer.pad_token_id,
        token_limit=args.token_limit,
        buffer_size=args.buffer_size,
        workers=args.dataloader_workers,
        iterable_dataset=args.iterable_dataset
    )

    model = Classifier(
        args.hidden_dim, args.lr, args.dropout,
        log_every=args.log_every,
        pad_idx=tokenizer.pad_token_id,
        model_type=args.model_type
    )

    if args.checkpoint_path is not None:
        print("Resuming from checkpoint (%s)" % args.checkpoint_path)
        model.load_from_checkpoint(args.checkpoint_path)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=2,
        monitor='val/f1',
        mode='max'
    )

    progress = utils.MaxStepsProgressBar(args.max_steps)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, progress],
        default_root_dir=args.default_root_dir,
        reload_dataloaders_every_epoch=True,
        move_metrics_to_cpu=True,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=args.gradient_clip_val,
        gpus=args.gpus,
        accelerator=args.accelerator,
        precision=args.precision,
        resume_from_checkpoint=args.checkpoint_path,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    if args.mode == 'train':
        trainer.fit(model, dls['train'], dls['valid'])

    if args.mode == 'valid':
        trainer.test(model, dls['valid'])




if __name__ == '__main__':
    cli_main()
