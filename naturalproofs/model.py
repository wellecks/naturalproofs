import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import pickle
import naturalproofs.dataloaders as dataloaders
import torch.nn.functional as F
import torch
import transformers
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm


class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3, log_every=50, pad_idx=0, model_type='bert-base-cased'):
        super().__init__()
        self.save_hyperparameters()
        self.metrics_train = Metrics('trn')
        self.metrics_valid = Metrics('val')

        self.x_encoder = transformers.AutoModel.from_pretrained(model_type)
        self.r_encoder = transformers.AutoModel.from_pretrained(model_type)

    def encode_x(self, x):
        xmask = x.ne(self.hparams.pad_idx).float()
        x_enc = self.x_encoder(x, attention_mask=xmask)[0][:, 0]
        return x_enc

    def encode_r(self, r):
        rmask = r.ne(self.hparams.pad_idx).float()
        r_enc = self.r_encoder(r, attention_mask=rmask)[0][:, 0]
        return r_enc

    def forward(self, x, r):
        x_enc = self.encode_x(x)
        r_enc = self.encode_r(r)
        logits = self.forward_clf(x_enc, r_enc)
        return logits

    def forward_clf(self, x_enc, r_enc):
        logits = x_enc.matmul(r_enc.transpose(0, 1))
        return logits

    def pre_encode_refs(self, ref_dl, progressbar=False):
        # pre-encode all references, store on CPU as a map from rid to vector
        print("Pre-encoding references...")
        r_encs = []
        rids = []
        if progressbar:
            iter_ = tqdm(ref_dl, total=len(ref_dl))
        else:
            iter_ = ref_dl
        for r, rid in iter_:
            r = r.cuda()
            r_enc = self.encode_r(r)
            r_encs.append(r_enc.cpu())
            rids.append(rid)
        r_encs = torch.cat(r_encs, 0).cuda()
        rids = torch.cat(rids, 0)
        return r_encs, rids

    def training_step(self, batch, batch_idx):
        x, r, y = batch
        logits = self(x, r)
        loss = F.cross_entropy(
            logits,
            torch.arange(logits.size(0), device=logits.device),
            reduction='sum'
        )
        avg_loss = loss / x.size(0)

        # -- logging
        self.metrics_train.update(loss.item(), logits, y)
        if batch_idx % self.hparams.log_every == 0:
            self.log_dict(self.metrics_train.report(), prog_bar=True)
            self.metrics_train.reset()

        return avg_loss

    def validation_step(self, batch, batch_idx):
        # pre-encode refs
        global valid_rdl, valid_x2rs
        global r_encs, rids
        if batch_idx == 0:
            r_encs, rids = self.pre_encode_refs(valid_rdl)

        x, xid = batch
        x = x.cuda()
        x_enc = self.encode_x(x)
        logits = self.forward_clf(x_enc, r_encs)
        mAPs = []
        for b in range(x.size(0)):
            ranked = list(zip(logits[b].tolist(), rids.tolist()))
            ranked = sorted(ranked, reverse=True)
            mAP = []
            num_hits = 0.0
            for rank, (_, rid) in enumerate(ranked):
                if rid in valid_x2rs[xid[b].item()]:
                    num_hits += 1.0
                    mAP.append(num_hits / (rank + 1.0))
            assert len(mAP) == len(valid_x2rs[xid[b].item()])
            mAP = np.mean(mAP)
            mAPs.append(mAP)
        self.metrics_valid.update_val(mAPs, logits)

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
        parser.add_argument('--lr', type=float, default=2e-5)
        return parser


class Metrics(object):
    def __init__(self, name):
        self._metrics = defaultdict(list)
        self.name = name

    def update(self, loss, logits, y):
        self._metrics['bsz'].append(logits.size(0))
        self._metrics['loss'].append(loss)

    def update_val(self, mAPs, logits):
        self._metrics['bsz'].append(logits.size(0))
        self._metrics['mAP'] += mAPs

    def report(self):
        out = {}
        for k, vs in self._metrics.items():
            if k == 'bsz':
                normalizer = len(self._metrics['bsz'])
            else:
                normalizer = len(self._metrics['bsz']) if k == 'bsz' else np.sum(self._metrics['bsz'])
            out['%s/%s' % (self.name, k)] = (
                np.sum(vs) / normalizer
            )
        return out

    def reset(self):
        self._metrics = defaultdict(list)


def cli_main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument('--expr-name', default='retrieval')

    parser.add_argument(
        '--datapath',
        default='/data/dataset_tokenized__random_splits_200.pkl'
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
    parser.add_argument('--debug-mode', type=int, choices=[0, 1], default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--iterable-dataset', type=int, default=1, choices=[0, 1])

    # --- Trainer/lightning
    parser.add_argument('--accumulate-grad-batches', type=int, default=1)
    parser.add_argument('--val-check-interval', type=int, default=2000)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--gpus', type=int, default=1)
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
    train_dls = dataloaders.get_train_dataloaders(
        ds_raw,
        pad=tokenizer.pad_token_id,
        token_limit=args.token_limit,
        buffer_size=args.buffer_size,
        workers=args.dataloader_workers
    )
    valid_dls = dataloaders.get_eval_dataloaders(
        ds_raw,
        pad=tokenizer.pad_token_id,
        token_limit=args.token_limit,
        workers=args.dataloader_workers,
        split_name='valid'
    )
    global valid_rdl, valid_x2rs
    valid_xdl, valid_rdl, valid_x2rs = valid_dls

    model = Classifier(
        args.lr,
        log_every=args.log_every,
        pad_idx=tokenizer.pad_token_id,
        model_type=args.model_type
    )

    if args.checkpoint_path is not None:
        print("Resuming from checkpoint (%s)" % args.checkpoint_path)
        model.load_from_checkpoint(args.checkpoint_path)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        monitor='val/mAP',
        mode='max',
        dirpath='%s/%s' % (args.default_root_dir, args.expr_name),
    )

    logger = TensorBoardLogger(
        save_dir='%s/tb_logs' % (args.default_root_dir),
        name=args.expr_name
    )

    progress = dataloaders.MaxStepsProgressBar(args.max_steps)
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
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=logger
    )

    if args.mode == 'train':
        trainer.fit(model, train_dls, valid_xdl)

    if args.mode == 'valid':
        trainer.test(model, valid_dls)


if __name__ == '__main__':
    cli_main()
