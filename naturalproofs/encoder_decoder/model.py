import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import pickle
import math
import naturalproofs.encoder_decoder.utils as utils
import torch.nn.functional as F
import torch
import transformers
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
from naturalproofs.encoder_decoder.utils import trim


class SequenceRetriever(pl.LightningModule):
    """The autoregressive model."""
    def __init__(self, vocab_size, bos, eos, xpad, ypad, lr, log_every, model_type='bert-base-cased', decode_max_length=20):
        super().__init__()
        self.save_hyperparameters()
        self.metrics_train = Metrics('trn', ypad)
        self.metrics_valid = Metrics('val', ypad)

        encoder_decoder_config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
            transformers.BertConfig.from_pretrained(model_type),
            transformers.BertConfig.from_pretrained(model_type)
        )
        encoder_decoder_config.decoder.vocab_size = self.hparams.vocab_size
        encoder_decoder_config.encoder.pad_token_id = self.hparams.xpad
        encoder_decoder_config.decoder.pad_token_id = self.hparams.ypad
        encdec = transformers.EncoderDecoderModel(encoder_decoder_config)
        self.encdec = encdec

    def initialize(self, encs_file, ckpt_file, dataset_rid2tok, init_enc, init_dec_emb, init_dec):
        # Initialize encoder with theorem-encoder's weights.
        ckpt = torch.load(ckpt_file)
        state_dict = ckpt['state_dict']
        if init_enc:
            state_dict_ = {}
            for k, v in state_dict.items():
                if 'x_encoder' in k:
                    k_ = k.replace('x_encoder.', '')
                    state_dict_[k_] = v
            self.encdec.encoder.load_state_dict(state_dict_)

        # Initialize decoder with reference-encoder's weights.
        if init_dec:
            state_dict_ = {}
            for k, v in state_dict.items():
                if 'r_encoder.encoder' in k:
                    k_ = k.replace('r_encoder.', '')
                    state_dict_[k_] = v
            # Use `strict=False` due to new cross-attention weights.
            self.encdec.decoder.bert.load_state_dict(state_dict_, strict=False)

        # Initialize decoder embedding matrix using reference-encoder's reference embeddings.
        if init_dec_emb:
            encs = torch.load(encs_file)
            idx2tok = utils.get_idx2tok_map(
                pretrained_model_idx2rid=encs['rids'],
                dataset_rid2tok=dataset_rid2tok
            )
            r_encs = encs['r_encs'].clone()
            for idx in range(r_encs.size(0)):
                if idx in idx2tok:
                    tok = idx2tok[idx]
                    self.encdec.decoder.bert.embeddings.word_embeddings.weight.data[tok] = r_encs[idx]
            assert (self.encdec.decoder.bert.embeddings.word_embeddings.weight ==
                    self.encdec.decoder.cls.predictions.decoder.weight).all()

    def freeze_parts(self, freeze_enc, freeze_dec_emb, freeze_dec):
        if freeze_enc:
            for param in self.encdec.encoder.parameters():
                param.requires_grad = False
        if freeze_dec_emb:
            for param in self.encdec.decoder.bert.embeddings.word_embeddings.parameters():
                param.requires_grad = False
        if freeze_dec:
            for param in self.encdec.decoder.bert.parameters():
                param.requires_grad = False

    def forward(self, x, y):
        xmask = x.ne(self.hparams.xpad).float()
        ymask = y.ne(self.hparams.ypad).float()
        out = self.encdec(
            input_ids=x,
            attention_mask=xmask,
            decoder_input_ids=y,
            decoder_attention_mask=ymask
        )
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_in, y_label = y[:, :-1], y[:, 1:]
        out = self(x, y_in)
        logits = out.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_label.contiguous().view(-1),
            reduction='sum',
            ignore_index=self.hparams.ypad
        )
        ntokens = y_label.ne(self.hparams.ypad).float().sum()
        avg_loss = loss / ntokens

        # -- logging
        self.metrics_train.update(loss.item(), logits.size(0), ntokens.item())
        if batch_idx % self.hparams.log_every == 0:
            self.log_dict(self.metrics_train.report(), prog_bar=True)
            self.metrics_train.reset()

        return avg_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_in, y_label = y[:, :-1], y[:, 1:]
        out = self(x, y_in)
        logits = out.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_label.contiguous().view(-1),
            reduction='sum',
            ignore_index=self.hparams.ypad
        )
        ntokens = y_label.ne(self.hparams.ypad).float().sum()
        avg_loss = loss / ntokens

        # -- logging
        self.metrics_valid.update(loss.item(), logits.size(0), ntokens.item())

        for xj, yj in zip(x, y):
            xj = xj.unsqueeze(0)

            yhatj = self.encdec.generate(
                input_ids=xj,
                attention_mask=xj.ne(self.hparams.xpad).float(),
                max_length=self.hparams.decode_max_length,
                decoder_start_token_id=self.hparams.bos,
                no_repeat_ngram_size=1,
                do_sample=False,
                num_beams=1
            )

            ranked = utils.extract_rankings(
                self, xj, yhatj,
                use_first=True,
                use_generations=True
            )
            actuals = trim(self, yj.cpu().view(-1)).tolist()
            preds = trim(self, yhatj.cpu().view(-1)).tolist()

            self.metrics_valid.update_val(preds, actuals, ranked)

        return avg_loss

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
    def __init__(self, name, ypad):
        self._metrics = defaultdict(list)
        self.name = name
        self.ypad = ypad

    def update(self, loss, bsz, ntokens):
        self._metrics['ntokens'].append(ntokens)
        self._metrics['bsz'].append(bsz)
        self._metrics['loss'].append(loss)

    def _f1(self, preds, actuals):
        tp = 0.0
        fp = 0.0

        for pred in preds:
            if pred in actuals:
                tp += 1.0
            else:
                fp += 1.0

        recall_denom = len(actuals)
        pk = tp / max(tp + fp, 1.0)
        rk = tp / recall_denom
        if pk + rk > 0:
            f1 = 2.0 * pk * rk / (pk + rk)
        else:
            f1 = 0.0
        return f1

    def _mAP(self, ranked, actuals):
        mAP = []
        num_hits = 0.0
        for rank, rid in enumerate(ranked):
            if rid in actuals:
                num_hits += 1.0
                mAP.append(num_hits / (rank + 1.0))
        if len(mAP) == 0:
            return 0.0
        mAP = np.mean(mAP)
        return mAP

    def update_val(self, preds, actuals, ranked):
        mAP = self._mAP(ranked, actuals)
        self._metrics['mAP'].append(mAP)

        if preds is not None:
            f1 = self._f1(preds, actuals)
            self._metrics['f1'].append(f1)

    def report(self):
        out = {}
        for k, vs in self._metrics.items():
            if k == 'bsz':
                normalizer = len(self._metrics['bsz'])
            elif k == 'ntokens':
                normalizer = len(self._metrics['ntokens'])
            elif k == 'mAP' or k == 'f1':
                normalizer = len(vs)
            else:
                normalizer = np.sum(self._metrics['ntokens'])
            out['%s/%s' % (self.name, k)] = (
                np.sum(vs) / normalizer
            )
        return out

    def reset(self):
        self._metrics = defaultdict(list)


def cli_main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument('--expr-name', default='autoregressive')
    parser.add_argument(
        '--datapath',
        default='/data/tokenized__bert-base-cased.pkl'
    )
    parser.add_argument(
        '--encs-file',
        default='/path/to/encs.pt'
    )
    parser.add_argument(
        '--pretrained-retrieval-checkpoint',
        default='/path/to/best.ckpt'
    )
    parser.add_argument(
        '--default-root-dir',
        default='/output'
    )
    parser.add_argument('--checkpoint-path', default=None)
    parser.add_argument('--token-limit', type=int, default=16384)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--dataloader-workers', type=int, default=0)
    parser.add_argument('--model-type', default='bert-base-cased')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--check-val-every-n-epoch', type=int, default=5)
    parser.add_argument('--decode-max-length', type=int, default=20)

    parser.add_argument('--set-mode', type=int, default=1, choices=[0, 1])
    parser.add_argument('--order', default='ground-truth', choices=['ground-truth'])

    parser.add_argument('--freeze-enc', type=int, default=0, choices=[0, 1])
    parser.add_argument('--freeze-dec-emb', type=int, default=1, choices=[0, 1])
    parser.add_argument('--freeze-dec', type=int, default=0, choices=[0, 1])

    parser.add_argument('--init-enc', type=int, default=1, choices=[0, 1])
    parser.add_argument('--init-dec-emb', type=int, default=1, choices=[0, 1])
    parser.add_argument('--init-dec', type=int, default=1, choices=[0, 1])

    parser.add_argument('--parallel', type=int, default=0, choices=[0, 1])

    # --- Trainer/lightning
    parser.add_argument('--accumulate-grad-batches', type=int, default=1)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--accelerator', default='ddp')
    parser.add_argument('--precision', type=int, default=16)

    parser = SequenceRetriever.add_model_specific_args(parser)
    args = parser.parse_args()

    print(args)
    pl.seed_everything(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)

    print("Loading data (%s)" % args.datapath)
    ds_raw = pickle.load(open(args.datapath, 'rb'))
    rid2tok = ds_raw['rid2tok']
    dls = utils.get_dataloaders(
        ds_raw['tokenized'],
        xpad=tokenizer.pad_token_id,
        ypad=rid2tok['<pad>'],
        token_limit=args.token_limit,
        buffer_size=args.buffer_size,
        workers=args.dataloader_workers,
        set_mode=bool(args.set_mode),
        order=args.order
    )

    if args.parallel:
        from naturalproofs.encoder_decoder.model_joint import ParallelSequenceRetriever
        model = ParallelSequenceRetriever(
            vocab_size=len(rid2tok),
            bos=rid2tok['<bos>'],
            eos=rid2tok['<eos>'],
            xpad=tokenizer.pad_token_id,
            ypad=rid2tok['<pad>'],
            lr=args.lr,
            log_every=args.log_every,
            model_type=args.model_type,
        )
    else:
        model = SequenceRetriever(
            vocab_size=len(rid2tok),
            bos=rid2tok['<bos>'],
            eos=rid2tok['<eos>'],
            xpad=tokenizer.pad_token_id,
            ypad=rid2tok['<pad>'],
            lr=args.lr,
            log_every=args.log_every,
            model_type=args.model_type,
            decode_max_length=args.decode_max_length
        )

    if args.checkpoint_path is not None:
        print("Resuming from checkpoint (%s)" % args.checkpoint_path)
        model.load_from_checkpoint(args.checkpoint_path)

    else:
        print("Initializing model using pretrained retrieval models")
        model.initialize(
            encs_file=args.encs_file,
            ckpt_file=args.pretrained_retrieval_checkpoint,
            dataset_rid2tok=rid2tok,
            init_enc=args.init_enc,
            init_dec_emb=args.init_dec_emb,
            init_dec=args.init_dec,
        )

    model.freeze_parts(
        freeze_enc=args.freeze_enc,
        freeze_dec_emb=args.freeze_dec_emb,
        freeze_dec=args.freeze_dec,
    )

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        monitor='val/mAP',
        mode='max',
        dirpath='%s/%s' % (args.default_root_dir, args.expr_name),
        filename='best-{val/mAP:.4f}'
    )

    logger = TensorBoardLogger(
        save_dir='%s/tb_logs' % (args.default_root_dir),
        name=args.expr_name
    )

    trainer = pl.Trainer(
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        default_root_dir=args.default_root_dir,
        reload_dataloaders_every_epoch=True,
        move_metrics_to_cpu=True,
        gradient_clip_val=args.gradient_clip_val,
        gpus=args.gpus,
        accelerator=args.accelerator,
        precision=args.precision,
        resume_from_checkpoint=args.checkpoint_path,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=logger
    )

    if args.mode == 'train':
        trainer.fit(model, dls['train'], dls['valid'])


if __name__ == '__main__':
    cli_main()
