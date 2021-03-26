# Baseline of simply using dot product with the pretrained BERT model

import pytorch_lightning as pl
import transformers
from argparse import ArgumentParser


class Classifier(pl.LightningModule):
    def __init__(self, pad_idx=0, model_type='bert-base-cased', aggregate='avg'):
        super().__init__()
        self.save_hyperparameters()

        self.x_encoder = transformers.AutoModel.from_pretrained(model_type)
        self.r_encoder = transformers.AutoModel.from_pretrained(model_type)

    def forward(self, x, r):
        x_enc = self.encode_x(x)
        r_enc = self.encode_r(r)
        score = (x_enc*r_enc).sum(1).view(-1)
        return score

    def encode_x(self, x):
        xmask = x.ne(self.hparams.pad_idx).float()
        if self.hparams.aggregate == 'cls':
            x_enc = self.x_encoder(x, attention_mask=xmask)[0][:, 0]
        elif self.hparams.aggregate == 'avg':
            x_enc = (self.x_encoder(x, attention_mask=xmask)[0] * xmask.unsqueeze(-1)).sum(1) / xmask.sum(1, keepdim=True)
        else:
            raise NotImplementedError(self.hparams.aggregate)
        return x_enc

    def encode_r(self, r):
        rmask = r.ne(self.hparams.pad_idx).float()
        if self.hparams.aggregate == 'cls':
            r_enc = self.r_encoder(r, attention_mask=rmask)[0][:, 0]
        elif self.hparams.aggregate == 'avg':
            r_enc = (self.r_encoder(r, attention_mask=rmask)[0] * rmask.unsqueeze(-1)).sum(1) / rmask.sum(1, keepdim=True)
        else:
            raise NotImplementedError(self.hparams.aggregate)
        return r_enc

    def forward_clf(self, x_enc, r_enc):
        score = (x_enc*r_enc).sum(1).view(-1)
        return score

    def training_step(self, batch, batch_idx):
        raise NotImplementedError('this model is not meant to be trained!')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--aggregate', type=str, default='avg', choices=['avg', 'cls'])
        return parser

