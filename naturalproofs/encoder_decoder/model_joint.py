import naturalproofs.encoder_decoder.utils as utils
import torch.nn.functional as F
import torch
from naturalproofs.encoder_decoder.utils import trim
from naturalproofs.encoder_decoder.model import SequenceRetriever


class ParallelSequenceRetriever(SequenceRetriever):
    """The joint retrieval model."""
    def __init__(self, vocab_size, bos, eos, xpad, ypad, lr, log_every, model_type='bert-base-cased'):
        super().__init__(vocab_size, bos, eos, xpad, ypad, lr, log_every, model_type)

    def _multihot_dist(self, y_label):
        vec = torch.zeros(y_label.size(0), self.hparams.vocab_size, device=y_label.device)
        vec = vec.scatter(1, y_label, 1)
        vec[:, self.hparams.eos] = 0
        vec[:, self.hparams.ypad] = 0
        vec = vec / vec.sum(1, keepdim=True)
        return vec

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

    def forward_ood(self, x, rs):
        y_in = torch.LongTensor(x.size(0), 1).to(x.device).fill_(self.hparams.bos)
        out = self.encdec(
            input_ids=x,
            attention_mask=x.ne(self.hparams.xpad).float(),
            decoder_input_ids=y_in,
            output_hidden_states=True
        )
        hidden = out.decoder_hidden_states[-1]
        hidden = self.encdec.decoder.cls.predictions.transform(hidden).squeeze(1)
        scores = hidden.matmul(rs.to(x.device).T)
        return scores

    def _get_loss(self, batch, batch_idx):
        x, y = batch
        y_in = y[:, :1]  # just use BOS for a single step prediction
        y_label = y[:, 1:]
        y_label_dist = self._multihot_dist(y_label)
        out = self(x, y_in)
        assert out.logits.size(1) == 1
        logits = out.logits.squeeze(1)
        logps = logits.log_softmax(-1)
        loss = F.kl_div(
            logps, y_label_dist, reduction='none'
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, batch_idx)
        loss = loss.sum(-1)
        avg_loss = loss.mean()

        # -- logging
        B = batch[0].size(0)
        sum_loss = loss.sum().item()
        self.metrics_train.update(sum_loss, B, B)  # always normalize by batch size
        if batch_idx % self.hparams.log_every == 0:
            self.log_dict(self.metrics_train.report(), prog_bar=True)
            self.metrics_train.reset()

        return avg_loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch, batch_idx)
        loss = loss.sum(-1)
        avg_loss = loss.mean()

        # -- logging
        B = batch[0].size(0)
        sum_loss = loss.sum().item()
        self.metrics_valid.update(sum_loss, B, B)  # always normalize by batch size
        x, y = batch
        for xj, yj in zip(x, y):
            xj = xj.unsqueeze(0)
            ranked = utils.extract_rankings(
                self, xj,
                yhatj=torch.tensor([[self.hparams.bos]], dtype=torch.long, device=xj.device),
                use_first=True,
                use_generations=True
            )
            actuals = trim(self, yj.cpu().view(-1)).tolist()
            self.metrics_valid.update_val(None, actuals, ranked)

        return avg_loss
