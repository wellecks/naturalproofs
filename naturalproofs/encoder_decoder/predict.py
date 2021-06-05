import pytorch_lightning as pl
import argparse
import os
import pickle
from pathlib import Path
import naturalproofs.encoder_decoder.model as mutils
import naturalproofs.encoder_decoder.utils as utils
from naturalproofs.encoder_decoder.analyze import analyze, analyze_rankings
import numpy as np
import torch
import transformers
from collections import defaultdict
from tqdm import tqdm


def get_gold_baseline(gold_baseline, yj):
    if gold_baseline == 'multiset':
        # randomly permute the sequence to simulate a multiset
        perm = np.random.permutation(len(yj))
        yhat = [yj[perm[i]] for i in range(len(perm))]
    elif gold_baseline == 'set':
        perm = np.random.permutation(len(yj))
        yhat = [yj[perm[i]] for i in range(len(perm))]
        yhat = list(set(yhat))
    elif gold_baseline == 'half-seq':
        yhat = yj[:len(yj)//2]
    else:
        raise NotImplementedError(gold_baseline)
    return yhat


def predict(
        dl, model, decoder, num_beams, top_p, no_repeat_ngram_size, max_length, rank_use_first, rank_use_generations,
        modeltok2datatok, parallel,
        gold_baseline=None,
        ood_rencs=None,
        ood_tok2rid=None,
        ood_idx2rid=None
):
    decoder_kwargs = {
        'max_length': max_length,
        'decoder_start_token_id': model.hparams.bos,
        'no_repeat_ngram_size': no_repeat_ngram_size
    }
    if decoder == 'beam':
        decoder_kwargs['num_beams'] = num_beams
        decoder_kwargs['do_sample'] = False
    if decoder == 'sample':
        decoder_kwargs['do_sample'] = True
        decoder_kwargs['top_p'] = top_p
    print("Decoder additional keyword args:\n", decoder_kwargs)

    model.eval()
    model.cuda()
    model.metrics_valid.reset()

    with torch.no_grad():
        out = defaultdict(list)
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            x, y = batch
            x = x.cuda()
            y = y.cuda()

            # token-level metrics (same as training)
            model.validation_step((x, y), i)

            for xj, yj in zip(x, y):
                xj = xj.unsqueeze(0)

                # Joint model
                if parallel:
                    # OOD textbook evaluation
                    if ood_rencs is not None:
                        scores = model.forward_ood(xj, ood_rencs)
                        idx_ranks = scores.argsort(descending=True)[0].tolist()
                        ranked = [ood_idx2rid[idx] for idx in idx_ranks]
                        yhatj_ = ranked[:5]
                        yj_ = utils.trim(model, yj.cpu().view(-1)).tolist()
                        yj_ = [ood_tok2rid[tok] for tok in yj_]
                    # In-domain evaluation
                    else:
                        ranked = utils.extract_rankings(
                            model, xj,
                            yhatj=torch.tensor([[model.hparams.bos]], dtype=torch.long, device=xj.device),
                            use_first=True,
                            use_generations=False
                        )
                        yhatj_ = ranked[:5]  # retrieval-only baseline: sequence using the top-ranked elements
                        yj_ = utils.trim(model, yj.cpu().view(-1)).tolist()
                # Gold baselines for generation.
                elif gold_baseline is not None:
                    yj_ = utils.trim(model, yj.cpu().view(-1)).tolist()
                    yhatj_ = get_gold_baseline(gold_baseline, yj_)
                    ranked = yhatj_
                # Autoregressive
                else:
                    yhatj = model.encdec.generate(
                        input_ids=xj,
                        attention_mask=xj.ne(model.hparams.xpad).float(),
                        **decoder_kwargs
                    )

                    # For retrieval evaluation.
                    ranked = utils.extract_rankings(
                        model, xj, yhatj,
                        use_first=rank_use_first,
                        use_generations=rank_use_generations
                    )

                    yhatj_ = utils.trim(model, yhatj.cpu().view(-1)).tolist()
                    yj_ = utils.trim(model, yj.cpu().view(-1)).tolist()

                # For combined proofwiki+stacks model: align the training and eval output spaces.
                if modeltok2datatok is not None:
                    # convert output space
                    yhatj_ = [modeltok2datatok[tok] for tok in yhatj_ if tok in modeltok2datatok]
                    # convert output space and filter out items not in the target output space
                    ranked = [modeltok2datatok[tok] for tok in ranked if tok in modeltok2datatok]

                out['y'].append(yj_)
                out['yhat'].append(yhatj_)
                out['ranked'].append(ranked)

    ms_tok = model.metrics_valid.report()
    return out, ms_tok


def _name(args):
    if args.predict_only:
        name = 'eval'
    else:
        name = 'full_eval'
    if args.parallel:
        name += '__parallel'

    if args.gold_baseline is not None:
        name = 'full_eval__gold_baseline_%s__%s.pkl' % (args.gold_baseline, args.split)
        return name

    name += '__%s__%s' % (args.split, args.decoder)
    if args.decoder == 'beam':
        name += '__num_beams=%d' % args.num_beams
    elif args.decoder == 'sample':
        name += '__top_p=%.2f' % args.top_p
    else:
        raise NotImplementedError()

    if args.no_repeat_ngram_size > 0:
        name += '__no_repeat_ngram_size=%d' % args.no_repeat_ngram_size

    if not args.rank_use_first:
        name += '__use_first=%d' % args.rank_use_first

    if not args.rank_use_generations:
        name += '__use_generations=%d' % args.rank_use_generations

    name += '__max_length=%d' % args.max_length
    name += '.pkl'
    return name


def cli_main():
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='valid', choices=['valid', 'test'])
    parser.add_argument(
        '--checkpoint-path',
        default='/checkpoint/best.ckpt'
    )
    parser.add_argument(
        '--output-dir',
        default='/output'
    )
    parser.add_argument(
        '--datapath',
        default='/data/autoregressive_tokenized__bert-base-cased.pkl'
    )

    parser.add_argument('--gold-baseline', type=str, default=None, choices=['multiset', 'set', 'half-seq'])

    parser.add_argument('--parallel', type=int, default=0, choices=[0, 1])
    parser.add_argument(
        '--modeltok2datatok',
        default=None,
        help='Both (P+S) model only: file for converting pw+stacks tokens to pw or stacks tokens'
    )
    parser.add_argument(
        '--ood-encs',
        default=None,
        help='OOD textbook evaluation only: reference encodings for OOD evaluation'
    )

    # Ranked-list strategy
    parser.add_argument('--rank-use-first', type=int, default=1, choices=[0, 1])
    parser.add_argument('--rank-use-generations', type=int, default=1, choices=[0, 1])

    # General decoding
    parser.add_argument('--decoder', type=str, default='beam', choices=['beam', 'sample'])
    parser.add_argument('--no-repeat-ngram-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=20)

    # Beam search
    parser.add_argument('--num-beams', type=int, default=1)

    # Sampling
    parser.add_argument('--top-p', type=float, default=1.0)

    parser.add_argument('--set-mode', type=int, default=1, choices=[0, 1])
    parser.add_argument('--order', default='ground-truth', choices=['ground-truth', 'ascending-id'])

    parser.add_argument('--token-limit', type=int, default=8192)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--dataloader-workers', type=int, default=0)
    parser.add_argument('--model-type', default='bert-base-cased')
    parser.add_argument(
        '--predict-only',
        action='store_true',
        help='when set, does not run analysis in analysis.py.'
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.parallel:
        from naturalproofs.encoder_decoder.model_joint import ParallelSequenceRetriever
        model = ParallelSequenceRetriever.load_from_checkpoint(args.checkpoint_path)
    else:
        model = mutils.SequenceRetriever.load_from_checkpoint(args.checkpoint_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)

    # This is used when using a model trained on the _combined_ proofwiki+stacks dataset.
    # We have to align the reference ids (via tokens) from the combined output space with the
    # evaluation output space (i.e. the individual proofwiki or individual stacks dataset).
    if args.modeltok2datatok is not None:
        temp = pickle.load(open(args.modeltok2datatok, 'rb'))
        if 'proofwiki' in args.datapath:
            modeltok2datatok = temp['both2pw']
        else:
            modeltok2datatok = temp['both2stacks']
    else:
        modeltok2datatok = None

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

    if args.ood_encs is not None:  # used for OOD textbook evaluation
        rencs = torch.load(args.ood_encs)
        ood_tok2rid = {tok: rid for rid, tok in rid2tok.items()}
        ood_idx2rid = {idx: rid.item() for idx, rid in enumerate(rencs['rids'])}
        r_encs = rencs['r_encs'].clone()
    else:
        r_encs = None
        ood_tok2rid = None
        ood_idx2rid = None

    dl = dls[args.split]
    print("%d examples" % (len(dl.dataset.data)))

    out, token_metrics = predict(
        dl=dl, model=model,
        decoder=args.decoder,
        num_beams=args.num_beams,
        top_p=args.top_p,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_length=args.max_length,
        rank_use_first=args.rank_use_first,
        rank_use_generations=args.rank_use_generations,
        modeltok2datatok=modeltok2datatok,
        parallel=args.parallel,
        gold_baseline=args.gold_baseline,
        ood_rencs=r_encs,
        ood_tok2rid=ood_tok2rid,
        ood_idx2rid=ood_idx2rid
    )

    ys, yhats = out['y'], out['yhat']
    gms = analyze(ys, yhats, args.split)

    rmetrics = analyze_rankings(ys, out['ranked'])
    print("\n== ranking metrics")
    for k, metric in rmetrics.items():
        print('\t%.5f\t%s' % (metric, k))

    print("\n== generation metrics")
    for k, metric in gms.items():
        print('\t%.5f\t%s' % (metric, k))

    output_file_contents = {
        'out': out,
        'rid2tok': rid2tok,
        'token_metrics': token_metrics,
        'ranking-metrics': rmetrics,
        'generation-metrics': gms,
        'args': vars(args),
    }
    outfile = os.path.join(args.output_dir, _name(args))
    pickle.dump(output_file_contents, open(outfile, 'wb'))

    print("\nWrote to %s" % outfile)
    print("== done.")


if __name__ == '__main__':
    cli_main()
