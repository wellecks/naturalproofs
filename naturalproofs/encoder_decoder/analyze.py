import editdistance
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk import ngrams
from collections import defaultdict
import argparse
import os
import pickle
from pathlib import Path


def _f1_set(preds, actuals):
    assert len(actuals) > 0

    tp = 0.0
    fp = 0.0

    for pred in preds:
        if pred in actuals:
            tp += 1.0
        else:
            fp += 1.0

    prec = tp / len(preds) if len(preds) > 0 else 0.0
    rec = tp / len(actuals)
    if prec + rec > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    else:
        f1 = 0.0

    return prec, rec, f1


def _make_multiset(items):
    multiset = set()
    multiplicities = defaultdict(int)
    for item in items:
        id_ = multiplicities[item]
        multiplicities[item] += 1
        item_with_multiplicity = '%d_%d' % (item, id_)
        multiset.add(item_with_multiplicity)
    return multiset


def _f1_multiset(preds, actuals):
    assert len(actuals) > 0
    preds_ = _make_multiset(preds)
    actuals_ = _make_multiset(actuals)
    prec, rec, f1 = _f1_set(preds_, actuals_)
    return prec, rec, f1


def _f1_set_corpus(all_preds, all_actuals):
    assert len(all_preds) == len(all_actuals)

    tp = 0.0
    fp = 0.0
    recall_denom = 0.0

    for preds, actuals in zip(all_preds, all_actuals):
        if not isinstance(preds, set):
            preds = set(preds)
        if not isinstance(actuals, set):
            actuals = set(actuals)
        assert len(actuals) > 0

        for pred in preds:
            if pred in actuals:
                tp += 1.0
            else:
                fp += 1.0

        recall_denom += len(actuals)

    prec = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    rec = tp / recall_denom if recall_denom > 0.0 else 0.0
    if prec + rec > 0:
        f1 = 2.0*prec*rec / (prec + rec)
    else:
        f1 = 0.0
    return prec, rec, f1


def _f1_multiset_corpus(all_preds, all_actuals):
    all_preds_ = [_make_multiset(preds) for preds in all_preds]
    all_actuals_ = [_make_multiset(actuals) for actuals in all_actuals]
    prec, rec, f1 = _f1_set_corpus(all_preds_, all_actuals_)
    return prec, rec, f1


def repetition(items, n):
    if len(items) < n:
        return 0.0
    ngrams_ = list(ngrams(items, n))
    rep = 1.0 - (len(set(ngrams_)) / len(ngrams_))
    return rep


class GenerationMetrics(object):
    def __init__(self, name):
        self.name = name
        self._metrics = defaultdict(list)
        self._corpus_metrics = defaultdict(list)

    def update(self, ys, yhats):
        """Sentence-level"""
        for y_, yhat_ in zip(ys, yhats):
            dist = editdistance.eval(y_, yhat_)
            dist = min(dist / len(y_), 1)
            self._metrics['edit'].append(dist)

            # exact match
            em = y_ == yhat_
            self._metrics['em'].append(em)
            em_set = set(y_) == set(yhat_)
            self._metrics['em_set'].append(em_set)
            em_multiset = _make_multiset(y_) == _make_multiset(yhat_)
            self._metrics['em_multiset'].append(em_multiset)

            len_ = len(yhat_) / len(y_)
            self._metrics['len_pred_over_true'].append(len_)

            # Set precision/recall/F1
            prec, rec, f1 = _f1_set(preds=set(yhat_), actuals=set(y_))
            self._metrics['prec'].append(prec)
            self._metrics['rec'].append(rec)
            self._metrics['f1'].append(f1)

            # Multiset precision/recall/F1
            prec, rec, f1 = _f1_multiset(preds=yhat_, actuals=y_)
            self._metrics['multi_prec'].append(prec)
            self._metrics['multi_rec'].append(rec)
            self._metrics['multi_f1'].append(f1)

            # Repetition
            for n in [1, 2, 3, 4]:
                self._metrics['rep%d' % n].append(repetition(yhat_, n))
                self._metrics['rep%d_true' % n].append(repetition(y_, n))

    def update_corpus(self, ys_corpus, yhats_corpus):
        """Corpus-level."""
        ys_corpus_ = [[y] for y in ys_corpus]
        bleu = corpus_bleu(ys_corpus_, yhats_corpus)
        self._corpus_metrics['bleu'] = bleu

        bleu1 = corpus_bleu(ys_corpus_, yhats_corpus, weights=(1.0, 0.0, 0.0, 0.0))
        self._corpus_metrics['bleu1'] = bleu1

        bleu2 = corpus_bleu(ys_corpus_, yhats_corpus, weights=(0.5, 0.5, 0.0, 0.0))
        self._corpus_metrics['bleu2'] = bleu2

        # Set precision/recall/F1
        prec, rec, f1 = _f1_set_corpus(all_preds=yhats_corpus, all_actuals=ys_corpus)
        self._corpus_metrics['prec'] = prec
        self._corpus_metrics['rec'] = rec
        self._corpus_metrics['f1'] = f1

        # Multiset precision/recall/F1
        prec, rec, f1 = _f1_multiset_corpus(all_preds=yhats_corpus, all_actuals=ys_corpus)
        self._corpus_metrics['multi_prec'] = prec
        self._corpus_metrics['multi_rec'] = rec
        self._corpus_metrics['multi_f1'] = f1

    def report(self):
        out = {}
        for k in self._metrics:
            out[k] = np.mean(self._metrics[k])

        for k in self._corpus_metrics:
            out['corpus_' + k] = self._corpus_metrics[k]

        return out

    def reset(self):
        self._metrics = defaultdict(list)
        self._corpus_metrics = defaultdict(list)


def analyze(ys, yhats, split):
    metrics = GenerationMetrics(name=split)
    metrics.update(ys, yhats)
    metrics.update_corpus(ys, yhats)
    ms = metrics.report()
    return ms


def analyze_rankings(ys, rankings):
    from naturalproofs.analyze import ranking_metrics
    x2ranked = {}
    x2rs = {}
    for i in range(len(ys)):
        x2ranked[i] = [(j, r) for j, r in enumerate(rankings[i])]
        # discard duplicates (NOTE: also discards order)
        x2rs[i] = list(set(ys[i]))

    metrics = ranking_metrics(x2ranked, x2rs)
    return metrics


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evalpath',
        default='/data/eval.pkl'
    )
    args = parser.parse_args()

    evalfile = pickle.load(open(args.evalpath, 'rb'))
    ys, yhats = evalfile['out']['y'], evalfile['out']['yhat']
    rankings = evalfile['out']['ranked']
    split = evalfile['args']['split']

    metrics = analyze(ys, yhats, split)

    rmetrics = analyze_rankings(ys, rankings)

    print("\n== token-level metrics")
    for k, metric in evalfile['token-metrics'].items():
        print('\t%.5f\t%s' % (metric, k))

    print("\n== generation metrics")
    for k, metric in metrics.items():
        print('\t%.5f\t%s' % (metric, k))

    print("\n== ranking metrics")
    for k, metric in rmetrics.items():
        print('\t%.5f\t%s' % (metric, k))

    # append full_ to the evaluation file name
    path = Path(args.evalpath)
    outfile = os.path.join(
        str(path.parent),
        'full_%s' % path.name
    )
    evalfile['generation-metrics'] = metrics
    evalfile['ranking-metrics'] = rmetrics
    pickle.dump(evalfile, open(outfile, 'wb'))

    print("\nWrote to %s" % outfile)
    print("== done.")


if __name__ == '__main__':
    cli_main()
