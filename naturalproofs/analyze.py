import argparse
import os
import pickle
import json
from collections import defaultdict
import numpy as np
from pathlib import Path

refs = None
id2ref = None

# If k is specified, it will first truncate to k, then filter by type
def filter_by_type(ids, t=None, k=None):
    global refs
    if k is not None:
        ids = ids[:k]
    if t is None:
        return ids
    elif t == 'theorem' or t == 'definition':
        return [id for id in ids if refs[id]['type'] == t]
    else:
        return [id for id in ids if refs[id]['type'] != 'theorem' and refs[id]['type'] != 'definition']


def _avg_precision(preds, actuals, k, t=None):
    actuals = filter_by_type(actuals, t)

    if len(actuals) == 0:
        return None

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(preds):
        if p in actuals:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
        if num_hits == len(actuals):
            break
    return score / (min(len(actuals), k) if k is not None else len(actuals))


def _f1_at_k(preds, actuals, k, t=None):
    preds = filter_by_type(preds, t, k=k)
    actuals = filter_by_type(actuals, t)

    assert k > 0
    if len(actuals) == 0:
        return 1.0, 1.0, 1.0

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
        f1 = 2.0*pk*rk / (pk + rk)
    else:
        f1 = 0.0
    return pk, rk, f1


def _f1_at_k_micro(x2ranked, x2rs, k, t=None, cat=None):
    global id2ref
    assert k > 0
    tp = 0.0
    fp = 0.0
    recall_denom = 0.0

    for xid, ranked in x2ranked.items():
        if cat is not None and cat not in id2ref[xid]['recursive_categories']:
            continue

        preds = [r[1] for r in ranked]
        actuals = x2rs[xid]

        preds = filter_by_type(preds, t, k=k)
        actuals = filter_by_type(actuals, t)

        if len(actuals) == 0:
            continue

        for pred in preds:
            if pred in actuals:
                tp += 1.0
            else:
                fp += 1.0

        recall_denom += len(actuals)

    pk = tp / max(tp + fp, 1.0)
    rk = tp / recall_denom if recall_denom > 0.0 else 0.0
    if pk + rk > 0:
        f1 = 2.0*pk*rk / (pk + rk)
    else:
        f1 = 0.0
    return pk, rk, f1


def _fully_predicted_at_k(preds, actuals, k, t=None):
    preds = filter_by_type(preds, t, k=k)
    actuals = filter_by_type(actuals, t)

    for a in actuals:
        if a not in preds:
            return 0.0
    return 1.0


def ranking_metrics(x2ranked, x2rs, t=None, cat=None):
    global id2ref
    metrics_ = defaultdict(list)

    for k in [10, 100]:
        pk, rk, f1k = _f1_at_k_micro(x2ranked, x2rs, k=k, t=t, cat=cat)
        metrics_['p@%d_micro' % k] = pk
        metrics_['r@%d_micro' % k] = rk
        metrics_['f1@%d_micro' % k] = f1k

    for xid, ranked in x2ranked.items():
        if cat is not None and cat not in id2ref[xid]['recursive_categories']:
            continue

        preds = [r[1] for r in ranked]
        actuals = x2rs[xid]

        mAP = _avg_precision(preds, actuals, k=None, t=t)
        if mAP is not None:
            metrics_['mAP'].append(mAP)
        mAP_100 = _avg_precision(preds, actuals, k=100, t=t)
        if mAP_100 is not None:
            metrics_['mAP_100'].append(mAP_100)

        metrics_['full_at_10'].append(
            _fully_predicted_at_k(preds, actuals, k=10, t=t)
        )
        metrics_['full_at_100'].append(
            _fully_predicted_at_k(preds, actuals, k=100, t=t)
        )

    # store return value after (if needed) normalizing
    metrics = {}
    for k, vs in metrics_.items():
        if '_micro' not in k:
            metrics[k] = np.mean(vs)
        else:
            metrics[k] = vs

    return metrics


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        required=True
    )
    parser.add_argument(
        '--eval-path',
        default='/eval.pkl'
    )
    parser.add_argument(
        '--datapath-base',
        default='/data/dataset.json'
    )
    parser.add_argument(
        '--cat',
        default=None
    )

    args = parser.parse_args()

    print('Analyzing for %s' % args.method)

    print("Loading data")
    raw_ds = json.load(open(args.datapath_base, 'r'))
    global refs
    refs = raw_ds['dataset']['theorems'] + raw_ds['dataset']['definitions'] + raw_ds['dataset']['others']
    global id2ref
    id2ref = {ref['id'] : ref for ref in refs}

    print("Loading evalfile")
    evalfile = args.eval_path
    dic = pickle.load(open(evalfile, 'rb'))
    x2ranked = dic['x2ranked']
    x2rs = dic['x2rs']

    print("Computing metrics")
    metrics = ranking_metrics(x2ranked, x2rs, cat=args.cat)
    for k in ['mAP', 'r@10_micro', 'r@100_micro', 'full_at_10', 'full_at_100']:
        v = metrics[k]
        v = round(100 * v, 2)
        print(k, v)
    print('')
    for t in ['theorem', 'definition', 'other']:
        metrics_by_type = ranking_metrics(x2ranked, x2rs, t=t)
        print(t)
        if 'mAP' in metrics_by_type:
            print('mAP', metrics_by_type['mAP'])
    print('')

    outfile = os.path.join(
        Path(args.eval_path).parent.as_posix(), '%s__analysis.pkl' % args.method
    )
    pickle.dump({
        'metrics': metrics,
    }, open(outfile, 'wb'))

    print("\nWrote to %s" % outfile)
    print("== done.")


if __name__ == '__main__':
    cli_main()
