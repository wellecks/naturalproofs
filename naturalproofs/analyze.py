import argparse
import pickle
import json
from collections import defaultdict
import numpy as np

refs = None

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
    preds = filter_by_type(preds, t, k=k)
    actuals = filter_by_type(actuals, t)

    if len(actuals) == 0:
        return 1.0

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(preds):
        if p in actuals:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
        if num_hits == len(actuals):
            break
    return score / min(len(actuals), k)


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


def _f1_at_k_macro(x2ranked, x2rs, k, t=None):
    assert k > 0
    tp = 0.0
    fp = 0.0
    recall_denom = 0.0

    for xid, ranked in x2ranked.items():
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


def _mrr_macro(x2ranked, x2rs, t=None):
    numer, denom = 0.0, 0.0

    for xid, ranked in x2ranked.items():
        preds = [r[1] for r in ranked]
        actuals = x2rs[xid]

        actuals = filter_by_type(actuals, t)

        for i, pred in enumerate(preds):
            if pred in actuals:
                numer += 1.0 / (i + 1.0)
                denom += 1.0

    mrr = numer / denom
    return mrr

def _fully_predicted_at_k(preds, actuals, k, t=None):
    preds = filter_by_type(preds, t, k=k)
    actuals = filter_by_type(actuals, t)

    for a in actuals:
        if a not in preds:
            return 0.0
    return 1.0


def ranking_metrics(x2ranked, x2rs, t=None):
    metrics_ = defaultdict(list)

    # macro-averaged
    for k in [10, 50, 100]:
        pk, rk, f1k = _f1_at_k_macro(x2ranked, x2rs, k=k, t=t)
        metrics_['p@%d_macro' % k] = pk
        metrics_['r@%d_macro' % k] = rk
        metrics_['f1@%d_macro' % k] = f1k

    metrics_['mrr_macro'] = _mrr_macro(x2ranked, x2rs, t=t)

    # micro-averaged
    for xid, ranked in x2ranked.items():
        preds = [r[1] for r in ranked]
        actuals = x2rs[xid]

        metrics_['avg_prec_100'].append(
            _avg_precision(preds, actuals, k=100, t=t)
        )
        metrics_['avg_prec_1000'].append(
            _avg_precision(preds, actuals, k=1000, t=t)
        )
        metrics_['full_at_100'].append(
            _fully_predicted_at_k(preds, actuals, k=100, t=t)
        )
        metrics_['full_at_1000'].append(
            _fully_predicted_at_k(preds, actuals, k=1000, t=t)
        )

        for k in [10, 50, 100]:
            pk, rk, f1k = _f1_at_k(preds, actuals, k=k, t=t)
            metrics_['p@%d' % k].append(pk)
            metrics_['r@%d' % k].append(rk)
            metrics_['f1@%d' % k].append(f1k)

    # store return value after (if needed) normalizing
    metrics = {}
    for k, vs in metrics_.items():
        if '_macro' not in k:
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
        '--analysis-path',
        default='/analysis.pkl'
    )
    parser.add_argument(
        '--datapath-base',
        default='/data/dataset.json'
    )

    args = parser.parse_args()

    print('Analyzing %s' % args.method)

    print("Loading data")
    raw_ds = json.load(open(args.datapath_base, 'r'))
    global refs
    refs = raw_ds['dataset']['theorems'] + raw_ds['dataset']['definitions'] + raw_ds['dataset']['other']

    print("Loading evalfile")
    evalfile = args.eval_path
    dic = pickle.load(open(evalfile, 'rb'))
    x2ranked = dic['x2ranked']
    x2rs = dic['x2rs']

    print("Computing metrics")
    metrics = ranking_metrics(x2ranked, x2rs)
    for k, v in metrics.items():
        print(k, v)
    metrics_by_type = {}
    for t in ['theorem', 'definition', 'other']:
        metrics_by_type[t] = ranking_metrics(x2ranked, x2rs, t=t)

    outfile = args.analysis_path
    pickle.dump({
        'metrics': metrics,
        'metrics_by_type': metrics_by_type
    }, open(outfile, 'wb'))

    print("\nWrote to %s" % outfile)
    print("== done.")


if __name__ == '__main__':
    cli_main()
