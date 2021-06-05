from tqdm import tqdm
from collections import defaultdict
import json
import transformers
import re


def replace_links(lines):
    def __replace(line):
        matches = re.findall(r'(\[\[([^]]*)\]\])', line)
        for match in matches:
            full, inner = match
            splt = inner.split('|')
            if len(splt) == 1:
                txt = splt[0]
            elif len(splt) == 2:
                txt = splt[1]
            else:
                txt = ''.join(splt[1:])
            if full in line:
                line = line.replace(full, txt)
        return line
    lines_ = [
        __replace(line) for line in lines
    ]
    return lines_


# ---- data tokenization and loading
def _tokenize_examples(split, dataset, tokenizer, title_only=False, content_only=False):
    stats = defaultdict(int)
    examples = split['examples']
    id2thm = {thm['id'] : thm for thm in dataset['theorems']}

    tokenized = []
    for eid, (tid, pix) in tqdm(enumerate(examples), total=len(examples)):
        ex = id2thm[tid]
        proof = ex['proofs'][pix]

        title = '' if content_only else ex['title']
        content = '' if title_only else ' '.join(replace_links(ex['contents']))
        inputs = "%s%s%s" % (
            title,
            tokenizer.sep_token,
            content
        )

        ids = tokenizer.encode(inputs)
        if len(ids) > tokenizer.model_max_length:
            ids = ids[:tokenizer.model_max_length-1] + [tokenizer.sep_token_id]
            stats['truncated'] += 1

        rids = sorted(set(proof['ref_ids']))
        tokenized.append({
            'x': ids,
            'rids': rids,
            'metadata': {
                'x_tokens': tokenizer.convert_ids_to_tokens(ids),
                'eid': eid
            }
        })
        if tokenizer.unk_token_id in ids:
            stats['hasunk'] += 1

    print("%d examples\n\t%d truncated\n\t%d have [UNK]" % (
        len(tokenized), stats['truncated'], stats['hasunk']
    ))
    return tokenized


def _tokenize_refs(split, dataset, tokenizer, title_only=False, content_only=False):
    stats = defaultdict(int)
    rids = split['ref_ids']
    refs = dataset['theorems'] + dataset['definitions'] + dataset['others']

    tokenized = {}
    for ref in tqdm(refs, total=len(refs)):
        if ref['id'] not in rids:
            continue
        title = '' if content_only else ref['title']
        content = '' if title_only else ' '.join(replace_links(ref['contents']))
        inputs = "%s%s%s" % (
            title,
            tokenizer.sep_token,
            content
        )

        ids = tokenizer.encode(inputs)
        if len(ids) > tokenizer.model_max_length:
            ids = ids[:tokenizer.model_max_length-1] + [tokenizer.sep_token_id]
            stats['truncated'] += 1

        tokenized[ref['id']] = {
            'r': ids,
            'metadata': {
                'r_tokens': tokenizer.convert_ids_to_tokens(ids),
                'rid': ref['id']
            }
        }
        if tokenizer.unk_token_id in ids:
            stats['hasunk'] += 1

    print("%d refs\n\t%d truncated\n\t%d have [UNK]" % (
        len(tokenized), stats['truncated'], stats['hasunk']
    ))
    reflist = [tokenized[r] for r in rids]
    return tokenized, reflist


def tokenize_dataset(
    filepath,
    model_type='bert-base-cased',
    ref_title_only=False,
    ex_title_only=False,
    ref_content_only=False,
    ex_content_only=False
):
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
    base = json.load(open(filepath, 'r'))
    dataset, splits = base['dataset'], base['splits']
    tokenized = {}

    print("---- assuming all refs are present in `valid` `ref_ids`")
    print("== tokenizing refs")
    rid2r, reflist = _tokenize_refs(
        splits['valid'], dataset, tokenizer,
        title_only=ref_title_only,
        content_only=ref_content_only
    )
    tokenized['refs'] = reflist

    for name, split in splits.items():
        print("== tokenizing %s" % name)
        xs = _tokenize_examples(
            split, dataset, tokenizer,
            title_only=ex_title_only,
            content_only=ex_content_only
        )
        tokenized[name] = {
            'rid2r': rid2r,
            'split_ref_ids': split['ref_ids'],
            'unmatched_data': xs
        }

    return tokenized


if __name__ == '__main__':
    import argparse
    import pickle
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default='/path/to/dataset_name.json')
    parser.add_argument('--output-path', default='/path/to/output')
    parser.add_argument('--model-type', default='bert-base-cased')
    parser.add_argument('--ref-title-only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ex-title-only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ref-content-only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ex-content-only', type=int, default=0, choices=[0, 1])

    parser.add_argument('--dataset-name', required=True)
    args = parser.parse_args()

    tokenized = tokenize_dataset(
        args.filepath,
        model_type=args.model_type,
        ref_title_only=bool(args.ref_title_only),
        ex_title_only=bool(args.ex_title_only),
        ref_content_only=bool(args.ref_content_only),
        ex_content_only=bool(args.ex_content_only)
    )

    suffix = '' if not args.ref_title_only else '_reftitleonly'
    suffix += '' if not args.ex_title_only else '_extitle_only'
    suffix += '' if not args.ref_content_only else '_refcontentonly'
    suffix += '' if not args.ex_content_only else '_excontentonly'
    outfile = os.path.join(args.output_path, 'pairwise_%s_tokenized__%s_%s.pkl' % (
        args.dataset_name, args.model_type.replace('/', '_'), suffix
    ))
    print("Writing dataset to %s" % outfile)
    with open(outfile, 'wb') as f:
        pickle.dump(tokenized, f)
    print("=== done.")

