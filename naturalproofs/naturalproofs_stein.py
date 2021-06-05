import re
from collections import defaultdict
import json
import os
import urllib.request
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='./data')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

outdir = args.outdir
dataset_name = 'nt-stein'
metadata = {
    'nt-stein': {
        'filenames': [
            './nt-stein/body.tex',
        ],
        'tex': {
            './nt-stein/body.tex': urllib.request.urlopen(
                'https://raw.githubusercontent.com/williamstein/ent/ed95075ac1275859fed7e3a37e06ec623086cfae/body.tex'
            ).read().decode('utf-8')
        },
        'theorem_kinds': ['theorem', 'lemma', 'corollary', 'proposition'],
        'definition_kinds': ['definition'],
        'other_kinds': [],
        'ref_kinds': ['thm', 'lem', 'defn', 'def', 'cor', 'prop'],
        'proof_head': '\\begin{proof}',
        'proof_tail': '\\end{proof}',
        'out_filename': os.path.join(outdir, 'naturalproofs_stein.json'),
    }
}

filenames = metadata[dataset_name]['filenames']
filename_to_tex = metadata[dataset_name]['tex']
ref_kinds = metadata[dataset_name]['ref_kinds']
proof_head = metadata[dataset_name]['proof_head']
proof_tail = metadata[dataset_name]['proof_tail']
out_filename = metadata[dataset_name]['out_filename']

stems = [filename.split('/')[-1].split('.tex')[0] for filename in filenames]

all_ref_kinds = set()
for filename in filenames:
    tex = filename_to_tex[filename]
    labels_ = re.findall(r'\\ref{([^}]*)}', tex)
    for l in labels_:
        all_ref_kinds.add(l.split(':')[0])

def extract_refs(s):
    refs = re.findall(r'\\ref{([^}]*)}', s)
    refs = [ref for ref in refs if ref.split(':')[0] in ref_kinds]
    refs = ['%s-%s' % (stem, ref) for ref in refs]
    return refs

def extract_title(statement):
    title = ''
    if '[' in statement.split('\n')[0]:
        titles = re.findall(r'\[([^\]]*)\]', statement)
        title = titles[0]
        statement = statement.split('[%s]' % title)[-1]
        title = title.strip('$').replace('\n', ' ')

        hyperlinks = re.findall(r'\\href{[^}]*}[^{]*{[^}]*}', title)
        for hyperlink in hyperlinks:
            surfaces = re.findall(r'\\href{[^}]*}[^{]*{([^}]*)}', hyperlink)
            assert len(surfaces) == 1
            surface = surfaces[0]
            title = title.replace(hyperlink, surface)
    return title, statement

def parse_proof(statement):
    contents = statement.split('\n')
    contents = list(filter(lambda s: s != '', contents))
    refs = extract_refs(proof)

    return {
        'contents': contents,
        'refs': refs,
    }

def parse_item(statement):
    lines = statement.strip(' \n').split('\n')
    start = 0
    label = None
    for i, line in enumerate(lines):
        if '\\label' in line:
            label = re.findall(r'\\label{([^}]*)}', line)[0]
            start = i + 1
            break
    label = "%s-%s" % (stem, label)
    contents = lines[start:]
    contents = list(filter(lambda s: s != '', contents))
    refs = extract_refs(statement)

    return {
        'label': label,
        'categories': [],
        'title': '',
        'contents': contents,
        'refs': refs,
    }

theorem_kinds = metadata[dataset_name]['theorem_kinds']
definition_kinds = metadata[dataset_name]['definition_kinds']
other_kinds = metadata[dataset_name]['other_kinds']
all_ref_kinds = theorem_kinds + definition_kinds + other_kinds

kind2type = {}
for kind in theorem_kinds:
    kind2type[kind] = 'theorem'
for kind in definition_kinds:
    kind2type[kind] = 'definition'

theorems = []
definitions = []
others = []
label2id = {}
cnt = 0

for filename in filenames:
    tex = filename_to_tex[filename]
    stem = filename.split('/')[-1].split('.tex')[0]

    for kind in all_ref_kinds:
        splits = tex.split('\\begin{%s}' % kind)[1:]
        for split in splits:
            item = {
                'id': cnt,
                'type': kind2type[kind],
            }
            cnt += 1

            statement, other = split.split('\\end{%s}' % kind)
            title, statement = extract_title(statement)
            item.update(parse_item(statement))
            if item['label'] is None:
                continue
            item['title'] = title

            if kind in theorem_kinds:
                proof = None
                secs = other.split(proof_head)
                if len(secs) > 1:
                    proof = secs[1].strip(' \n')
                    proof = proof.split(proof_tail)[0].strip(' \n')
                    proof = parse_proof(proof)
                item['proofs'] = [proof] if proof is not None else []

                theorems.append(item)

            elif kind in definition_kinds:
                definitions.append(item)

            elif kind in other_kinds:
                others.append(item)

            label2id[item['label']] = item['id']

for item in theorems:
    item['ref_ids'] = [label2id[label] for label in item['refs']]
    for proof in item['proofs']:
        proof['ref_ids'] = [label2id[label] for label in proof['refs']]
for item in definitions:
    item['ref_ids'] = [label2id[label] for label in item['refs']]
for item in others:
    item['ref_ids'] = [label2id[label] for label in item['refs']]

retrieval_examples = [thm['id'] for thm in theorems if len(thm['proofs']) > 0 and len(thm['proofs'][0]['refs']) > 0]

dataset = {
    'theorems': theorems,
    'definitions': definitions,
    'others': others,
    'retrieval_examples': retrieval_examples,
}

refs = theorems + definitions + others

splits = defaultdict(set)
splits['eval_thms'] = retrieval_examples
splits['eval_refs'] = [ref['id'] for ref in refs]

for k in splits:
    splits[k] = list(splits[k])

final_splits = {'train': {}, 'valid': {}, 'test': {}}
final_splits['train']['ref_ids'] = []
final_splits['train']['examples'] = []
final_splits['valid']['ref_ids'] = splits['eval_refs']
final_splits['valid']['examples'] = []
final_splits['test']['ref_ids'] = splits['eval_refs']
final_splits['test']['examples'] = [(tid, 0) for tid in splits['eval_thms']]

js = {
    'dataset': dataset,
    'splits': final_splits,
}
assert len(final_splits['test']['ref_ids']) == 105
assert len(final_splits['test']['examples']) == 40

with open(out_filename, 'w') as f:
    json.dump(js, f, indent=4)