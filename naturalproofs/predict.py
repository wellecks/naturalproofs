import pytorch_lightning as pl
import argparse
import os
import pickle
import naturalproofs.dataloaders as dataloaders
import naturalproofs.model as mutils
import torch
import transformers
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path


def eval_with_refset(ex_dl, ref_dl, model):
    model.eval()
    model.cuda()
    with torch.no_grad():
        r_encs, rids = model.pre_encode_refs(ref_dl, progressbar=True)

        print("Evaluating pairs...")
        x_encs = []
        xids = []
        x2ranked = defaultdict(list)
        for x, xid in tqdm(ex_dl, total=len(ex_dl)):
            x = x.cuda()
            x_enc = model.encode_x(x) # (B, D)
            logits = model.forward_clf(x_enc, r_encs) # (B, R)
            for b in range(logits.size(0)):
                ranked = list(zip(logits[b].tolist(), rids.tolist()))
                ranked = sorted(ranked, reverse=True)
                x2ranked[xid[b].item()] = ranked
            x_encs.append(x_enc)
            xids.append(xid)
        x_encs = torch.cat(x_encs, 0)
        xids = torch.cat(xids, 0)
    return x2ranked, rids, r_encs, x_encs, xids


def cli_main():
    pl.seed_everything(42)
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='valid', choices=['valid', 'test'])
    parser.add_argument(
        '--method',
        required=True
    )
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
        default='/data/dataset_tokenized.pkl'
    )
    parser.add_argument(
        '--datapath-base',
        default='/data/dataset.json'
    )
    parser.add_argument('--token-limit', type=int, default=8192)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--dataloader-workers', type=int, default=0)
    parser.add_argument('--model-type', default='bert-base-cased')
    parser.add_argument('--tokenizer-type', default='bert-base-cased')

    # --- Trainer/lightning
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--accelerator', default='ddp')
    parser.add_argument('--precision', type=int, default=16)

    # for no-training `aggregate` argument
    import naturalproofs.model_bert_no_training
    parser = naturalproofs.model_bert_no_training.Classifier.add_model_specific_args(parser)

    args = parser.parse_args()

    if args.model_type == 'bert-no-training':
        import naturalproofs.model_bert_no_training as model_bert
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
        model = model_bert.Classifier(
            pad_idx=tokenizer.pad_token_id,
            model_type='bert-base-cased',
            aggregate=args.aggregate
        )
    else:
        args.tokenizer_type = args.model_type
        model = mutils.Classifier.load_from_checkpoint(args.checkpoint_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)

    print("Loading data")
    ds_raw = pickle.load(open(args.datapath, 'rb'))

    xdl, rdl, x2rs = dataloaders.get_eval_dataloaders(
        ds_raw,
        pad=tokenizer.pad_token_id,
        token_limit=args.token_limit,
        workers=args.dataloader_workers,
        split_name=args.split
    )

    print("%d examples\t%d refs" % (len(xdl.dataset.data), len(rdl.dataset.data)))
    x2ranked, rids, r_encs, x_encs, xids = eval_with_refset(xdl, rdl, model)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    outfile = os.path.join(args.output_dir, '%s__encs.pt' % args.method)
    torch.save({
        'x_encs': x_encs,
        'r_encs': r_encs,
        'rids': rids,
        'xids': xids
    }, open(outfile, 'wb'))
    print("\nWrote to %s" % outfile)

    outfile = os.path.join(args.output_dir, '%s__eval.pkl' % args.method)
    pickle.dump({
        'x2ranked': x2ranked,
        'x2rs': x2rs,
        'rids': rids,
        'name': args.method
    }, open(outfile, 'wb'))

    print("\nWrote to %s" % outfile)
    print("== done.")


if __name__ == '__main__':
    cli_main()
