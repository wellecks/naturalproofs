import pytorch_lightning as pl
import argparse
import os
import pickle
import naturalproofs.utils as utils
import naturalproofs.model as mutils
import torch
import transformers
from collections import defaultdict
from tqdm import tqdm
import numpy as np

def eval_with_refset(ex_dl, ref_dl, model, model_type):
    # pre-encode all references, store on CPU as a map from rid to vector
    model.eval()
    model.cuda()
    with torch.no_grad():
        r_encs = []
        rids = []
        print("Pre-encoding references...")
        for r, rid in tqdm(ref_dl, total=len(ref_dl)):
            r = r.cuda()
            r_enc = model.encode_r(r)
            r_encs.append(r_enc.cpu())
            rids.append(rid.cpu())
        r_encs = torch.cat(r_encs, 0)
        rids = torch.cat(rids, 0)
        print("Pre-encoding examples...")
        x_encs = []
        xids = []
        for x, xid in tqdm(ex_dl, total=len(ex_dl)):
            x = x.cuda()
            x_enc = model.encode_x(x)
            x_encs.append(x_enc.cpu())
            xids.append(xid.cpu())
        x_encs = torch.cat(x_encs, 0)
        xids = torch.cat(xids, 0).tolist()

        print("Evaluating pairs...")
        bsz = 10000
        nbs = max(int(np.ceil(rids.size(0)/bsz)), 1)
        x2ranked = defaultdict(list)
        for i in tqdm(range(x_encs.size(0))):
            x_enc = x_encs[i]
            x_enc = x_encs[i].unsqueeze(0).expand(bsz, x_enc.size(-1))
            x_enc = x_enc.cuda()

            xid = xids[i]
            for j in range(nbs):
                r_enc = r_encs[j*bsz:(j+1)*bsz]
                r_enc = r_enc.cuda()

                x_enc_ = x_enc[:r_enc.size(0)]
                logits = model.forward_clf(x_enc_, r_enc)

                if model_type == 'bert-no-training':
                    probs = logits
                else:
                    probs = torch.sigmoid(logits)
                rid = rids[j*bsz:(j+1)*bsz]
                pairs = list(zip(probs.tolist(), rid.tolist()))
                x2ranked[xid].extend(pairs)

            x2ranked[xid] = sorted(x2ranked[xid], key=lambda x: -x[0])
    return x2ranked, rids, r_encs, x_encs, xids


def cli_main():
    pl.seed_everything(42)

    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='bert-base-cased', help='name tag stored in the eval file')
    parser.add_argument('--split', default='valid', choices=['valid', 'test'])
    parser.add_argument(
        '--checkpoint-path',
        default='/path/to/best.ckpt'
    )
    parser.add_argument(
        '--output-dir',
        default='/output'
    )
    parser.add_argument(
        '--datapath',
        default='/data/dataset_tokenized__bert-base-cased_200.pkl'
    )
    parser.add_argument(
        '--datapath-base',
        default='/data/dataset.json'
    )
    parser.add_argument('--token-limit', type=int, default=8192)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--dataloader-workers', type=int, default=0)
    parser.add_argument('--model-type', default='bert-base-cased')
    parser.add_argument('--tokenizer-type', default='bert-base-cased', help='only used when model-type is lstm')

    # --- Trainer/lightning
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--accelerator', default='ddp')
    parser.add_argument('--precision', type=int, default=16)

    # for no-training `aggregate` argument
    import naturalproofs.model_bert_no_training
    parser = naturalproofs.model_bert_no_training.Classifier.add_model_specific_args(parser)

    args = parser.parse_args()

    if args.model_type == 'lstm':
        import naturalproofs.model_lstm as model_lstm
        model = model_lstm.Classifier.load_from_checkpoint(args.checkpoint_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_type)
    elif args.model_type == 'bert-no-training':
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

    xdl, rdl, x2rs = utils.get_eval_dataloaders(
        ds_raw,
        pad=tokenizer.pad_token_id,
        token_limit=args.token_limit,
        workers=args.dataloader_workers,
        split_name=args.split
    )

    print("%d examples\t%d refs" % (len(xdl.dataset.data), len(rdl.dataset.data)))
    x2ranked, rids, r_encs, x_encs, xids = eval_with_refset(xdl, rdl, model, args.model_type)

    outfile = os.path.join(args.output_dir, 'encs.pt')
    torch.save({
        'x_encs': x_encs,
        'r_encs': r_encs,
        'rids': rids,
        'xids': xids
    }, open(outfile, 'wb'))
    print("\nWrote to %s" % outfile)

    outfile = os.path.join(args.output_dir, 'eval.pkl')
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
