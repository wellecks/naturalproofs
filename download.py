import gdown
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', default='./')
parser.add_argument('--naturalproofs', action='store_true')
parser.add_argument('--tokenized', action='store_true')
parser.add_argument('--checkpoint', action='store_true')
parser.add_argument('--other', action='store_true')

args = parser.parse_args()
os.makedirs(args.savedir, exist_ok=True)

if args.naturalproofs:
    url = 'https://drive.google.com/uc?id=1vgohULQD7HfbotskkVX4li9YanIeG3u1'
    out = os.path.join(args.savedir, 'naturalproofs.tar.gz')
    gdown.download(url, out, quiet=False)
    gdown.extractall(out, os.path.join(args.savedir, 'data'))
    process = subprocess.Popen(
        ['python', 'naturalproofs/naturalproofs_stein.py', '--outdir', os.path.join(args.savedir, 'data')]
    ).wait()

if args.tokenized:
    url = 'https://drive.google.com/uc?id=1OCIvcCyKTyRJeV7QiHdtQQhPJ6QknMpV'
    out = os.path.join(args.savedir, 'tok.tar.gz')
    gdown.download(url, out, quiet=False)
    gdown.extractall(out, os.path.join(args.savedir, 'data'))

if args.checkpoint:
    url = 'https://drive.google.com/uc?id=1uIBeI7fw5vJBhDOl2WL3SbXWmzHgfK3W'
    out = os.path.join(args.savedir, 'ckpt.tar.gz')
    gdown.download(url, out, quiet=False)
    gdown.extractall(out, os.path.join(args.savedir, 'ckpt'))

if args.other:
    url = 'https://drive.google.com/uc?id=178UxFLXOYH9CeIRkMcAWm_DJ3omI_eik'
    out = os.path.join(args.savedir, 'other.tar.gz')
    gdown.download(url, out, quiet=False)
    gdown.extractall(out, os.path.join(args.savedir, 'other'))



