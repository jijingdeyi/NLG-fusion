"""Run once: offline CLIP text encoding for degradation token bank initialization."""

from argparse import ArgumentParser

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldt.token_init import save_degradation_token_init


def main():
    parser = ArgumentParser(description='Initialize degradation tokens with CLIP (offline, once).')
    parser.add_argument(
        '--out', type=str, default='checkpoint/degradation_tokens_init.pt',
        help='output path for CLIP text features',
    )
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    device = 'cpu' if args.no_cuda else 'cuda'
    path = save_degradation_token_init(args.out, device=device)
    print(f'Saved CLIP degradation token init to {path}')


if __name__ == '__main__':
    main()
