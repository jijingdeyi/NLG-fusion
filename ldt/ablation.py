"""Token-ablation variants for Table tab:token-ablation (single CLI switch)."""

from __future__ import annotations

# Paper row labels (row 2 uses Text-IF in tab:llvip instead — not implemented here).
TOKEN_ABLATION_VARIANTS = (
    'full',           # (5) learnable CLIP-init tokens + router
    'no_token',       # (1) no token / router / SRC
    'freeze_token',   # (3) frozen CLIP-init tokens + router
    'no_router',      # (4) learnable tokens, uniform mix (no image router)
    'random_token',   # (6) random-init tokens + router
    'online_clip',    # (2) not supported — compare Text-IF in tab:llvip
)

PAPER_ROW = {
    'no_token': '(1)',
    'online_clip': '(2)',
    'freeze_token': '(3)',
    'no_router': '(4)',
    'full': '(5)',
    'random_token': '(6)',
}


def validate_ablation_variant(name: str) -> str:
    if name not in TOKEN_ABLATION_VARIANTS:
        raise ValueError(
            f'Unknown ablation_variant="{name}". Choose from: {list(TOKEN_ABLATION_VARIANTS)}',
        )
    if name == 'online_clip':
        raise ValueError(
            'ablation_variant=online_clip is not implemented (requires online CLIP encoder). '
            'Use Text-IF scores in Table tab:llvip for row (2), or remove that row.',
        )
    return name
