"""
Hybrid name+address retrieval: one rare-token bag holding both the name
content tokens and the address tokens (prefixed so they never cross-match).
A candidate ranks high when it shares rare tokens in either or both fields,
which covers matches whose name was replaced or written in another script
(address carries them) and matches with missing addresses (name carries them).
"""

from .token_blocks import token_channel


def hybrid_channel(name_view: str, address_view: str, max_df: int, top_k: int,
                   transforms: dict | None = None, ref_transforms: dict | None = None):
    return token_channel([name_view, address_view], max_df=max_df, top_k=top_k,
                         transforms=transforms, ref_transforms=ref_transforms)
