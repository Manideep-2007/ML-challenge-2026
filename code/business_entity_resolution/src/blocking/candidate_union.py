"""
Candidate union with provenance. Pairs are int64 keys (query_idx * n_ref +
ref_idx); each unique pair keeps a bitmask of the channels that produced it.
"""

import numpy as np


class CandidateUnion:
    def __init__(self, n_ref: int, channel_names: list[str]):
        self.n_ref = n_ref
        self.channel_names = list(channel_names)
        self.bits = {name: np.uint32(1 << i) for i, name in enumerate(self.channel_names)}
        self.keys = np.empty(0, dtype=np.int64)
        self.masks = np.empty(0, dtype=np.uint32)

    def pair_keys(self, query_idx: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
        return np.unique(query_idx.astype(np.int64) * self.n_ref + ref_idx.astype(np.int64))

    def add(self, keys: np.ndarray, channel: str):
        """keys: unique pair keys from one channel."""
        all_keys = np.concatenate([self.keys, keys])
        all_masks = np.concatenate([self.masks, np.full(len(keys), self.bits[channel], dtype=np.uint32)])
        order = np.argsort(all_keys, kind="stable")
        all_keys, all_masks = all_keys[order], all_masks[order]
        starts = np.flatnonzero(np.concatenate(([True], all_keys[1:] != all_keys[:-1])))
        self.keys = all_keys[starts]
        self.masks = np.bitwise_or.reduceat(all_masks, starts) if len(starts) else all_masks

    def subset(self, channels: list[str]) -> np.ndarray:
        """Pair keys produced by at least one of `channels`."""
        wanted = np.uint32(0)
        for name in channels:
            wanted |= self.bits[name]
        return self.keys[(self.masks & wanted) != 0]

    def query_ref(self, keys: np.ndarray | None = None):
        keys = self.keys if keys is None else keys
        return keys // self.n_ref, keys % self.n_ref

    def channel_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Per pair: 'a|b' channel label and number of channels."""
        distinct, inverse = np.unique(self.masks, return_inverse=True)
        labels = np.array(
            ["|".join(n for n in self.channel_names if m & self.bits[n]) for m in distinct],
            dtype=object,
        )
        counts = np.array([bin(int(m)).count("1") for m in distinct], dtype=np.int8)
        return labels[inverse], counts[inverse]
