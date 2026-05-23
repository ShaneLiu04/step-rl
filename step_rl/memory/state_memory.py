"""
State Memory Module v2.0
- Deterministic hashing for reproducible state deduplication
- Loop detection with penalty
- Novelty bonus for exploration
"""

import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StateSignature:
    """Compact representation of a state for deduplication."""

    url_hash: str = ""
    dom_hash: str = ""
    visual_hash: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def combined_hash(self) -> str:
        parts = [self.url_hash, self.dom_hash]
        if self.visual_hash:
            parts.append(self.visual_hash)
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


class StateMemory:
    """
    Tracks visited states, detects loops, and provides novelty bonuses.
    Uses deterministic hashing for reproducibility across runs.
    """

    def __init__(
        self,
        hash_method: str = "minhash",  # minhash, simple
        max_states: int = 500,
        loop_window: int = 3,
        loop_penalty_base: float = -0.1,
        novelty_bonus_base: float = 0.05,
    ):
        self.hash_method = hash_method
        self.max_states = max_states
        self.loop_window = loop_window
        self.loop_penalty_base = loop_penalty_base
        self.novelty_bonus_base = novelty_bonus_base

        # Use OrderedDict for true LRU eviction (Python 3.7+ dict preserves insertion order)
        self._visited_hashes: OrderedDict[str, None] = OrderedDict()
        self._state_history: Deque[str] = deque(maxlen=200)
        self._loop_counter: Dict[str, int] = {}
        self._visit_count: Dict[str, int] = {}

    # -----------------------------
    # Hashing
    # -----------------------------

    def compute_hash(
        self,
        observation_text: str,
        url: str = "",
        screenshot: Optional[np.ndarray] = None,
    ) -> str:
        """Compute a deterministic hash for the current state."""
        if self.hash_method == "simple":
            return self._simple_hash(observation_text, url)
        elif self.hash_method == "minhash":
            return self._minhash(observation_text, url)
        else:
            return self._simple_hash(observation_text, url)

    def _simple_hash(self, text: str, url: str) -> str:
        """Fast but coarse-grained hash."""
        content = f"{url}|{text[:500]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _minhash(self, text: str, url: str, num_perm: int = 64) -> str:
        """
        Deterministic MinHash on word shingles.
        Uses precomputed hash functions instead of calling MD5 in a tight loop.
        Falls back to simple hash for very short text.
        """
        words = text.lower().split()

        # Fallback for short text: no meaningful shingles
        if len(words) < 2:
            return self._simple_hash(text, url)

        # Build shingles
        shingles = set()
        for i in range(len(words) - 1):
            shingles.add(f"{words[i]} {words[i+1]}")

        # Use a fast precomputed permutation approach:
        # For each permutation, compute (hash(s) * a + b) % p and take min.
        # Constants chosen to give good distribution.
        p = (1 << 61) - 1
        hashes = []
        for seed in range(num_perm):
            a = (seed * 1234567891 + 1) & 0xFFFFFFFFFFFFFFFF
            b = (seed * 9876543211 + 1) & 0xFFFFFFFFFFFFFFFF
            min_val = p
            for s in shingles:
                # Fast deterministic hash of string
                h = int(hashlib.md5(s.encode()).hexdigest(), 16)
                perm = ((h * a + b) & 0xFFFFFFFFFFFFFFFF) % p
                if perm < min_val:
                    min_val = perm
            hashes.append(min_val)

        # Fold into compact string using deterministic hashing
        band_size = 4
        bands = []
        for i in range(0, num_perm, band_size):
            band = tuple(hashes[i : i + band_size])
            band_hash = hashlib.md5(str(band).encode()).hexdigest()[:4]
            bands.append(band_hash)
        url_h = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"mh-{url_h}-{'-'.join(bands)}"

    # -----------------------------
    # Loop Detection & Novelty
    # -----------------------------

    def update(self, state_hash: str) -> Tuple[float, float, Dict[str, Any]]:
        """
        Update memory with new state hash.
        Returns (r_loop, r_novelty, info).
        """
        self._state_history.append(state_hash)
        self._visit_count[state_hash] = self._visit_count.get(state_hash, 0) + 1

        # Loop detection: check if state_hash appeared in recent window
        recent = list(self._state_history)[-self.loop_window :]
        loop_count = recent.count(state_hash) - 1  # exclude current
        r_loop = 0.0
        if loop_count > 0:
            self._loop_counter[state_hash] = self._loop_counter.get(state_hash, 0) + 1
            r_loop = self.loop_penalty_base * self._loop_counter[state_hash]

        # Novelty detection
        is_novel = state_hash not in self._visited_hashes
        r_novelty = 0.0
        if is_novel:
            self._visited_hashes[state_hash] = None
            # Enforce max_states limit with true LRU eviction
            if len(self._visited_hashes) > self.max_states:
                self._visited_hashes.popitem(last=False)  # evict oldest
            decay = 1.0 - min(1.0, len(self._visited_hashes) / self.max_states)
            r_novelty = self.novelty_bonus_base * decay

        info = {
            "is_novel": is_novel,
            "visit_count": self._visit_count[state_hash],
            "loop_count": loop_count,
            "total_visited": len(self._visited_hashes),
        }
        return r_loop, r_novelty, info

    def is_looping(self, state_hash: str, threshold: int = 2) -> bool:
        """Check if currently in a detected loop."""
        recent = list(self._state_history)[-self.loop_window * 2 :]
        return recent.count(state_hash) >= threshold

    def reset(self) -> None:
        """Clear history for a new episode (keep visited set for novelty)."""
        self._state_history.clear()
        self._loop_counter.clear()

    def full_reset(self) -> None:
        """Complete reset including visited set."""
        self.reset()
        self._visited_hashes.clear()
        self._visit_count.clear()

    @property
    def visited_count(self) -> int:
        return len(self._visited_hashes)
