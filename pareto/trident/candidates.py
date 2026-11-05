"""Passage candidate representation and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Any


@dataclass
class Passage:
    """Candidate passage/snippet for retrieval."""

    pid: str  # Passage ID
    text: str  # Passage text content
    cost: int  # Token cost (prompt tokens)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Make passage hashable for set operations."""
        return hash(self.pid)

    def __eq__(self, other: Any) -> bool:
        """Equality based on passage ID."""
        if not isinstance(other, Passage):
            return False
        return self.pid == other.pid

    def short_text(self, limit: int = 120) -> str:
        """Return truncated text for logging."""
        if len(self.text) <= limit:
            return self.text
        return self.text[:limit - 3] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'pid': self.pid,
            'text': self.text,
            'cost': self.cost,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Passage":
        """Create from dictionary."""
        return cls(
            pid=data['pid'],
            text=data['text'],
            cost=data['cost'],
            metadata=data.get('metadata', {})
        )


def top_k_by_score(
    scores: Dict[str, float],
    passages: Dict[str, Passage],
    k: int
) -> List[Passage]:
    """Return top-k passages according to scores."""
    ordered_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [passages[pid] for pid in ordered_ids if pid in passages]


def group_passages_by_field(
    passages: Iterable[Passage],
    field_name: str
) -> Dict[Any, List[Passage]]:
    """Group passages by a metadata field."""
    groups: Dict[Any, List[Passage]] = {}
    for passage in passages:
        key = passage.metadata.get(field_name)
        groups.setdefault(key, []).append(passage)
    return groups


def filter_passages_by_cost(
    passages: Iterable[Passage],
    max_cost: int
) -> List[Passage]:
    """Filter passages that fit within cost budget."""
    return [p for p in passages if p.cost <= max_cost]


def compute_total_cost(passages: Iterable[Passage]) -> int:
    """Compute total token cost of passages."""
    return sum(p.cost for p in passages)


def deduplicate_passages(passages: Iterable[Passage]) -> List[Passage]:
    """Remove duplicate passages while preserving order."""
    seen = set()
    unique = []
    for passage in passages:
        if passage.pid not in seen:
            seen.add(passage.pid)
            unique.append(passage)
    return unique


def merge_passage_lists(
    *passage_lists: List[Passage],
    strategy: str = "union"
) -> List[Passage]:
    """
    Merge multiple passage lists.

    Strategies:
    - union: Include all unique passages
    - intersection: Only passages in all lists
    - first: Prioritize earlier lists
    """
    if not passage_lists:
        return []

    if strategy == "union":
        all_passages = []
        for plist in passage_lists:
            all_passages.extend(plist)
        return deduplicate_passages(all_passages)

    elif strategy == "intersection":
        if len(passage_lists) == 1:
            return passage_lists[0]

        common_pids = set(p.pid for p in passage_lists[0])
        for plist in passage_lists[1:]:
            common_pids &= set(p.pid for p in plist)

        # Return passages in order of first list
        return [p for p in passage_lists[0] if p.pid in common_pids]

    elif strategy == "first":
        return deduplicate_passages(passage_lists[0]) if passage_lists else []

    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def rank_passages_by_coverage(
    passages: List[Passage],
    covered_facets: Dict[str, List[str]]
) -> List[Tuple[Passage, int]]:
    """Rank passages by number of facets they cover."""
    rankings = []
    for passage in passages:
        coverage = len(covered_facets.get(passage.pid, []))
        rankings.append((passage, coverage))

    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


@dataclass
class PassagePool:
    """Pool of candidate passages with efficient operations."""

    passages: Dict[str, Passage] = field(default_factory=dict)

    def add(self, passage: Passage) -> None:
        """Add passage to pool."""
        self.passages[passage.pid] = passage

    def add_many(self, passages: Iterable[Passage]) -> None:
        """Add multiple passages."""
        for passage in passages:
            self.add(passage)

    def get(self, pid: str) -> Optional[Passage]:
        """Get passage by ID."""
        return self.passages.get(pid)

    def remove(self, pid: str) -> Optional[Passage]:
        """Remove and return passage."""
        return self.passages.pop(pid, None)

    def filter_by_cost(self, max_cost: int) -> List[Passage]:
        """Get passages within cost budget."""
        return [p for p in self.passages.values() if p.cost <= max_cost]

    def filter_by_metadata(
        self,
        field: str,
        value: Any
    ) -> List[Passage]:
        """Filter passages by metadata field."""
        return [
            p for p in self.passages.values()
            if p.metadata.get(field) == value
        ]

    def sample(self, n: int, seed: Optional[int] = None) -> List[Passage]:
        """Sample n passages randomly."""
        import random
        if seed is not None:
            random.seed(seed)

        all_passages = list(self.passages.values())
        if n >= len(all_passages):
            return all_passages

        return random.sample(all_passages, n)

    def __len__(self) -> int:
        """Number of passages in pool."""
        return len(self.passages)

    def __contains__(self, pid: str) -> bool:
        """Check if passage ID is in pool."""
        return pid in self.passages

    def __iter__(self):
        """Iterate over passages."""
        return iter(self.passages.values())
