from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


def _sha1(text: str) -> str:
    return hashlib.sha1(
        (text or "").encode("utf-8"), usedforsecurity=False
    ).hexdigest()


@dataclass(frozen=True)
class RelationSpec:
    pid: str
    name: str
    label: str
    aliases: List[str]
    keywords: Set[str]
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    schema_source: str = "wikidata"

    def keyword_set(self) -> Set[str]:
        kws: Set[str] = set()
        for a in self.aliases:
            for tok in a.lower().split():
                if tok:
                    kws.add(tok)
        kws.update({k.lower() for k in self.keywords})
        return kws

    def default_predicate(self) -> str:
        if self.aliases:
            return self.aliases[0]
        if self.keywords:
            return sorted(self.keywords)[0]
        return self.label or ""


class RelationRegistry:
    def __init__(self, version: str, specs: Dict[str, RelationSpec]):
        self.version = version
        self._specs_by_pid = specs
        self._specs_by_name = {v.name.upper(): v for v in specs.values()}
        self._alias_map: Dict[str, RelationSpec] = {}
        for spec in specs.values():
            for alias in spec.aliases:
                key = alias.upper()
                if key not in self._alias_map:
                    self._alias_map[key] = spec
            if spec.label:
                key = spec.label.upper()
                if key not in self._alias_map:
                    self._alias_map[key] = spec

    @classmethod
    def from_json(cls, path: Path) -> "RelationRegistry":
        data = json.loads(path.read_text())
        version = data.get("version", "unknown")
        allowlist = data.get("allowlist", {})
        specs: Dict[str, RelationSpec] = {}
        for pid, raw in allowlist.items():
            specs[pid] = RelationSpec(
                pid=pid,
                name=str(raw.get("name") or pid).upper(),
                label=str(raw.get("label") or "").strip(),
                aliases=[a.strip() for a in raw.get("aliases", []) if a],
                keywords=set(str(k).strip() for k in raw.get("keywords", []) if k),
                subject_type=raw.get("subject_type"),
                object_type=raw.get("object_type"),
                schema_source=str(raw.get("schema_source", "wikidata")),
            )
        return cls(version=version, specs=specs)

    def lookup(self, key: Optional[str]) -> Optional[RelationSpec]:
        if not key:
            return None
        key_u = str(key).upper()
        if key_u in self._specs_by_pid:
            return self._specs_by_pid[key_u]
        if key_u in self._specs_by_name:
            return self._specs_by_name[key_u]
        return self._alias_map.get(key_u)

    def keywords_for(self, key: Optional[str]) -> Set[str]:
        spec = self.lookup(key)
        if not spec:
            return set()
        return spec.keyword_set()

    def keyword_hash(self) -> str:
        entries = []
        for pid, spec in sorted(self._specs_by_pid.items()):
            kws = sorted(spec.keyword_set())
            entries.append(f"{pid}:{','.join(kws)}")
        return _sha1("|".join(entries))

    def default_predicate(self, key: Optional[str]) -> str:
        spec = self.lookup(key)
        return spec.default_predicate() if spec else ""

    def get(self, key: Optional[str], default=None) -> Optional[RelationSpec]:
        """Dictionary-style accessor used by some callers."""

        spec = self.lookup(key)
        return spec if spec is not None else default

    def match_from_name(self, name: Optional[str]) -> Optional[RelationSpec]:
        """Best-effort match using pid, canonical name, label, or aliases."""

        if not name:
            return None

        # Normalize common variations (e.g., json schema enums)
        key = str(name).replace("_", " ").strip()
        spec = self.lookup(key)
        if spec:
            return spec

        key_lower = key.lower()
        for candidate in self._specs_by_pid.values():
            if key_lower == candidate.label.lower():
                return candidate
            if key_lower in {alias.lower() for alias in candidate.aliases}:
                return candidate

        return None

    def all_names(self) -> List[str]:
        return sorted({spec.name for spec in self._specs_by_pid.values()})

    def specs(self) -> List[RelationSpec]:
        return list(self._specs_by_pid.values())


_DEFAULT_PATH = Path(__file__).with_name("relation_schema_wd.json")
_default_registry: Optional[RelationRegistry] = None


def get_default_registry() -> RelationRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = RelationRegistry.from_json(_DEFAULT_PATH)
    return _default_registry
