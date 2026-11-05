"""Verifier-driven Query Compiler (VQC) for targeted query rewriting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

from .facets import Facet, FacetType
from .candidates import Passage


@dataclass
class QueryRewrite:
    """A rewritten query for targeted retrieval."""
    original_query: str
    rewritten_query: str
    rewrite_type: str  # entity_alias, temporal_qualifier, bridge_hint, etc.
    target_facets: List[str]
    metadata: Dict[str, Any]


class VerifierQueryCompiler:
    """
    VQC generates typed query rewrites based on verifier feedback.

    From the TRIDENT spec: generates entity aliases, temporal qualifiers,
    bridge hints, and predicate inversions/expansions to improve coverage.
    """

    def __init__(self, config: Any, nli_scorer: Any):
        self.config = config
        self.nli_scorer = nli_scorer
        self.rewrite_history = []

        # Entity alias patterns
        self.entity_aliases = {
            'United States': ['USA', 'US', 'America', 'United States of America'],
            'United Kingdom': ['UK', 'Britain', 'Great Britain', 'England'],
            'World War II': ['WWII', 'WW2', 'Second World War', 'World War 2'],
            'World War I': ['WWI', 'WW1', 'First World War', 'World War 1'],
        }

        # Temporal patterns
        self.temporal_patterns = {
            'year': r'\b(19\d{2}|20\d{2})\b',
            'decade': r'\b(19\d0s|20\d0s)\b',
            'century': r'\b(\d{1,2}(?:st|nd|rd|th)\s+century)\b'
        }

        # Relation inversions
        self.relation_inversions = {
            'wrote': 'written by',
            'created': 'created by',
            'invented': 'invented by',
            'discovered': 'discovered by',
            'directed': 'directed by',
            'composed': 'composed by',
            'founded': 'founded by',
            'parent': 'child',
            'teacher': 'student',
            'employer': 'employee'
        }

    def generate_rewrites(
        self,
        query: str,
        uncovered_facets: List[Facet],
        current_passages: List[Passage],
        max_rewrites: int = 5
    ) -> List[str]:
        """
        Generate query rewrites to target uncovered facets.

        Analyzes deficit facets and current passages to generate
        focused rewrites that might retrieve better evidence.
        """
        rewrites = []

        # Analyze uncovered facets
        facet_analysis = self._analyze_uncovered_facets(uncovered_facets)

        # Generate rewrites by type
        if facet_analysis['missing_entities']:
            rewrites.extend(
                self._generate_entity_rewrites(
                    query, facet_analysis['missing_entities']
                )
            )

        if facet_analysis['missing_relations']:
            rewrites.extend(
                self._generate_relation_rewrites(
                    query, facet_analysis['missing_relations']
                )
            )

        if facet_analysis['missing_temporal']:
            rewrites.extend(
                self._generate_temporal_rewrites(
                    query, facet_analysis['missing_temporal']
                )
            )

        if facet_analysis['missing_bridges']:
            rewrites.extend(
                self._generate_bridge_rewrites(
                    query,
                    facet_analysis['missing_bridges'],
                    current_passages
                )
            )

        # Limit number of rewrites
        rewrites = rewrites[:max_rewrites]

        # Store history
        self.rewrite_history.extend(rewrites)

        return rewrites

    def _analyze_uncovered_facets(
        self,
        uncovered_facets: List[Facet]
    ) -> Dict[str, List[Facet]]:
        """Categorize uncovered facets by type."""
        analysis = {
            'missing_entities': [],
            'missing_relations': [],
            'missing_temporal': [],
            'missing_bridges': [],
            'missing_numeric': []
        }

        for facet in uncovered_facets:
            if facet.facet_type == FacetType.ENTITY:
                analysis['missing_entities'].append(facet)
            elif facet.facet_type == FacetType.RELATION:
                analysis['missing_relations'].append(facet)
            elif facet.facet_type == FacetType.TEMPORAL:
                analysis['missing_temporal'].append(facet)
            elif facet.facet_type == FacetType.BRIDGE:
                analysis['missing_bridges'].append(facet)
            elif facet.facet_type == FacetType.NUMERIC:
                analysis['missing_numeric'].append(facet)

        return analysis

    def _generate_entity_rewrites(
        self,
        query: str,
        missing_entities: List[Facet]
    ) -> List[str]:
        """Generate entity alias rewrites."""
        rewrites = []

        for facet in missing_entities:
            entity = facet.template.get('mention', '')

            # Try aliases
            if entity in self.entity_aliases:
                for alias in self.entity_aliases[entity]:
                    rewrite = query.replace(entity, alias)
                    if rewrite != query:
                        rewrites.append(rewrite)

            # Try adding context
            context = facet.template.get('context', '')
            if context and entity:
                # Add qualifying context
                rewrites.append(f"{query} {entity} {context}")
                rewrites.append(f"{entity} {context} {query}")

            # Try more specific query
            if entity:
                rewrites.append(f'"{entity}" AND ({query})')

        return rewrites

    def _generate_relation_rewrites(
        self,
        query: str,
        missing_relations: List[Facet]
    ) -> List[str]:
        """Generate relation-focused rewrites."""
        rewrites = []

        for facet in missing_relations:
            subject = facet.template.get('subject', '')
            predicate = facet.template.get('predicate', '')
            obj = facet.template.get('object', '')

            # Try relation inversion
            if predicate in self.relation_inversions:
                inverted = self.relation_inversions[predicate]
                rewrites.append(f"{obj} {inverted} {subject}")

            # Try explicit relation query
            if subject and obj:
                rewrites.append(f"{subject} {predicate} {obj}")
                rewrites.append(f'relationship between {subject} and {obj}')
                rewrites.append(f'{subject} connected to {obj}')

            # Try focusing on one entity at a time
            if subject:
                rewrites.append(f"{subject} {predicate}")
            if obj:
                rewrites.append(f"{predicate} {obj}")

        return rewrites

    def _generate_temporal_rewrites(
        self,
        query: str,
        missing_temporal: List[Facet]
    ) -> List[str]:
        """Generate temporal-focused rewrites."""
        rewrites = []

        for facet in missing_temporal:
            time = facet.template.get('time', '')
            event = facet.template.get('event', '')

            if time:
                # Add temporal qualifier
                rewrites.append(f"{query} in {time}")
                rewrites.append(f"{query} during {time}")
                rewrites.append(f"{time} {query}")

                # Try specific temporal queries
                if event:
                    rewrites.append(f"{event} {time}")
                    rewrites.append(f"when did {event}")
                    rewrites.append(f"{event} date year time")

        return rewrites

    def _generate_bridge_rewrites(
        self,
        query: str,
        missing_bridges: List[Facet],
        current_passages: List[Passage]
    ) -> List[str]:
        """Generate bridge hint rewrites for multi-hop reasoning."""
        rewrites = []

        # Extract entities from current passages
        passage_entities = self._extract_entities_from_passages(current_passages)

        for facet in missing_bridges:
            entity1 = facet.template.get('entity1', '')
            entity2 = facet.template.get('entity2', '')
            relation = facet.template.get('relation', '')

            # Try to find intermediate entities
            if entity1 and entity2:
                # Direct bridge query
                rewrites.append(f"{entity1} AND {entity2}")
                rewrites.append(f"connection between {entity1} and {entity2}")

                # Look for potential bridge entities in passages
                for bridge_entity in passage_entities:
                    if bridge_entity not in [entity1, entity2]:
                        rewrites.append(f"{entity1} {bridge_entity} {entity2}")
                        rewrites.append(f"{bridge_entity} related to {entity1}")

                # Try decomposed queries
                rewrites.append(f"{entity1} {relation}")
                rewrites.append(f"{relation} {entity2}")

        return rewrites

    def _extract_entities_from_passages(
        self,
        passages: List[Passage],
        max_entities: int = 10
    ) -> Set[str]:
        """Extract potential entities from passages."""
        entities = set()

        for passage in passages[:5]:  # Look at top passages
            # Simple capitalized word extraction
            words = passage.text.split()
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2:
                    # Check if it's part of a multi-word entity
                    entity = word
                    j = i + 1
                    while j < len(words) and words[j][0].isupper():
                        entity += " " + words[j]
                        j += 1

                    entities.add(entity)
                    if len(entities) >= max_entities:
                        return entities

        return entities

    def generate_procedural_rewrites(
        self,
        query: str,
        steps: List[str]
    ) -> List[str]:
        """Generate rewrites for procedural queries."""
        rewrites = []

        # Break down into step-wise queries
        for i, step in enumerate(steps):
            rewrites.append(f"step {i+1}: {step}")
            rewrites.append(f"{query} specifically {step}")

        # Try different phrasings
        rewrites.append(f"how to {query}")
        rewrites.append(f"process of {query}")
        rewrites.append(f"{query} procedure steps")

        return rewrites

    def generate_comparison_rewrites(
        self,
        query: str,
        entity1: str,
        entity2: str,
        attribute: str
    ) -> List[str]:
        """Generate rewrites for comparison queries."""
        rewrites = []

        # Individual entity queries
        rewrites.append(f"{entity1} {attribute}")
        rewrites.append(f"{entity2} {attribute}")

        # Comparison queries
        rewrites.append(f"compare {entity1} and {entity2} {attribute}")
        rewrites.append(f"{entity1} versus {entity2} {attribute}")
        rewrites.append(f"difference between {entity1} and {entity2} {attribute}")

        # Statistical queries
        rewrites.append(f"{attribute} statistics {entity1} {entity2}")

        return rewrites

    def get_rewrite_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated rewrites."""
        if not self.rewrite_history:
            return {'total_rewrites': 0}

        rewrite_types = {}
        for rewrite in self.rewrite_history:
            rtype = self._classify_rewrite_type(rewrite)
            rewrite_types[rtype] = rewrite_types.get(rtype, 0) + 1

        return {
            'total_rewrites': len(self.rewrite_history),
            'rewrite_types': rewrite_types,
            'unique_rewrites': len(set(self.rewrite_history))
        }

    def _classify_rewrite_type(self, rewrite: str) -> str:
        """Classify the type of rewrite."""
        if ' AND ' in rewrite or ' OR ' in rewrite:
            return 'boolean'
        elif any(year in rewrite for year in ['19', '20']):
            return 'temporal'
        elif 'between' in rewrite.lower():
            return 'relation'
        elif '"' in rewrite:
            return 'exact_match'
        else:
            return 'general'