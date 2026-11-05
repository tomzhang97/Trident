"""Enhanced facet representation and mining for TRIDENT."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import spacy
from transformers import pipeline


class FacetType(Enum):
    """Types of reasoning facets.

    Per TRIDENT framework Section 2.1:
    - ENTITY: Entity mention with type
    - RELATION: Relation assertion between entities
    - TEMPORAL: Temporal expression with time normalization
    - NUMERIC: Numeric value with normalization
    - BRIDGE_HOP1: First hop of multi-hop bridge (e1, r1, e_bridge)
    - BRIDGE_HOP2: Second hop of multi-hop bridge (e_bridge, r2, e2)
    - COMPARISON: Comparative statement (retained for compatibility)
    - CAUSAL: Causal relationship (retained for compatibility)
    - PROCEDURAL: Procedural steps (retained for compatibility)
    """
    ENTITY = "ENTITY"
    RELATION = "RELATION"
    TEMPORAL = "TEMPORAL"
    NUMERIC = "NUMERIC"
    BRIDGE_HOP1 = "BRIDGE_HOP1"  # First atomic hop: (e1, r1, e_bridge)
    BRIDGE_HOP2 = "BRIDGE_HOP2"  # Second atomic hop: (e_bridge, r2, e2)
    # Legacy types retained for backward compatibility
    BRIDGE = "BRIDGE"  # Deprecated: use BRIDGE_HOP1 and BRIDGE_HOP2
    COMPARISON = "COMPARISON"
    CAUSAL = "CAUSAL"
    PROCEDURAL = "PROCEDURAL"

    @classmethod
    def core_types(cls) -> list:
        """Return the six core facet types from the framework specification."""
        return [cls.ENTITY, cls.RELATION, cls.TEMPORAL, cls.NUMERIC, cls.BRIDGE_HOP1, cls.BRIDGE_HOP2]


@dataclass(frozen=True)
class Facet:
    """Represents a reasoning requirement from a query."""

    facet_id: str
    facet_type: FacetType
    template: Dict[str, Any]
    weight: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

        facet_type_val = object.__getattribute__(self, 'facet_type')
        if isinstance(facet_type_val, str):
            object.__setattr__(self, 'facet_type', FacetType(facet_type_val))

    def to_hypothesis(self) -> str:
        """Convert facet to NLI hypothesis format."""
        if self.facet_type == FacetType.ENTITY:
            entity = self.template.get('mention', '')
            context = self.template.get('context', '')
            if context:
                return f"{entity} is mentioned in the context of {context}"
            return f"{entity} is mentioned"

        elif self.facet_type == FacetType.RELATION:
            subject = self.template.get('subject', '')
            predicate = self.template.get('predicate', '')
            object_ent = self.template.get('object', '')
            return f"{subject} {predicate} {object_ent}"

        elif self.facet_type == FacetType.TEMPORAL:
            event = self.template.get('event', '')
            time = self.template.get('time', '')
            return f"{event} occurred in {time}"

        elif self.facet_type == FacetType.NUMERIC:
            entity = self.template.get('entity', '')
            value = self.template.get('value', '')
            attribute = self.template.get('attribute', '')
            return f"{entity} has {attribute} of {value}"

        elif self.facet_type == FacetType.BRIDGE_HOP1:
            # First hop: (e1, r1, e_bridge)
            entity1 = self.template.get('entity1', '')
            relation = self.template.get('relation', '')
            bridge_entity = self.template.get('bridge_entity', '')
            return f"{entity1} {relation} {bridge_entity}"

        elif self.facet_type == FacetType.BRIDGE_HOP2:
            # Second hop: (e_bridge, r2, e2)
            bridge_entity = self.template.get('bridge_entity', '')
            relation = self.template.get('relation', '')
            entity2 = self.template.get('entity2', '')
            return f"{bridge_entity} {relation} {entity2}"

        elif self.facet_type == FacetType.BRIDGE:
            # Legacy bridge type (deprecated)
            entity1 = self.template.get('entity1', '')
            entity2 = self.template.get('entity2', '')
            relation = self.template.get('relation', '')
            return f"{entity1} is connected to {entity2} through {relation}"

        elif self.facet_type == FacetType.COMPARISON:
            entity1 = self.template.get('entity1', '')
            entity2 = self.template.get('entity2', '')
            attribute = self.template.get('attribute', '')
            return f"{entity1} and {entity2} are compared by {attribute}"

        elif self.facet_type == FacetType.CAUSAL:
            cause = self.template.get('cause', '')
            effect = self.template.get('effect', '')
            return f"{cause} causes {effect}"

        elif self.facet_type == FacetType.PROCEDURAL:
            steps = self.template.get('steps', [])
            return f"The process involves: {', '.join(steps)}"

        else:
            return str(self.template)

    def get_keywords(self) -> List[str]:
        """Extract keywords from facet for retrieval."""
        keywords = []

        for value in self.template.values():
            if isinstance(value, str):
                # Extract words, ignoring common stopwords
                words = value.split()
                keywords.extend([w for w in words if len(w) > 2])
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        words = item.split()
                        keywords.extend([w for w in words if len(w) > 2])

        return list(set(keywords))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'facet_id': self.facet_id,
            'facet_type': self.facet_type.value,
            'template': self.template,
            'weight': self.weight,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Facet":
        """Create from dictionary."""
        return cls(
            facet_id=data['facet_id'],
            facet_type=FacetType(data['facet_type']),
            template=data['template'],
            weight=data.get('weight', 1.0),
            metadata=data.get('metadata')
        )


class FacetMiner:
    """Extract facets from queries using NLP techniques."""

    def __init__(self, config: Any):
        self.config = config

        # Load spaCy model for entity and relation extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback if spaCy model not installed
            self.nlp = None

        # Question decomposition model (optional)
        self.decomposer = None
        if hasattr(config, 'use_decomposer') and config.use_decomposer:
            try:
                self.decomposer = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base"
                )
            except:
                pass

    def extract_facets(
        self,
        query: str,
        supporting_facts: Optional[List[Tuple[str, int]]] = None
    ) -> List[Facet]:
        """Extract facets from query."""
        facets = []

        # Extract different types of facets
        facets.extend(self._extract_entity_facets(query))
        facets.extend(self._extract_relation_facets(query))
        facets.extend(self._extract_temporal_facets(query))
        facets.extend(self._extract_numeric_facets(query))
        facets.extend(self._extract_comparison_facets(query))

        # For multi-hop questions, extract bridge facets
        if self._is_multi_hop(query):
            facets.extend(self._extract_bridge_facets(query, supporting_facts))

        # Deduplicate and assign IDs
        unique_facets = self._deduplicate_facets(facets)

        return unique_facets

    def _extract_entity_facets(self, query: str) -> List[Facet]:
        """Extract entity-based facets."""
        facets = []

        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                facet = Facet(
                    facet_id=f"entity_{len(facets)}",
                    facet_type=FacetType.ENTITY,
                    template={
                        'mention': ent.text,
                        'label': ent.label_,
                        'context': self._get_entity_context(ent, doc)
                    }
                )
                facets.append(facet)
        else:
            # Simple regex-based extraction as fallback
            # Extract capitalized words as potential entities
            pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            entities = re.findall(pattern, query)
            for i, entity in enumerate(entities):
                facet = Facet(
                    facet_id=f"entity_{i}",
                    facet_type=FacetType.ENTITY,
                    template={'mention': entity}
                )
                facets.append(facet)

        return facets

    def _extract_relation_facets(self, query: str) -> List[Facet]:
        """Extract relation-based facets."""
        facets = []

        # Common relation patterns
        relation_patterns = [
            (r'(\w+)\s+(?:is|was|are|were)\s+(\w+)\s+(?:of|by|in)\s+(\w+)', 'is_related_to'),
            (r'(\w+)\s+(?:wrote|created|invented|discovered)\s+(\w+)', 'created'),
            (r'(\w+)\s+(?:born|died|lived)\s+(?:in|at)\s+(\w+)', 'temporal_relation'),
            (r'(\w+)\s+(?:located|situated)\s+(?:in|at)\s+(\w+)', 'location')
        ]

        for pattern, rel_type in relation_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    facet = Facet(
                        facet_id=f"relation_{len(facets)}",
                        facet_type=FacetType.RELATION,
                        template={
                            'subject': groups[0],
                            'predicate': rel_type,
                            'object': groups[-1] if len(groups) > 2 else groups[1]
                        }
                    )
                    facets.append(facet)

        return facets

    def _extract_temporal_facets(self, query: str) -> List[Facet]:
        """Extract temporal facets."""
        facets = []

        # Temporal patterns
        temporal_patterns = [
            r'\b(19\d{2}|20\d{2})\b',  # Years
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(before|after|during|since|until)\s+(\d{4}|\w+)\b'
        ]

        for pattern in temporal_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                facet = Facet(
                    facet_id=f"temporal_{len(facets)}",
                    facet_type=FacetType.TEMPORAL,
                    template={
                        'time': match.group(),
                        'event': self._extract_temporal_event(query, match.span())
                    }
                )
                facets.append(facet)

        return facets

    def _extract_numeric_facets(self, query: str) -> List[Facet]:
        """Extract numeric facets."""
        facets = []

        # Numeric patterns
        numeric_pattern = r'\b(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand|hundred)?\b'
        matches = re.finditer(numeric_pattern, query, re.IGNORECASE)

        for match in matches:
            value = match.group(1)
            unit = match.group(2) if match.group(2) else ''

            # Get context around the number
            start, end = match.span()
            context_start = max(0, start - 20)
            context_end = min(len(query), end + 20)
            context = query[context_start:context_end]

            facet = Facet(
                facet_id=f"numeric_{len(facets)}",
                facet_type=FacetType.NUMERIC,
                template={
                    'value': value,
                    'unit': unit,
                    'context': context
                }
            )
            facets.append(facet)

        return facets

    def _extract_comparison_facets(self, query: str) -> List[Facet]:
        """Extract comparison facets."""
        facets = []

        comparison_keywords = ['more', 'less', 'better', 'worse', 'higher', 'lower', 'bigger', 'smaller']

        for keyword in comparison_keywords:
            if keyword in query.lower():
                # Extract entities being compared
                pattern = rf'(\w+)\s+(?:is|was|are|were)?\s*{keyword}\s+than\s+(\w+)'
                matches = re.finditer(pattern, query, re.IGNORECASE)

                for match in matches:
                    facet = Facet(
                        facet_id=f"comparison_{len(facets)}",
                        facet_type=FacetType.COMPARISON,
                        template={
                            'entity1': match.group(1),
                            'entity2': match.group(2),
                            'attribute': keyword
                        }
                    )
                    facets.append(facet)

        return facets

    def _extract_bridge_facets(
        self,
        query: str,
        supporting_facts: Optional[List[Tuple[str, int]]] = None
    ) -> List[Facet]:
        """
        Extract bridge facets for multi-hop reasoning.

        Per TRIDENT Framework Section 2.1 (Bridge decomposition):
        Multi-hop requirements are decomposed into two atomic hops:
        - BRIDGE_HOP1(e1, r1, e_bridge): First hop from entity1 to bridge entity
        - BRIDGE_HOP2(e_bridge, r2, e2): Second hop from bridge entity to entity2

        Each hop is a mandatory facet in Safe-Cover, restoring classical
        set-cover semantics (one passage may cover one or more atomic facets).
        """
        facets = []

        # Look for patterns that indicate multi-hop reasoning
        if supporting_facts:
            # Use supporting facts to identify bridge entities
            entities = set()
            for title, _ in supporting_facts:
                entities.add(title)

            if len(entities) >= 2:
                entities_list = list(entities)
                hop_counter = 0

                # For each adjacent pair, create BRIDGE_HOP1 and BRIDGE_HOP2
                for i in range(len(entities_list) - 1):
                    e1 = entities_list[i]
                    e_bridge = entities_list[i + 1] if i + 1 < len(entities_list) else entities_list[i]
                    e2 = entities_list[i + 2] if i + 2 < len(entities_list) else e_bridge

                    # Infer relations from query if possible
                    r1 = self._infer_relation(query, e1, e_bridge)
                    r2 = self._infer_relation(query, e_bridge, e2)

                    # BRIDGE_HOP1: (e1, r1, e_bridge)
                    facet_hop1 = Facet(
                        facet_id=f"bridge_hop1_{hop_counter}",
                        facet_type=FacetType.BRIDGE_HOP1,
                        template={
                            'entity1': e1,
                            'relation': r1,
                            'bridge_entity': e_bridge,
                        },
                        metadata={'hop_index': 1, 'bridge_chain': hop_counter}
                    )
                    facets.append(facet_hop1)

                    # BRIDGE_HOP2: (e_bridge, r2, e2) - only if we have a third entity
                    if i + 2 < len(entities_list) or e_bridge != e2:
                        facet_hop2 = Facet(
                            facet_id=f"bridge_hop2_{hop_counter}",
                            facet_type=FacetType.BRIDGE_HOP2,
                            template={
                                'bridge_entity': e_bridge,
                                'relation': r2,
                                'entity2': e2,
                            },
                            metadata={'hop_index': 2, 'bridge_chain': hop_counter}
                        )
                        facets.append(facet_hop2)

                    hop_counter += 1

        return facets

    def _infer_relation(self, query: str, entity1: str, entity2: str) -> str:
        """Infer relation between entities from query context."""
        # Common relation patterns to look for
        relation_indicators = [
            ('wrote', 'authored'),
            ('created', 'created'),
            ('directed', 'directed'),
            ('born in', 'birthplace'),
            ('died in', 'death_place'),
            ('located in', 'location'),
            ('capital of', 'capital'),
            ('member of', 'membership'),
            ('part of', 'part_of'),
            ('played for', 'team'),
            ('starred in', 'appeared_in'),
        ]

        query_lower = query.lower()
        for pattern, relation in relation_indicators:
            if pattern in query_lower:
                return relation

        # Default relation if none found
        return "related_to"

    def _is_multi_hop(self, query: str) -> bool:
        """Check if query requires multi-hop reasoning."""
        multi_hop_indicators = [
            'and', 'both', 'which', 'that', 'who',
            'when did', 'where did', 'how many'
        ]

        query_lower = query.lower()
        indicator_count = sum(1 for ind in multi_hop_indicators if ind in query_lower)

        # Heuristic: multiple indicators suggest multi-hop
        return indicator_count >= 2 or 'which' in query_lower

    def _get_entity_context(self, entity: Any, doc: Any) -> str:
        """Get context around an entity."""
        if not doc:
            return ""

        # Get surrounding tokens
        start = max(0, entity.start - 3)
        end = min(len(doc), entity.end + 3)
        context_tokens = doc[start:end]

        return " ".join([token.text for token in context_tokens])

    def _extract_temporal_event(self, query: str, time_span: Tuple[int, int]) -> str:
        """Extract event associated with temporal expression."""
        # Simple heuristic: get surrounding words
        start = max(0, time_span[0] - 30)
        end = min(len(query), time_span[1] + 30)
        context = query[start:end]

        # Remove the time expression itself
        time_text = query[time_span[0]:time_span[1]]
        event = context.replace(time_text, "").strip()

        return event

    def _deduplicate_facets(self, facets: List[Facet]) -> List[Facet]:
        """Remove duplicate facets."""
        seen = set()
        unique = []

        for facet in facets:
            signature = f"{facet.facet_type}:{str(facet.template)}"
            if signature not in seen:
                seen.add(signature)
                unique.append(facet)

        return unique

    def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-questions."""
        if self.decomposer:
            prompt = f"Break down this question into simpler sub-questions: {query}"
            result = self.decomposer(prompt, max_length=200)
            sub_questions = result[0]['generated_text'].split('\n')
            return [q.strip() for q in sub_questions if q.strip()]
        else:
            # Simple heuristic decomposition
            return [query]  # Return original query if no decomposer