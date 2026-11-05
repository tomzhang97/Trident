"""Multi-hop chain builder for extracting answers from certified passages.
This module builds explicit reasoning chains from certified evidence,
ensuring that the answer is grounded in the actual passages selected.
"""

from __future__ import annotations

import json
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

# Entity extraction regex - Title Case spans
ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b")

# =============================================================================
# CERTIFIED-ONLY CONSTANTS
# =============================================================================
# Deny-listed document titles that should never be used for answer extraction
DENY_TITLES = frozenset([
    "disambiguation", "list of", "index of", "outline of", "portal:",
    "category:", "template:", "wikipedia:", "help:", "file:", "mediawiki:"
])

# Month names for date extraction
MONTHS = frozenset([
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
])


# =============================================================================
# CERTIFIED-ONLY HELPER FUNCTIONS
# =============================================================================

def _cert_pid(certificates: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract certified passage IDs from certificates.

    Returns:
        Set of passage IDs that have at least one valid certificate.
    """
    return {c.get("passage_id", "") for c in certificates if c.get("passage_id")}


def _norm_ws(s: str) -> str:
    """
    Normalize whitespace in a string.

    Collapses multiple spaces/tabs/newlines into single spaces and strips.
    """
    return " ".join(s.split())


def _normalize_text_unicode(text: str) -> str:
    """
    Normalize text for substring matching across Unicode variants.

    Handles:
    - Unicode normalization (NFKC)
    - Curly quotes -> straight quotes
    - En/em dashes -> hyphens
    - Whitespace normalization
    """
    import unicodedata

    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Curly quotes -> straight
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")

    # Dashes -> hyphen
    text = text.replace("–", "-").replace("—", "-")

    # Whitespace normalization
    text = _norm_ws(text)

    return text


def _is_deny_title(title: str) -> bool:
    """Check if a title should be denied for answer extraction."""
    if not title:
        return False
    t_lower = title.lower()
    return any(deny in t_lower for deny in DENY_TITLES)


# Relation triggers for different relation kinds (used as fallback when LLM router unavailable)
REL_TRIGGERS = {
    "DIRECTOR": ["directed by", "director", "film directed by", "directed", "filmmaker"],
    "BORN": ["was born", "born in", "birthplace", "native of", "born on"],
    "AWARD": ["won", "award", "nominated", "prize", "received", "honored"],
    "CREATED": ["created", "founded", "wrote", "written by", "author", "composed"],
    "LOCATION": ["located in", "capital of", "situated", "based in", "headquarters"],
    "MARRIAGE": ["married", "spouse", "wife", "husband", "wed"],
    "MOTHER": ["mother", "mom", "son of", "daughter of", "child of"],
    "FATHER": ["father", "dad", "son of", "daughter of", "child of"],
    "PARENT": ["parent", "mother", "father", "son of", "daughter of", "child of"],
    "SPOUSE": ["married", "spouse", "wife", "husband", "wed", "partner"],
    "NATIONALITY": ["nationality", "citizen", "national of", "from"],
    "BIRTHPLACE": ["born in", "birthplace", "native of", "birth place"],
}

# All known relation types for LLM router
KNOWN_RELATION_TYPES = [
    "DIRECTOR", "PRODUCER", "CREATOR", "AUTHOR", "COMPOSER", "PERFORMER",
    "MOTHER", "FATHER", "PARENT", "CHILD", "SPOUSE", "SIBLING",
    "BIRTHPLACE", "BIRTHDATE", "NATIONALITY", "OCCUPATION",
    "AWARD", "LOCATION", "CAPITAL", "HEADQUARTERS",
    "OTHER"
]


# =============================================================================
# P2: DETERMINISTIC SOLVERS FOR TEMPORAL/COMPARISON QUESTIONS
# =============================================================================

def try_deterministic_solver(
    question: str,
    passages: List[Dict[str, Any]],
    debug: bool = False
) -> Optional[str]:
    """
    Try deterministic pattern matching for common question types.

    These are cheap, 100% faithful extractors for specific question patterns.
    Run BEFORE LLM cert to avoid unnecessary LLM calls.

    Supported patterns:
    - "Which film came out first" → extract years, compare
    - "Where was X born" → "born in LOCATION" regex
    - "When did X die" → death year extraction
    - "Which person died first" → compare death years

    Returns:
        Answer string if pattern matches and answer found, None otherwise
    """
    q_lower = question.lower()

    # Pattern 1: "Which ... first" (temporal comparison)
    if "which" in q_lower and ("first" in q_lower or "earlier" in q_lower):
        if "film" in q_lower or "movie" in q_lower:
            # Film release comparison
            answer = _solve_film_comparison(question, passages, debug)
            if answer:
                return answer
        elif "die" in q_lower or "death" in q_lower:
            # Death date comparison
            answer = _solve_death_comparison(question, passages, debug)
            if answer:
                return answer

    # Pattern 2: "Where was X born"
    if "where" in q_lower and "born" in q_lower:
        answer = _solve_birthplace(question, passages, debug)
        if answer:
            return answer

    # Pattern 3: "When did X die"
    if "when" in q_lower and ("die" in q_lower or "death" in q_lower):
        answer = _solve_death_date(question, passages, debug)
        if answer:
            return answer

    # Pattern 4: "Who is the X of Y" (explicit relation extraction)
    if "who" in q_lower and " is the " in q_lower and " of " in q_lower:
        answer = _solve_who_is_x_of_y(question, passages, debug)
        if answer:
            return answer

    # P1-4: Pattern 5: Family relations (grandfather, grandmother, parent)
    if "who" in q_lower and any(rel in q_lower for rel in ["grandfather", "grandmother", "mother", "father", "parent"]):
        answer = _solve_family_relation(question, passages, debug)
        if answer:
            return answer

    # P1-4: Pattern 6: YES/NO same country comparison
    if q_lower.startswith(("is ", "are ", "was ", "were ", "do ", "did ")) and "same country" in q_lower:
        answer = _solve_yesno_same_country(question, passages, debug)
        if answer:
            return answer

    # P1-5: Pattern 7: Director born earlier/later (2-hop comparison)
    if "director" in q_lower and ("born" in q_lower or "birth" in q_lower) and ("first" in q_lower or "earlier" in q_lower or "later" in q_lower):
        answer = _solve_director_birth_comparison(question, passages, debug)
        if answer:
            return answer

    return None


def _solve_film_comparison(question: str, passages: List[Dict[str, Any]], debug: bool) -> Optional[str]:
    """Extract film release years and return earlier film."""
    # Extract film names from question (between quotes or title case)
    films = _extract_film_names_from_question(question)
    if len(films) < 2:
        return None

    # Find years for each film in passages
    film_years = {}
    for passage in passages:
        text = passage.get("text", "")
        for film in films:
            if film.lower() in text.lower():
                # Look for year patterns near the film name
                year = _extract_year_near_text(text, film)
                if year and film not in film_years:
                    film_years[film] = year

    if len(film_years) >= 2:
        # Return the film with the earliest year
        earliest_film = min(film_years.items(), key=lambda x: x[1])
        if debug:
            print(f"[DETERMINISTIC] Film comparison: {film_years}")
            print(f"  Answer: '{earliest_film[0]}' ({earliest_film[1]})")
        return earliest_film[0]

    return None


def _solve_death_comparison(question: str, passages: List[Dict[str, Any]], debug: bool) -> Optional[str]:
    """Extract death dates and return person who died first."""
    # Extract person names from question
    persons = _extract_person_names_from_question(question)
    if len(persons) < 2:
        return None

    # Find death years for each person
    death_years = {}
    for passage in passages:
        text = passage.get("text", "")
        for person in persons:
            if person.lower() in text.lower():
                # Look for death year patterns
                year = _extract_death_year(text, person)
                if year and person not in death_years:
                    death_years[person] = year

    if len(death_years) >= 2:
        earliest_person = min(death_years.items(), key=lambda x: x[1])
        if debug:
            print(f"[DETERMINISTIC] Death comparison: {death_years}")
            print(f"  Answer: '{earliest_person[0]}' (died {earliest_person[1]})")
        return earliest_person[0]

    return None


def _solve_birthplace(question: str, passages: List[Dict[str, Any]], debug: bool) -> Optional[str]:
    """Extract birthplace from passages."""
    # Extract person name from question
    person = _extract_subject_from_question(question)
    if not person:
        return None

    # Search for "born in LOCATION" patterns
    for passage in passages:
        text = passage.get("text", "")
        if person.lower() in text.lower():
            # Pattern: "born in LOCATION"
            match = re.search(r'born in ([A-Z][^.,;\n]{2,40})', text)
            if match:
                location = match.group(1).strip()
                if debug:
                    print(f"[DETERMINISTIC] Birthplace for '{person}': '{location}'")
                return location

    return None


def _solve_death_date(question: str, passages: List[Dict[str, Any]], debug: bool) -> Optional[str]:
    """Extract death date from passages."""
    person = _extract_subject_from_question(question)
    if not person:
        return None

    for passage in passages:
        text = passage.get("text", "")
        if person.lower() in text.lower():
            # Look for death date patterns
            year = _extract_death_year(text, person)
            if year:
                if debug:
                    print(f"[DETERMINISTIC] Death year for '{person}': {year}")
                return str(year)

    return None


def _solve_who_is_x_of_y(question: str, passages: List[Dict[str, Any]], debug: bool) -> Optional[str]:
    """Extract answer to 'Who is the X of Y' questions."""
    # Parse question to extract relation and subject
    # Example: "Who is the mother of John?" → relation="mother", subject="John"
    q_lower = question.lower()

    # Extract relation type (between "is the" and "of")
    match = re.search(r'who\s+is\s+the\s+(\w+)\s+of\s+([^?]+)', q_lower)
    if not match:
        return None

    relation = match.group(1).strip()
    subject = match.group(2).strip()

    # Map common relations to extraction patterns
    relation_patterns = {
        "mother": [r"mother[^.]*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", r"son of [^,]*and ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"],
        "father": [r"father[^.]*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", r"son of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"],
        "director": [r"directed by ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", r"director ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"],
        "spouse": [r"married (?:to )?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", r"spouse ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"],
        "founder": [r"founded by ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", r"founder ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"],
    }

    patterns = relation_patterns.get(relation, [])
    if not patterns:
        return None

    # Search passages for the relation near the subject
    for passage in passages:
        text = passage.get("text", "")
        if subject.lower() not in text.lower():
            continue

        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return first match (most prominent)
                answer = matches[0].strip()
                if debug:
                    print(f"[DETERMINISTIC] '{relation}' of '{subject}': '{answer}'")
                return answer

    return None


def _solve_family_relation(question: str, passages: List[Dict[str, Any]], debug: bool) -> Optional[str]:
    """
    P1-4: Extract family relations (grandfather, grandmother, father, mother).

    Handles patterns like:
    - "Who is the maternal grandfather of X"
    - "Who is the father of X"
    - "Who is X's grandmother"
    """
    q_lower = question.lower()

    # Extract subject (person whose relative we're looking for)
    subject = _extract_subject_from_question(question)
    if not subject:
        return None

    # Determine relation type
    relation_type = None
    if "paternal grandfather" in q_lower or "father's father" in q_lower:
        relation_type = "paternal_grandfather"
    elif "maternal grandfather" in q_lower or "mother's father" in q_lower:
        relation_type = "maternal_grandfather"
    elif "grandfather" in q_lower:
        relation_type = "grandfather"
    elif "grandmother" in q_lower:
        relation_type = "grandmother"
    elif "father" in q_lower:
        relation_type = "father"
    elif "mother" in q_lower:
        relation_type = "mother"
    elif "parent" in q_lower:
        relation_type = "parent"
    else:
        return None

    # Search passages for family relation mentions
    for passage in passages:
        text = passage.get("text", "")
        if subject.lower() not in text.lower():
            continue

        # Try different patterns based on relation type
        if relation_type in ["father", "paternal_grandfather"]:
            # "son/daughter of FATHER_NAME"
            match = re.search(r'(?:son|daughter|child) of ([A-ZÀ-Ÿ][^\n.,;()]*)', text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[\s,\.\(\)]+$', '', name)
                if _looks_like_person(name):
                    if debug:
                        print(f"[DETERMINISTIC] {relation_type} of '{subject}': '{name}'")
                    return name

            # "father was NAME"
            match = re.search(r'father\s+(?:was|is)\s+([A-ZÀ-Ÿ][^\n.,;()]*)', text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[\s,\.\(\)]+$', '', name)
                if _looks_like_person(name):
                    if debug:
                        print(f"[DETERMINISTIC] {relation_type} of '{subject}': '{name}'")
                    return name

        if relation_type in ["mother", "maternal_grandfather"]:
            # "mother was NAME"
            match = re.search(r'mother\s+(?:was|is)\s+([A-ZÀ-Ÿ][^\n.,;()]*)', text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[\s,\.\(\)]+$', '', name)
                if _looks_like_person(name):
                    if debug:
                        print(f"[DETERMINISTIC] {relation_type} of '{subject}': '{name}'")
                    return name

            # "son/daughter of X and MOTHER_NAME"
            match = re.search(r'(?:son|daughter|child) of [^,]+ and ([A-ZÀ-Ÿ][^\n.,;()]*)', text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[\s,\.\(\)]+$', '', name)
                if _looks_like_person(name):
                    if debug:
                        print(f"[DETERMINISTIC] {relation_type} of '{subject}': '{name}'")
                    return name

        if relation_type in ["grandfather", "paternal_grandfather", "maternal_grandfather"]:
            # "grandfather was NAME"
            match = re.search(r'grandfather\s+(?:was|is)\s+([A-ZÀ-Ÿ][^\n.,;()]*)', text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[\s,\.\(\)]+$', '', name)
                if _looks_like_person(name):
                    if debug:
                        print(f"[DETERMINISTIC] {relation_type} of '{subject}': '{name}'")
                    return name

        if relation_type == "grandmother":
            # "grandmother was NAME"
            match = re.search(r'grandmother\s+(?:was|is)\s+([A-ZÀ-Ÿ][^\n.,;()]*)', text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[\s,\.\(\)]+$', '', name)
                if _looks_like_person(name):
                    if debug:
                        print(f"[DETERMINISTIC] {relation_type} of '{subject}': '{name}'")
                    return name

    return None


def _solve_yesno_same_country(question: str, passages: List[Dict[str, Any]], debug: bool) -> Optional[str]:
    """
    P1-4: YES/NO questions about whether two entities are from the same country.

    Example: "Are X and Y from the same country?"
    """
    q_lower = question.lower()

    # Extract two entity names from question
    entities = _extract_person_names_from_question(question)
    if len(entities) < 2:
        # Try extracting from "X and Y" pattern
        match = re.search(r'are\s+([A-ZÀ-Ÿ][^\s]+(?:\s+[A-ZÀ-Ÿ][^\s]+)*)\s+and\s+([A-ZÀ-Ÿ][^\s]+(?:\s+[A-ZÀ-Ÿ][^\s]+)*)\s+from', question, re.IGNORECASE)
        if match:
            entities = [match.group(1).strip(), match.group(2).strip()]
        else:
            return None

    if len(entities) < 2:
        return None

    # Extract countries for each entity from passages
    entity_countries = {}
    for passage in passages:
        text = passage.get("text", "")

        for entity in entities:
            if entity.lower() in text.lower() and entity not in entity_countries:
                # Look for country patterns near entity name
                # Pattern 1: "from COUNTRY"
                match = re.search(rf'{re.escape(entity)}[^.]*?from\s+([A-Z][a-zA-Z\s]{{2,30}})', text, re.IGNORECASE)
                if match:
                    country = match.group(1).strip()
                    # Clean up (remove trailing words like "where", "in", etc.)
                    country = re.split(r'\s+(?:where|in|and|,)', country)[0].strip()
                    entity_countries[entity] = country
                    continue

                # Pattern 2: "born in COUNTRY"
                match = re.search(rf'{re.escape(entity)}[^.]*?born in\s+([A-Z][a-zA-Z\s]{{2,30}})', text, re.IGNORECASE)
                if match:
                    country = match.group(1).strip()
                    country = re.split(r'\s+(?:where|in|and|,)', country)[0].strip()
                    entity_countries[entity] = country
                    continue

                # Pattern 3: "COUNTRY-born" or "COUNTRY native"
                match = re.search(r'([A-Z][a-zA-Z\s]{2,30})-born', text)
                if match and entity.lower() in text[max(0, match.start() - 100):match.end() + 50].lower():
                    country = match.group(1).strip()
                    entity_countries[entity] = country
                    continue

    if debug:
        print(f"[DETERMINISTIC] Country extraction: {entity_countries}")

    # Check if we found countries for both entities
    if len(entity_countries) >= 2:
        countries = list(entity_countries.values())
        same_country = countries[0].lower() == countries[1].lower()
        answer = "yes" if same_country else "no"

        if debug:
            print(f"[DETERMINISTIC] Same country? {entities[0]} ({countries[0]}) vs {entities[1]} ({countries[1]}) = {answer}")

        return answer

    return None


def _solve_director_birth_comparison(question: str, passages: List[Dict[str, Any]], debug: bool) -> Optional[str]:
    """
    P1-5: Director born earlier/later comparison (2-hop extraction).

    Example: "Which film's director was born first, X or Y?"

    2-hop logic:
    1. Extract film names from question
    2. For each film, find its director in passages
    3. For each director, find their birth year in passages
    4. Compare birth years and return the film whose director was born earlier/later
    """
    q_lower = question.lower()

    # Extract film names from question
    films = _extract_film_names_from_question(question)
    if len(films) < 2:
        if debug:
            print(f"[DETERMINISTIC] Director birth comparison: need 2 films, found {len(films)}")
        return None

    # HOP 1: Extract director for each film
    film_directors = {}
    for passage in passages:
        text = passage.get("text", "")

        for film in films:
            if film.lower() in text.lower() and film not in film_directors:
                # Look for "directed by" pattern near film name
                # Create a context window around the film mention
                film_idx = text.lower().find(film.lower())
                if film_idx != -1:
                    # Get 200 chars before and after film mention
                    context_start = max(0, film_idx - 200)
                    context_end = min(len(text), film_idx + len(film) + 200)
                    context = text[context_start:context_end]

                    # Pattern: "directed by Name"
                    match = re.search(r'directed\s+by\s+([\w\u00C0-\u024F]+(?:\s+[\w\u00C0-\u024F]+){0,3})', context, re.IGNORECASE)
                    if match:
                        director = match.group(1).strip()
                        # Validate: first word starts with uppercase
                        words = director.split()
                        if words and words[0] and words[0][0].isalpha() and words[0][0].isupper():
                            film_directors[film] = director
                            if debug:
                                print(f"[DETERMINISTIC] Found director for '{film}': '{director}'")
                            continue

    if len(film_directors) < 2:
        if debug:
            print(f"[DETERMINISTIC] Director birth comparison: found directors for {len(film_directors)}/2 films")
        return None

    # HOP 2: Extract birth year for each director
    director_birth_years = {}
    for passage in passages:
        text = passage.get("text", "")

        for film, director in film_directors.items():
            if director.lower() in text.lower() and director not in director_birth_years:
                # Look for birth year patterns near director name
                director_idx = text.lower().find(director.lower())
                if director_idx != -1:
                    # Get 300 chars before and after director mention
                    context_start = max(0, director_idx - 300)
                    context_end = min(len(text), director_idx + len(director) + 300)
                    context = text[context_start:context_end]

                    # Pattern: "born ... YEAR" or "born in ... YEAR"
                    match = re.search(r'born[^.]*?(\d{4})', context, re.IGNORECASE)
                    if match:
                        year = int(match.group(1))
                        # Sanity check: year should be reasonable (1800-2020)
                        if 1800 <= year <= 2020:
                            director_birth_years[director] = year
                            if debug:
                                print(f"[DETERMINISTIC] Found birth year for '{director}': {year}")
                            continue

    if len(director_birth_years) < 2:
        if debug:
            print(f"[DETERMINISTIC] Director birth comparison: found birth years for {len(director_birth_years)}/2 directors")
        return None

    # Compare birth years and return appropriate film
    # Determine if question asks for "first" (earlier) or "later"
    want_earlier = "first" in q_lower or "earlier" in q_lower

    # Map directors back to films
    film_years = {film: director_birth_years[director]
                  for film, director in film_directors.items()
                  if director in director_birth_years}

    if len(film_years) < 2:
        return None

    # Return film with earlier or later director birth year
    if want_earlier:
        result_film = min(film_years.items(), key=lambda x: x[1])
    else:
        result_film = max(film_years.items(), key=lambda x: x[1])

    if debug:
        print(f"[DETERMINISTIC] Director birth comparison:")
        for film, year in film_years.items():
            print(f"  {film}: director born {year}")
        print(f"  Answer: '{result_film[0]}' (director born {result_film[1]})")

    return result_film[0]


# Helper functions for deterministic solvers

def _extract_film_names_from_question(question: str) -> List[str]:
    """Extract film names from question (quoted or title case)."""
    films = []
    # Try quoted strings first
    quoted = re.findall(r'["""]([^"""]+)["""]', question)
    films.extend(quoted)
    # If less than 2, try title case spans
    if len(films) < 2:
        title_case = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question)
        films.extend(title_case)
    return films[:2]  # Limit to 2


def _extract_person_names_from_question(question: str) -> List[str]:
    """Extract person names from question (title case spans)."""
    return re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', question)[:2]


def _extract_subject_from_question(question: str) -> Optional[str]:
    """Extract the subject (person/entity) from a question."""
    # Look for title case name after "was" or "were"
    match = re.search(r'(?:was|were)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', question)
    if match:
        return match.group(1)
    # Fallback: first title case span
    match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', question)
    if match:
        return match.group(1)
    return None


def _extract_year_near_text(text: str, anchor: str, window: int = 100) -> Optional[int]:
    """Extract year (4-digit number) near an anchor text."""
    # Find anchor position
    idx = text.lower().find(anchor.lower())
    if idx == -1:
        return None

    # Look in window around anchor
    start = max(0, idx - window)
    end = min(len(text), idx + len(anchor) + window)
    snippet = text[start:end]

    # Find 4-digit years (1900-2099)
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', snippet)
    if years:
        return int(years[0])
    return None


def _extract_death_year(text: str, person: str) -> Optional[int]:
    """Extract death year from text mentioning a person."""
    idx = text.lower().find(person.lower())
    if idx == -1:
        return None

    # Look for patterns like "died 1990", "1920-1990", "died on ... 1990"
    snippet = text[max(0, idx - 50):min(len(text), idx + len(person) + 200)]

    # Pattern 1: "died ... YEAR"
    match = re.search(r'died[^0-9]{0,30}(19\d{2}|20\d{2})', snippet, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Pattern 2: "YEAR1–YEAR2" or "YEAR1-YEAR2" (birth-death range)
    match = re.search(r'(19\d{2}|20\d{2})\s*[–-]\s*(19\d{2}|20\d{2})', snippet)
    if match:
        return int(match.group(2))  # Return second year (death)

    return None


# =============================================================================
# CHANGE 1: QUESTION-INTENT → ANSWER-TYPE GATING
# =============================================================================

def detect_question_intent(question: str) -> str:
    """
    Detect the high-level intent of a question to gate answer types.

    This prevents type mismatches like:
    - "Who won award X?" returning entity instead of award name
    - "Which film came first?" returning date instead of film title

    Returns:
        Intent category: "temporal", "quantity", "comparison", "award", "person",
                        "location", "yesno", "entity"
    """
    q_lower = question.lower()

    # Temporal questions: when, what year, what date
    if any(word in q_lower for word in ["when did", "what year", "what date"]):
        return "temporal"

    # Quantity questions: how many, how much
    if any(phrase in q_lower for phrase in ["how many", "how much"]):
        return "quantity"

    # Comparison questions: which ... first, which ... earlier
    if "which" in q_lower and any(word in q_lower for word in ["first", "earlier", "later", "before", "after"]):
        return "comparison"

    # Award questions: what award, which award, what prize
    if any(phrase in q_lower for phrase in ["what award", "which award", "what prize", "which prize", "won the", "received the"]):
        return "award"

    # Location questions: where, what city, what country
    if any(word in q_lower for word in ["where was", "where is", "what city", "what country", "what place"]):
        return "location"

    # Person questions: who is, who was, who did
    if any(phrase in q_lower for phrase in ["who is", "who was", "who were", "who did", "who directed", "who wrote"]):
        return "person"

    # Yes/no questions
    if q_lower.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "will ", "would ")):
        return "yesno"

    # Default to entity
    return "entity"


def get_allowed_answer_types(intent: str) -> Set[str]:
    """
    Map question intent to allowed answer types.

    This enforces answer-type consistency to prevent LLM cert from
    returning wrong answer category.
    """
    INTENT_TO_TYPES = {
        "temporal": {"date", "number", "entity"},  # Date or year (as number/entity)
        "comparison": {"entity", "choice"},  # For "which X" questions
        "award": {"entity"},  # Award names are entities
        "person": {"entity"},
        "location": {"entity"},  # Locations are entities (cities, countries, etc.)
        "yesno": {"yesno"},
        "quantity": {"number", "entity"},  # For "how many" questions
        "entity": {"entity", "date", "number"},  # Generic fallback
        "other": {"entity", "date", "number", "yesno", "choice"}  # Permissive
    }
    return INTENT_TO_TYPES.get(intent, {"entity", "date", "number", "yesno", "choice"})


def llm_route_relation(
    llm,
    question: str,
    available_types: Optional[List[str]] = None
) -> Tuple[str, float]:
    """
    Use LLM to determine the primary relation type being asked about.

    This replaces keyword-based relation detection with semantic understanding.

    Args:
        llm: LLM interface with generate() method
        question: The question to analyze
        available_types: Optional list of relation types to choose from

    Returns:
        Tuple of (relation_type, confidence)
    """
    import json as json_module
    import os

    debug = os.environ.get("TRIDENT_DEBUG_ROUTER", "0") == "1"

    if available_types is None:
        available_types = KNOWN_RELATION_TYPES

    types_str = ", ".join(available_types)

    prompt = f"""Identify the relation type being asked about in this question.

Question: {question}

Available relation types: {types_str}

Rules:
1. Pick the SINGLE most specific relation type
2. For "who is the mother of X" -> MOTHER (not PARENT)
3. For "where was X born" -> BIRTHPLACE
4. For "who directed X" -> DIRECTOR
5. Return JSON: {{"relation": "TYPE", "confidence": 0.9}}

JSON:"""

    try:
        raw = llm.generate(prompt, temperature=0.0, max_new_tokens=96)
        raw_text = raw.text if hasattr(raw, 'text') else str(raw)
    except Exception as e:
        if debug:
            print(f"[LLM-ROUTER] Error: {e}")
        return keyword_route_relation(question), 0.5

    if debug:
        print(f"[LLM-ROUTER] Raw output: {raw_text[:100]}...")

    # Parse JSON response
    try:
        out = json_module.loads(_extract_first_json_object(raw_text))
    except Exception as e:
        if debug:
            print(f"[LLM-ROUTER] Parse error: {e}")
        return keyword_route_relation(question), 0.5

    relation = (out.get("relation") or "").upper().strip()
    confidence = float(out.get("confidence", 0.7))

    # Validate relation type
    if relation not in available_types and relation != "OTHER":
        # Try to find closest match
        for t in available_types:
            if t in relation or relation in t:
                relation = t
                break
        else:
            # Fallback to keyword-based
            if debug:
                print(f"[LLM-ROUTER] Unknown relation '{relation}', falling back to keywords")
            return keyword_route_relation(question), 0.5

    if debug:
        print(f"[LLM-ROUTER] Result: {relation} (confidence={confidence})")

    return relation, confidence


def keyword_route_relation(question: str) -> str:
    """
    Fallback keyword-based relation routing.

    Used when LLM router is unavailable or fails.
    """
    q_lower = question.lower()

    # Check each relation type's triggers
    for relation_type, triggers in REL_TRIGGERS.items():
        for trigger in triggers:
            if trigger in q_lower:
                return relation_type

    return "OTHER"


def route_relation(
    question: str,
    llm: Optional[Any] = None,
    use_llm: bool = True
) -> Tuple[str, float, str]:
    """
    Route question to relation type using LLM (if available) or keywords.

    Args:
        question: The question to analyze
        llm: Optional LLM interface
        use_llm: Whether to use LLM routing (default True)

    Returns:
        Tuple of (relation_type, confidence, method)
        method is "llm" or "keyword"
    """
    if use_llm and llm is not None:
        try:
            relation, confidence = llm_route_relation(llm, question)
            return relation, confidence, "llm"
        except Exception:
            pass

    # Fallback to keyword routing
    relation = keyword_route_relation(question)
    return relation, 0.5, "keyword"


def _looks_like_person(name: str) -> bool:
    """
    General PERSON-ish validator for CSS entity binding.

    This is NOT relation-specific - it's a lightweight type check that accepts
    strings that look like person names and rejects obvious non-names.

    Accepts: "Xawery Żuławski", "John Smith", "Madonna", "李明"
    Rejects: "Polish film", "the movie", "123", "", "directed by someone"
    """
    if not name or len(name) < 2:
        return False

    # Must contain at least one letter (handles Unicode)
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", name):
        return False

    # Split into tokens
    toks = [t for t in re.split(r"\s+", name.strip()) if t]
    if not toks:
        return False

    # Single-token names must start with uppercase (or be CJK)
    if len(toks) == 1:
        first_char = toks[0][0]
        # Allow CJK characters
        if re.match(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", first_char):
            return True
        # Must start with uppercase for Latin names
        return bool(re.match(r"^[A-ZÀ-ÖØ-Þ]", toks[0]))

    # Multi-token: reject if starts with true lowercase (not uppercase/CJK)
    first_char = name[0]
    if re.match(r"^[a-zà-öø-ÿ]", first_char):
        return False

    # Too many tokens is suspicious (likely a description, not a name)
    if len(toks) > 5:
        return False

    # Reject obvious non-name patterns (case-insensitive for these)
    bad_words = {"film", "movie", "city", "country", "war", "book", "song",
                 "album", "series", "the", "directed", "starring", "featuring"}
    toks_lower = [t.lower().strip(".,()") for t in toks]
    if any(t in bad_words for t in toks_lower):
        return False

    return True


@dataclass
class ChainHop:
    """A single hop in the reasoning chain."""
    passage_id: str
    passage_text: str
    passage_title: Optional[str]
    entities: Set[str]
    relation_kinds: Set[str]
    hop_number: int  # 1 for hop1, 2 for hop2


@dataclass
class ReasoningChain:
    """A complete multi-hop reasoning chain."""
    hop1: ChainHop
    bridge_entity: Optional[str]
    hop2: ChainHop
    score: float

    def get_ordered_passages(self) -> List[Dict[str, Any]]:
        """Return passages in hop order for prompt building."""
        return [
            {
                'pid': self.hop1.passage_id,
                'text': self.hop1.passage_text,
                'title': self.hop1.passage_title,
                'hop': 1,
                'bridge_entity': self.bridge_entity,
            },
            {
                'pid': self.hop2.passage_id,
                'text': self.hop2.passage_text,
                'title': self.hop2.passage_title,
                'hop': 2,
                'bridge_entity': self.bridge_entity,
            }
        ]


@dataclass
class AnswerCertificate:
    """
    LLM Answer Certificate with provenance (Step 3).

    This is a SEPARATE certificate type from facet conformal certificates.
    Used as fallback when Safe-Cover has no answer facets certified.

    Hard acceptance rules (non-negotiable):
    - quote must be a substring of passage_text(pid)
    - answer must be a substring of quote
    - for yes/no or choice: quote must contain decisive fact
    """
    answer: str  # The extracted answer
    pid: str  # Supporting passage ID
    quote: str  # Exact substring from that passage containing answer
    answer_type: str  # entity/date/number/yesno/choice
    confidence: Optional[float] = None  # Optional, don't trust alone
    verified: bool = False  # Set to True after verification passes
    tier2_backend: Optional[str] = None  # P0-2: Track backend used (vllm, hf)
    tier2_fail_reason: Optional[str] = None  # P0-2: Track why vLLM failed (if HF was used)


def extract_entities(text: str, doc_title: Optional[str] = None) -> Set[str]:
    """Extract entity mentions from text.

    Uses Title Case spans and optionally includes document title.
    """
    ents = set(ENTITY_RE.findall(text))
    if doc_title:
        ents.add(doc_title)
        # Also add normalized version
        ents.add(doc_title.strip())
    return ents


def extract_relation_kinds(text: str) -> Set[str]:
    """Identify which relation kinds are mentioned in the passage."""
    t = text.lower()
    hits = set()
    for kind, triggers in REL_TRIGGERS.items():
        if any(trigger in t for trigger in triggers):
            hits.add(kind)
    return hits


def get_question_entities(question: str, facets: List[Dict[str, Any]]) -> Set[str]:
    """Extract entities from question and facets.

    Uses ENTITY facets if available, otherwise falls back to regex.
    """
    q_ents = set()

    # Extract from ENTITY facets
    for facet in facets:
        if facet.get('facet_type') == 'ENTITY':
            template = facet.get('template', {})
            mention = template.get('mention', '')
            if mention:
                q_ents.add(mention)

    # Also extract from RELATION facets
    for facet in facets:
        if facet.get('facet_type') == 'RELATION':
            template = facet.get('template', {})
            subj = template.get('subject', '')
            obj = template.get('object', '')
            # Clean WH-words
            if subj and subj.lower() not in {'who', 'what', 'where', 'when', 'which'}:
                q_ents.add(subj)
            if obj:
                # Extract entity from object like "director of film Polish-Russian War"
                obj_ents = extract_entities(obj)
                q_ents.update(obj_ents)

    # Fallback: regex on question
    q_ents.update(extract_entities(question))

    return q_ents


def get_required_relation_kinds(facets: List[Dict[str, Any]]) -> Set[str]:
    """Get required relation kinds from facets."""
    required = set()

    for facet in facets:
        if facet.get('facet_type') == 'RELATION':
            template = facet.get('template', {})
            # Check predicate and full facet text
            predicate = template.get('predicate', '').lower()
            facet_text = f"{template.get('subject', '')} {template.get('object', '')} {predicate}".lower()

            for kind, triggers in REL_TRIGGERS.items():
                if any(t in facet_text for t in triggers):
                    required.add(kind)

    return required


def build_chain_from_certified(
    certified_passages: List[Dict[str, Any]],
    question: str,
    facets: List[Dict[str, Any]],
    certificates: Optional[List[Dict[str, Any]]] = None
) -> Optional[ReasoningChain]:
    """Build a multi-hop reasoning chain from certified passages.

    Args:
        certified_passages: Passages that have been certified by Safe-Cover
        question: The original question
        facets: List of facet dicts
        certificates: Optional certificate info for filtering

    Returns:
        ReasoningChain if a valid chain is found, None otherwise
    """
    if len(certified_passages) < 2:
        # Need at least 2 passages for a 2-hop chain
        # If only 1 passage, return it as a single-hop chain (no bridge required)
        if len(certified_passages) == 1:
            p = certified_passages[0]
            ents = extract_entities(p.get('text', ''), p.get('title'))
            kinds = extract_relation_kinds(p.get('text', ''))
            hop = ChainHop(
                passage_id=p.get('pid', ''),
                passage_text=p.get('text', ''),
                passage_title=p.get('title'),
                entities=ents,
                relation_kinds=kinds,
                hop_number=1
            )
            # Single passage chain - explicit None for bridge
            return ReasoningChain(hop1=hop, bridge_entity=None, hop2=hop, score=0.5)
        return None

    # Get question entities and required relation kinds
    q_ents = get_question_entities(question, facets)
    required_kinds = get_required_relation_kinds(facets)

    # Build hop info for each passage
    hops = []
    for p in certified_passages:
        text = p.get('text', '')
        title = p.get('title')
        ents = extract_entities(text, title)
        kinds = extract_relation_kinds(text)
        hops.append(ChainHop(
            passage_id=p.get('pid', ''),
            passage_text=text,
            passage_title=title,
            entities=ents,
            relation_kinds=kinds,
            hop_number=0  # Will be set later
        ))

    # Index entities to passages
    ent2hop = defaultdict(list)
    for i, hop in enumerate(hops):
        for ent in hop.entities:
            ent2hop[ent].append(i)

    # Find bridge entities (appear in >= 2 passages)
    bridges = [ent for ent, idxs in ent2hop.items() if len(set(idxs)) >= 2]

    # Score candidate chains
    best_chain = None
    best_score = -1

    for bridge in bridges:
        passage_indices = list(set(ent2hop[bridge]))

        # Try all pairs of passages containing the bridge
        for i in passage_indices:
            for j in passage_indices:
                if i == j:
                    continue

                hop1 = hops[i]
                hop2 = hops[j]

                # Score this chain
                score = 0.0

                # Hop1 should connect to question entities
                if hop1.entities & q_ents:
                    score += 2.0

                # Hop2 should have required relation kind
                if required_kinds and (hop2.relation_kinds & required_kinds):
                    score += 2.0
                elif not required_kinds:
                    score += 1.0  # No specific requirement

                # Bonus for distinct passages
                score += 1.0

                # Bonus if bridge entity is in the question
                if bridge in q_ents:
                    score += 0.5

                if score > best_score:
                    best_score = score
                    hop1_copy = ChainHop(
                        passage_id=hop1.passage_id,
                        passage_text=hop1.passage_text,
                        passage_title=hop1.passage_title,
                        entities=hop1.entities,
                        relation_kinds=hop1.relation_kinds,
                        hop_number=1
                    )
                    hop2_copy = ChainHop(
                        passage_id=hop2.passage_id,
                        passage_text=hop2.passage_text,
                        passage_title=hop2.passage_title,
                        entities=hop2.entities,
                        relation_kinds=hop2.relation_kinds,
                        hop_number=2
                    )
                    best_chain = ReasoningChain(
                        hop1=hop1_copy,
                        bridge_entity=bridge,
                        hop2=hop2_copy,
                        score=best_score
                    )

    # If no bridge found, fall back to ordering by question entity overlap
    if best_chain is None and len(hops) >= 2:
        # Order by overlap with question entities (descending)
        hops_sorted = sorted(
            enumerate(hops),
            key=lambda x: len(x[1].entities & q_ents),
            reverse=True
        )
        idx1, hop1 = hops_sorted[0]
        idx2, hop2 = hops_sorted[1] if len(hops_sorted) > 1 else hops_sorted[0]

        # Find a common entity as bridge
        common = hop1.entities & hop2.entities
        # Use None if no common bridge exists to avoid prompt hallucinations
        bridge = next(iter(common), None)

        hop1_copy = ChainHop(
            passage_id=hop1.passage_id,
            passage_text=hop1.passage_text,
            passage_title=hop1.passage_title,
            entities=hop1.entities,
            relation_kinds=hop1.relation_kinds,
            hop_number=1
        )
        hop2_copy = ChainHop(
            passage_id=hop2.passage_id,
            passage_text=hop2.passage_text,
            passage_title=hop2.passage_title,
            entities=hop2.entities,
            relation_kinds=hop2.relation_kinds,
            hop_number=2
        )
        best_chain = ReasoningChain(
            hop1=hop1_copy,
            bridge_entity=bridge,
            hop2=hop2_copy,
            score=0.0
        )

    return best_chain


def extract_grounded_answer(
    llm_answer: str,
    hop2_text: str,
    min_overlap_ratio: float = 0.5
) -> Tuple[str, bool]:
    """Extract answer that is grounded in hop2 passage.

    Args:
        llm_answer: Raw answer from LLM
        hop2_text: Text of the hop2 passage
        min_overlap_ratio: Minimum token overlap for grounding

    Returns:
        Tuple of (grounded_answer, is_grounded)
    """
    if not llm_answer or not hop2_text:
        return llm_answer, False

    # Helper for normalization
    def norm(s):
        s = s.lower().strip()
        # Remove punctuation
        s = s.translate(str.maketrans("", "", string.punctuation))
        # Normalize whitespace
        s = " ".join(s.split())
        return s

    # 1. Exact substring check (original)
    answer_lower = llm_answer.lower().strip()
    hop2_lower = hop2_text.lower()

    if answer_lower in hop2_lower:
        return llm_answer.strip(), True

    # 2. Normalized substring check (punctuation agnostic)
    norm_answer = norm(llm_answer)
    norm_hop2 = norm(hop2_text)

    if norm_answer and norm_answer in norm_hop2:
        return llm_answer.strip(), True

    # 3. Sliding window token overlap (fallback)
    # Try to find the best matching substring
    answer_tokens = answer_lower.split()
    if not answer_tokens:
        return llm_answer, False

    # Sliding window to find best match
    hop2_tokens = hop2_lower.split()
    best_match = None
    best_overlap = 0

    for window_size in range(len(answer_tokens), 0, -1):
        for start in range(len(hop2_tokens) - window_size + 1):
            window = hop2_tokens[start:start + window_size]
            overlap = len(set(answer_tokens) & set(window))
            overlap_ratio = overlap / len(answer_tokens)

            if overlap_ratio >= min_overlap_ratio and overlap > best_overlap:
                best_overlap = overlap
                best_match = ' '.join(hop2_text.split()[start:start + window_size])

    if best_match:
        return best_match, True

    # Fall back to original answer with grounding flag
    return llm_answer.strip(), False


def build_single_hop_prompt(question: str, chain: ReasoningChain) -> str:
    """Build a simplified prompt for single-passage questions."""
    prompt_parts = []
    prompt_parts.append("Answer the following question using ONLY the provided evidence passage.")
    prompt_parts.append("")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("")
    prompt_parts.append("=== Evidence ===")
    if chain.hop1.passage_title:
        prompt_parts.append(f"[{chain.hop1.passage_title}]")
    prompt_parts.append(chain.hop1.passage_text)
    prompt_parts.append("")
    prompt_parts.append("Instructions:")
    prompt_parts.append("1. The answer must be found in the Evidence passage")
    prompt_parts.append("2. Extract the exact answer span")
    prompt_parts.append("3. If the answer cannot be found, respond with: I cannot answer based on the given context.")
    prompt_parts.append("")
    prompt_parts.append("Answer:")
    return "\n".join(prompt_parts)


def build_chain_prompt(
    question: str,
    chain: ReasoningChain,
    facets: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build a prompt that uses the reasoning chain structure.

    This prompt explicitly shows the hop structure and asks the LLM
    to extract an answer that is grounded in the evidence.
    """
    # Use simplified prompt for single-hop cases (Hop1 == Hop2)
    if chain.hop1.passage_id == chain.hop2.passage_id:
        return build_single_hop_prompt(question, chain)

    prompt_parts = []
    prompt_parts.append("Answer the following question using ONLY the provided evidence passages.")
    prompt_parts.append("The passages are ordered as a reasoning chain: Hop 1 introduces context, Hop 2 contains the answer.")
    prompt_parts.append("")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("")

    # Only include Bridge Entity if it exists (multi-hop with overlap)
    if chain.bridge_entity:
        prompt_parts.append(f"Bridge Entity: {chain.bridge_entity}")
        prompt_parts.append("")

    prompt_parts.append("=== HOP 1 (Context) ===")
    if chain.hop1.passage_title:
        prompt_parts.append(f"[{chain.hop1.passage_title}]")
    prompt_parts.append(chain.hop1.passage_text)
    prompt_parts.append("")
    prompt_parts.append("=== HOP 2 (Answer Source) ===")
    if chain.hop2.passage_title:
        prompt_parts.append(f"[{chain.hop2.passage_title}]")
    prompt_parts.append(chain.hop2.passage_text)
    prompt_parts.append("")
    prompt_parts.append("Instructions:")
    prompt_parts.append("1. If the answer is stated verbatim in the evidence, extract the EXACT span (copy it exactly).")
    prompt_parts.append("2. Return ONLY the answer - no explanation, no extra words.")
    prompt_parts.append("3. Preserve accents and special characters exactly as written.")
    prompt_parts.append("4. If the answer is not stated in the evidence, respond exactly: I cannot answer based on the given context.")
    prompt_parts.append("")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def get_winning_passages_from_certificates(
    certificates: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    facets: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Map facet_id -> winning passage, plus facet_type -> [winning passages].

    CRITICAL: Maps by facet_id to preserve all facets (a query can have multiple
    RELATION or ENTITY facets). Also provides a grouped view by facet_type.

    Returns:
        Tuple of:
        - facet_id_map: Dict[facet_id -> {'passage', 'facet', 'p_value', 'passage_id', 'facet_id'}]
        - facet_type_map: Dict[facet_type -> List[winning info dicts]] (stable order by p_value)
    """
    if not certificates:
        return {}, {}

    # Build passage lookup by pid
    pid_to_passage = {p.get('pid', ''): p for p in passages}

    # Build facet lookup by facet_id
    fid_to_facet = {f.get('facet_id', ''): f for f in facets}

    # Primary map: facet_id -> winning info
    facet_id_map: Dict[str, Dict[str, Any]] = {}

    for cert in certificates:
        fid = cert.get('facet_id', '')
        pid = cert.get('passage_id', '')
        p_val = cert.get('p_value', 1.0)

        facet = fid_to_facet.get(fid, {})
        passage = pid_to_passage.get(pid, {})

        if not facet or not passage:
            continue

        ftype = facet.get('facet_type', '')

        # Only keep best certificate per facet_id (in case of duplicates)
        if fid not in facet_id_map or p_val < facet_id_map[fid]['p_value']:
            facet_id_map[fid] = {
                'passage': passage,
                'facet': facet,
                'p_value': p_val,
                'passage_id': pid,
                'facet_id': fid,
                'facet_type': ftype
            }

    # Derived map: facet_type -> [winning info dicts] (sorted by p_value)
    facet_type_map: Dict[str, List[Dict[str, Any]]] = {}
    for info in facet_id_map.values():
        ftype = info['facet_type']
        if ftype not in facet_type_map:
            facet_type_map[ftype] = []
        facet_type_map[ftype].append(info)

    # Sort each list by p_value (best first)
    for ftype in facet_type_map:
        facet_type_map[ftype].sort(key=lambda x: x['p_value'])

    return facet_id_map, facet_type_map


def build_certificate_aware_prompt(
    question: str,
    certificates: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    facets: List[Dict[str, Any]]
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Build a prompt using the certificate-winning passages.

    CRITICAL: Uses the passages that actually certified each facet,
    not arbitrary chain hops based on entity overlap. Includes ALL
    winning passages (not just one per facet type).

    Returns:
        Tuple of:
        - prompt_string: The constructed prompt
        - relation_winning_info: The best RELATION facet winner (for typed extraction)
        - facet_id_map: Complete facet_id -> winning info map (for debugging/validation)
    """
    facet_id_map, facet_type_map = get_winning_passages_from_certificates(
        certificates, passages, facets
    )

    if not facet_id_map:
        # Fallback: no certificates, can't build certificate-aware prompt
        return "", None, {}

    # Get ALL RELATION-winning passages (sorted by p_value, best first)
    relation_winners = facet_type_map.get('RELATION', [])
    entity_winners = facet_type_map.get('ENTITY', [])

    # Also check for BRIDGE_HOP types
    bridge_winners = []
    for ftype in ['BRIDGE_HOP', 'BRIDGE_HOP1', 'BRIDGE_HOP2']:
        if ftype in facet_type_map:
            bridge_winners.extend(facet_type_map[ftype])

    # The best RELATION winner (lowest p-value) is the primary answer source
    best_relation_info = relation_winners[0] if relation_winners else None

    prompt_parts = []
    prompt_parts.append("Answer the following question using ONLY the provided certified evidence.")
    prompt_parts.append("These passages have been verified to contain the information needed to answer.")
    prompt_parts.append("")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("")

    # Include ALL winning passages (deduplicated by pid)
    seen_pids = set()
    passage_num = 1

    # ENTITY passages first (context)
    for info in entity_winners:
        p = info['passage']
        pid = p.get('pid', '')
        if pid and pid not in seen_pids:
            seen_pids.add(pid)
            title = p.get('title', '')
            text = p.get('text', '')
            prompt_parts.append(f"=== Evidence {passage_num} (Context) ===")
            if title:
                prompt_parts.append(f"[{title}]")
            prompt_parts.append(text)
            prompt_parts.append("")
            passage_num += 1

    # BRIDGE passages (if different from ENTITY)
    for info in bridge_winners:
        p = info['passage']
        pid = p.get('pid', '')
        if pid and pid not in seen_pids:
            seen_pids.add(pid)
            title = p.get('title', '')
            text = p.get('text', '')
            prompt_parts.append(f"=== Evidence {passage_num} (Bridge) ===")
            if title:
                prompt_parts.append(f"[{title}]")
            prompt_parts.append(text)
            prompt_parts.append("")
            passage_num += 1

    # RELATION passages (answer sources - MOST IMPORTANT!)
    for i, info in enumerate(relation_winners):
        p = info['passage']
        pid = p.get('pid', '')
        if pid and pid not in seen_pids:
            seen_pids.add(pid)
            title = p.get('title', '')
            text = p.get('text', '')
            label = "Answer Source" if i == 0 else f"Answer Source {i + 1}"
            prompt_parts.append(f"=== Evidence {passage_num} ({label}) ===")
            if title:
                prompt_parts.append(f"[{title}]")
            prompt_parts.append(text)
            prompt_parts.append("")
            passage_num += 1
        elif passage_num == 1 and i == 0:
            # RELATION is the only passage - still show it
            title = p.get('title', '')
            text = p.get('text', '')
            prompt_parts.append("=== Certified Evidence ===")
            if title:
                prompt_parts.append(f"[{title}]")
            prompt_parts.append(text)
            prompt_parts.append("")
            passage_num += 1

    # If no passages were added, bail
    if passage_num == 1:
        return "", None, facet_id_map

    prompt_parts.append("Instructions:")
    prompt_parts.append("1. If the answer is stated verbatim in the evidence, extract the EXACT span (copy it exactly).")
    prompt_parts.append("2. Return ONLY the answer - no explanation, no extra words.")
    prompt_parts.append("3. Preserve accents and special characters exactly as written.")
    prompt_parts.append("4. If the answer is not stated in the evidence, respond exactly: I cannot answer based on the given context.")
    prompt_parts.append("")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts), best_relation_info, facet_id_map


def _expand_name_span(text: str, initial_match: str, passage_title: str = "") -> str:
    """
    P0-1 FIX: Expand a partial name match to the longest plausible person-name span.

    This fixes CSS binding expansion failures like "Xawery" → "Xawery Żuławski".

    Args:
        text: The full passage text
        initial_match: The initially matched name substring (e.g., "Xawery")
        passage_title: Optional passage title to prefer spans from

    Returns:
        The expanded name (e.g., "Xawery Żuławski")
    """
    # Find the token index where initial_match occurs
    idx = text.find(initial_match)
    if idx == -1:
        return initial_match

    # Expand left over contiguous name tokens (Unicode letters, apostrophes, hyphens, spaces)
    start = idx
    while start > 0:
        char = text[start - 1]
        # Stop at non-name characters
        if not (char.isalpha() or char in "'-\u00C0-\u024F\u1E00-\u1EFF" or (char == ' ' and start > 1 and text[start - 2].isalpha())):
            break
        start -= 1

    # Expand right over contiguous name tokens
    end = idx + len(initial_match)
    while end < len(text):
        char = text[end]
        # Stop at sentence delimiters or non-name characters
        if char in '.,;()\n':
            break
        # Continue over letters, accented chars, apostrophes, hyphens, and spaces between name parts
        if not (char.isalpha() or char in "'-\u00C0-\u024F\u1E00-\u1EFF" or (char == ' ' and end + 1 < len(text) and (text[end + 1].isalpha() or text[end + 1] in '\u00C0-\u024F\u1E00-\u1EFF'))):
            break
        end += 1

    expanded = text[start:end].strip()

    # Prefer spans with ≥2 tokens
    words = expanded.split()
    if len(words) < 2:
        # Try to find a better match in title or first sentence
        if passage_title:
            title_words = passage_title.split()
            for i in range(len(title_words) - 1):
                candidate = ' '.join(title_words[i:i+2])
                if initial_match in candidate:
                    expanded = candidate
                    break

        # If still single-word, try first sentence
        if len(expanded.split()) < 2:
            first_sent = text.split('.')[0] if '.' in text else text[:200]
            # P0-3: Unicode-safe multi-word name extraction using .isupper() and .isalpha()
            # This handles Polish (Ż), French (É), etc. that fall outside ASCII or À-Ÿ range
            tokens = first_sent.split()
            name_matches = []
            i = 0
            while i < len(tokens):
                # Start of a potential name: first char is uppercase letter
                if tokens[i] and tokens[i][0].isalpha() and tokens[i][0].isupper():
                    # Greedily collect subsequent title-case tokens
                    name_parts = [tokens[i]]
                    j = i + 1
                    while j < len(tokens):
                        # Continue if token starts with uppercase letter (handles diacritics)
                        if tokens[j] and tokens[j][0].isalpha() and tokens[j][0].isupper():
                            name_parts.append(tokens[j])
                            j += 1
                        else:
                            break
                    # Only keep multi-word names (≥2 tokens)
                    if len(name_parts) >= 2:
                        name_matches.append(' '.join(name_parts))
                    i = j
                else:
                    i += 1

            # Find candidate containing initial_match
            for candidate in name_matches:
                if initial_match in candidate:
                    expanded = candidate
                    break

    return expanded


def bind_entity_from_hop1_winner(
    relation_type: str,
    passage_text: str
) -> Optional[str]:
    """
    Typed binding of the intermediate entity from hop-1 winner passage.

    This is NOT answer extraction - it's binding a variable for hop-2.
    Much easier because we're extracting a named entity from one known relation.

    P0-1 FIX: Now expands partial names to full names (e.g., "Xawery" → "Xawery Żuławski")

    Args:
        relation_type: The inner relation type (DIRECTOR, AUTHOR, etc.)
        passage_text: The text of the hop-1 winning passage

    Returns:
        The bound entity name (e.g., "Xawery Żuławski"), or None if not found
    """
    if not passage_text:
        return None

    t = passage_text
    t_lower = t.lower()

    # P0-1: Extract passage title if available (often in first line before newline or period)
    passage_title = t.split('\n')[0] if '\n' in t else t.split('.')[0]

    if relation_type == "DIRECTOR":
        # Must have "directed" to extract director name
        if "directed" not in t_lower and "director" not in t_lower:
            return None

        # P0-1 FIX: Use simpler patterns + expand to longest plausible name
        # Pattern: "directed by Name" (most reliable)
        m = re.search(r"\bdirected\s+by\s+([A-ZÀ-Ÿ][^\n.,;()]*)", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            # Expand to full name (e.g., "Xawery" → "Xawery Żuławski")
            name = _expand_name_span(t, initial_name, passage_title)
            words = name.split()
            if 1 <= len(words) <= 4:
                return name

        # Pattern: "Name directed" or "Name, who directed"
        m = re.search(r"([A-ZÀ-Ÿ][^\n.,;()]*?)\s+(?:,\s*who\s+)?directed\b", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            name = _expand_name_span(t, initial_name, passage_title)
            words = name.split()
            if 1 <= len(words) <= 4 and words[0].lower() not in {'the', 'a', 'an', 'film', 'movie'}:
                return name

        # Pattern: "X is an Australian director" -> extract X
        m = re.search(r"([A-ZÀ-Ÿ][^\n.,;()]*?)\s+(?:is|was)\s+(?:an?\s+)?(?:\w+\s+)?director", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            name = _expand_name_span(t, initial_name, passage_title)
            words = name.split()
            if 1 <= len(words) <= 4:
                return name

    elif relation_type == "AUTHOR":
        if "written" not in t_lower and "wrote" not in t_lower and "author" not in t_lower:
            return None

        m = re.search(r"\bwritten\s+by\s+([A-ZÀ-Ÿ][^\n.,;()]*)", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            return _expand_name_span(t, initial_name, passage_title)

        m = re.search(r"([A-ZÀ-Ÿ][^\n.,;()]*?)\s+wrote\b", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            return _expand_name_span(t, initial_name, passage_title)

    elif relation_type == "CREATOR":
        if "created" not in t_lower and "founder" not in t_lower:
            return None

        m = re.search(r"\bcreated\s+by\s+([A-ZÀ-Ÿ][^\n.,;()]*)", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            return _expand_name_span(t, initial_name, passage_title)

        m = re.search(r"\bfounded\s+by\s+([A-ZÀ-Ÿ][^\n.,;()]*)", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            return _expand_name_span(t, initial_name, passage_title)

    elif relation_type == "PRODUCER":
        if "produced" not in t_lower and "producer" not in t_lower:
            return None

        m = re.search(r"\bproduced\s+by\s+([A-ZÀ-Ÿ][^\n.,;()]*)", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            return _expand_name_span(t, initial_name, passage_title)

    elif relation_type == "COMPOSER":
        if "composed" not in t_lower and "composer" not in t_lower:
            return None

        m = re.search(r"\bcomposed\s+by\s+([A-ZÀ-Ÿ][^\n.,;()]*)", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            return _expand_name_span(t, initial_name, passage_title)

    elif relation_type == "PERFORMER":
        if "starred" not in t_lower and "performed" not in t_lower and "starring" not in t_lower:
            return None

        m = re.search(r"\bstarring\s+([A-ZÀ-Ÿ][^\n.,;()]*)", t, re.IGNORECASE)
        if m:
            initial_name = m.group(1).strip()
            return _expand_name_span(t, initial_name, passage_title)

    return None


def typed_extract_from_winning_passage(
    question: str,
    relation_info: Dict[str, Any]
) -> Optional[str]:
    """
    Facet-ID–aware typed extraction from the RELATION-winning passage.

    CRITICAL FIXES:
    1. Uses facet template (subject/object/predicate) to constrain extraction
    2. Checks semantic presence before extracting (abort if answer type absent)
    3. Anchors on known entity spans from the facet template
    4. Returns None to allow controlled abstention rather than garbage

    Args:
        question: The original question
        relation_info: Dict with 'passage', 'facet', 'p_value', etc.

    Returns:
        Extracted answer string, or None if extraction not possible
    """
    if not relation_info:
        return None

    passage = relation_info.get('passage', {})
    facet = relation_info.get('facet', {})
    text = passage.get('text', '')

    if not text:
        return None

    # Get facet template - this is the key to facet-ID awareness
    template = facet.get('template', {})
    subject = template.get('subject', '').strip()
    obj = template.get('object', '').strip()
    predicate = template.get('predicate', '').strip()

    # Build context for relation kind detection
    facet_text = f"{subject} {obj} {predicate}".lower()

    # Determine relation kind from facet template
    relation_kind = None
    for kind, triggers in REL_TRIGGERS.items():
        if any(t in facet_text for t in triggers):
            relation_kind = kind
            break

    # Fallback: infer from question (less reliable)
    if not relation_kind:
        q = question.lower()
        if "director" in q or "directed" in q:
            relation_kind = "DIRECTOR"
        elif "born" in q or "birth" in q:
            relation_kind = "BORN"
        elif "award" in q or "won" in q or "prize" in q:
            relation_kind = "AWARD"
        elif "located" in q or "capital" in q:
            relation_kind = "LOCATION"
        elif "married" in q or "spouse" in q or "wife" in q or "husband" in q:
            relation_kind = "MARRIAGE"

    if not relation_kind:
        return None

    t = text
    t_lower = t.lower()
    q = question.lower()

    # ==== SEMANTIC PRESENCE CHECKS ====
    # Abort extraction if the answer type is not present in the passage

    if relation_kind == "DIRECTOR":
        # Must have "directed" or "director" to extract a director name
        if "directed" not in t_lower and "director" not in t_lower:
            return None

        # P0-4: Unicode-safe DIRECTOR extraction
        # Problem: [A-Z] only matches ASCII uppercase, missing Polish Ż, French É, etc.
        # Solution: Use \w+ to capture words, then validate first char with .isupper()

        # Extract PERSON name after "directed by" (most reliable pattern)
        # Pattern: "directed by <Name>" where Name is 1-4 title-case words
        m = re.search(r"(?i)\bdirected\s+by\s+([\w\u00C0-\u024F]+(?:\s+[\w\u00C0-\u024F]+){0,3})(?:\s*[,\.\(\)]|$|\s+(?:is|was|and|who))", t)
        if m:
            name = m.group(1).strip()
            words = name.split()
            # Validate: 1-4 words, first word starts with uppercase letter, not a common word
            if (1 <= len(words) <= 4 and
                words[0] and words[0][0].isalpha() and words[0][0].isupper() and
                words[0].lower() not in {'the', 'a', 'an', 'this', 'that', 'film', 'movie', 'polish', 'australian'}):
                return name

        # Pattern: "Name directed the film" or "Name, who directed"
        m = re.search(r"(?i)([\w\u00C0-\u024F]+(?:\s+[\w\u00C0-\u024F]+){0,3})\s+(?:,\s*who\s+)?directed\b", t)
        if m:
            name = m.group(1).strip()
            words = name.split()
            # Validate: 1-4 words, first word starts with uppercase letter
            if (1 <= len(words) <= 4 and
                words[0] and words[0][0].isalpha() and words[0][0].isupper() and
                words[0].lower() not in {'the', 'a', 'an', 'this', 'that', 'film', 'movie'}):
                return name

        # Pattern: "director Name" (less common but valid)
        m = re.search(r"(?i)\bdirector\s+([\w\u00C0-\u024F]+(?:\s+[\w\u00C0-\u024F]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            name = m.group(1).strip()
            words = name.split()
            # Validate: 1-4 words, first word starts with uppercase letter
            if (1 <= len(words) <= 4 and
                words[0] and words[0][0].isalpha() and words[0][0].isupper() and
                words[0].lower() not in {'of', 'the', 'a', 'an', 'is', 'was'}):
                return name

    elif relation_kind == "BORN":
        if "where" in q or "place" in q:
            # Birthplace - must have "born" to extract
            if "born" not in t_lower:
                return None
            # P0-4: Unicode-safe birthplace extraction
            # Pattern: "born in Location"
            m = re.search(r"(?i)\bborn\s+in\s+([\w\u00C0-\u024F\s\-]+?)(?:\s*[,\.\(\)]|on\s|$)", t)
            if m:
                loc = m.group(1).strip().rstrip(",")
                # Validate: should be 1-4 words, first word starts with uppercase letter
                words = loc.split()
                if (1 <= len(words) <= 4 and
                    words[0] and words[0][0].isalpha() and words[0][0].isupper()):
                    return loc
        else:
            # Birth date - must have "born" and a year
            if "born" not in t_lower or not re.search(r'\d{4}', t):
                return None
            # P0-4: Unicode-safe birth date extraction
            # Pattern: "born on/in Month Day, Year" or "born on Day Month Year"
            m = re.search(r"(?i)\bborn\s+(?:on\s+)?([a-zA-Z]+\s+\d{1,2},?\s+\d{4})", t)
            if m:
                return m.group(1).strip()
            m = re.search(r"(?i)\bborn\s+(?:on\s+)?(\d{1,2}\s+[a-zA-Z]+\s+\d{4})", t)
            if m:
                return m.group(1).strip()

    elif relation_kind == "AWARD":
        # CRITICAL: Must have award-related words to extract an award
        award_indicators = ['award', 'prize', 'medal', 'oscar', 'emmy', 'grammy', 'trophy', 'won', 'awarded', 'received']
        if not any(ind in t_lower for ind in award_indicators):
            return None  # Don't extract - passage doesn't contain award info

        # P0-4: Unicode-safe award extraction
        # Pattern: "won the X Award/Prize"
        m = re.search(r"(?i)\bwon\s+(?:the\s+)?([\w\u00C0-\u024F\s\-]+?(?:Award|Prize|Medal|Oscar|Emmy|Grammy|Trophy))", t)
        if m:
            award = m.group(1).strip()
            # Validate: first word should start with uppercase
            words = award.split()
            if words and words[0] and words[0][0].isalpha() and words[0][0].isupper():
                return award
        # Pattern: "received the X Award"
        m = re.search(r"(?i)\breceived\s+(?:the\s+)?([\w\u00C0-\u024F\s\-]+?(?:Award|Prize|Medal))", t)
        if m:
            award = m.group(1).strip()
            words = award.split()
            if words and words[0] and words[0][0].isalpha() and words[0][0].isupper():
                return award
        # Pattern: "awarded the X"
        m = re.search(r"(?i)\bawarded\s+(?:the\s+)?([\w\u00C0-\u024F\s\-]+?(?:Award|Prize|Medal))", t)
        if m:
            award = m.group(1).strip()
            words = award.split()
            if words and words[0] and words[0][0].isalpha() and words[0][0].isupper():
                return award

    elif relation_kind == "LOCATION":
        # Must have location-related words
        loc_indicators = ['located', 'capital', 'headquarters', 'based', 'situated']
        if not any(ind in t_lower for ind in loc_indicators):
            return None

        m = re.search(r"(?i)\blocated\s+in\s+([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip().rstrip(",")
        m = re.search(r"(?i)\bcapital\s+(?:city\s+)?(?:is|of)\s+([A-Z][a-zA-ZÀ-ÿ\s\-]+?)(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()
        m = re.search(r"(?i)\bheadquarters\s+(?:is|are)\s+(?:in\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip().rstrip(",")

    elif relation_kind == "MARRIAGE":
        # Must have marriage-related words
        if "married" not in t_lower and "spouse" not in t_lower and "wife" not in t_lower and "husband" not in t_lower:
            return None

        m = re.search(r"(?i)\bmarried\s+(?:to\s+)?([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|in\s|$)", t)
        if m:
            return m.group(1).strip()
        m = re.search(r"(?i)\bspouse\s+(?:is\s+)?([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()

    # No valid extraction possible - return None for controlled abstention
    return None


def regex_extract_answer_from_hop2(
    question: str,
    facets: List[Dict[str, Any]],
    hop2_text: str
) -> Optional[str]:
    """
    DEPRECATED: Use typed_extract_from_winning_passage instead.

    This function extracts from arbitrary hop2_text, which may NOT be
    the passage that certified the RELATION facet. It can produce
    garbage like "of film and TV" from unrelated passages.

    Kept for backwards compatibility but should be replaced.
    """
    if not hop2_text:
        return None

    t = hop2_text
    q = question.lower()

    # Check if we have RELATION facets
    has_relation = any(f.get("facet_type") == "RELATION" for f in facets)

    if has_relation:
        # DIRECTOR: "directed by X" or "X directed"
        if "director" in q or "directed" in q or "direct" in q:
            # Pattern: "directed by Name"
            m = re.search(r"(?i)\bdirected\s+by\s+([A-Z][a-zA-ZÀ-ÿ\s\-\.]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip()
            # Pattern: "director Name"
            m = re.search(r"(?i)\bdirector\s+([A-Z][a-zA-ZÀ-ÿ\s\-\.]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip()
            # Pattern: "film by Name"
            m = re.search(r"(?i)\bfilm\s+by\s+([A-Z][a-zA-ZÀ-ÿ\s\-\.]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip()

        # BORN: "born in X" or "was born on X"
        if "born" in q or "birth" in q:
            # Birthplace
            if "where" in q or "place" in q:
                m = re.search(r"(?i)\bborn\s+in\s+([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|on\s|$)", t)
                if m:
                    return m.group(1).strip().rstrip(",")
            # Birth date
            else:
                m = re.search(r"(?i)\bborn\s+(?:on\s+)?([A-Z]?[a-zA-Z0-9\s\,]+\d{4})", t)
                if m:
                    return m.group(1).strip()

        # AWARD: "won X" or "received X award"
        if "award" in q or "won" in q or "prize" in q:
            m = re.search(r"(?i)\bwon\s+(?:the\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-]+?(?:Award|Prize|Medal|Oscar|Emmy|Grammy))", t)
            if m:
                return m.group(1).strip()
            m = re.search(r"(?i)\breceived\s+(?:the\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-]+?(?:Award|Prize|Medal))", t)
            if m:
                return m.group(1).strip()

        # LOCATION: "located in X" or "capital of X is Y"
        if "located" in q or "capital" in q or "headquarters" in q:
            m = re.search(r"(?i)\blocated\s+in\s+([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip().rstrip(",")
            m = re.search(r"(?i)\bcapital\s+(?:is|of)\s+([A-Z][a-zA-ZÀ-ÿ\s\-]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip()
            m = re.search(r"(?i)\bheadquarters\s+(?:is|are)\s+(?:in\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip().rstrip(",")

        # MARRIAGE: "married X" or "spouse X"
        if "married" in q or "spouse" in q or "wife" in q or "husband" in q:
            m = re.search(r"(?i)\bmarried\s+(?:to\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-\.]+?)(?:\s*[,\.\(\)]|in\s|$)", t)
            if m:
                return m.group(1).strip()

    return None


# ==============================================================================
# CERTIFIED SPAN SELECTION (CSS)
# ==============================================================================
# General, relation-agnostic answer extraction that requires exact substring
# grounding in certified evidence. No per-relation regex patterns.
# ==============================================================================

@dataclass
class CSSResult:
    """Result from Certified Span Selection."""
    abstain: bool
    answer: str
    passage_id: str
    reason: str  # OK, NO_SPAN, SPAN_NOT_VERBATIM, PID_NOT_IN_WINNERS, PARSE_ERROR
    confidence: float = 0.0
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _find_exact_substring(haystack: str, needle: str) -> Optional[Tuple[int, int]]:
    """Find exact substring match (case-sensitive first, then case-insensitive)."""
    if not needle or not haystack:
        return None

    # Try case-sensitive first
    i = haystack.find(needle)
    if i >= 0:
        return (i, i + len(needle))

    # Try case-insensitive
    i = haystack.lower().find(needle.lower())
    if i >= 0:
        return (i, i + len(needle))

    # Normalized unicode fallback (handles dash/quote variants)
    hay_norm = _normalize_text_unicode(haystack)
    needle_norm = _normalize_text_unicode(needle)
    if needle_norm:
        j = hay_norm.find(needle_norm)
        if j >= 0:
            return (j, j + len(needle_norm))

    return None


def _extract_first_json_object(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text")

    # reject code fences before the JSON
    if "```" in stripped[:start]:
        raise ValueError("Code fences found before JSON")

    depth = 0
    end_idx = -1
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break

    if depth != 0 or end_idx == -1:
        raise ValueError("Unbalanced JSON braces in text")

    return stripped[start:end_idx + 1]


def strict_json_call(llm, prompt: str, max_new_tokens: int = 160, temperature: float = 0.0):
    """Run an LLM call that must return exactly one JSON object.

    This helper centralizes the JSON-only contract so downstream callers do not
    attempt their own permissive parsing. It extracts the first balanced JSON
    object and ignores any trailing text, while still rejecting code fences or
    unbalanced braces to avoid ambiguous parses.

    Returns:
        tuple(parsed_json, raw_output)
    """

    guard = "Return ONLY JSON. No extra keys. No commentary."
    wrapped_prompt = f"{guard}\n{prompt.strip()}\n{guard}"
    raw = llm.generate(
        wrapped_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    raw_text = raw.text if hasattr(raw, "text") else str(raw)
    parsed = json.loads(_extract_first_json_object(raw_text))
    return parsed, raw


def _find_fuzzy_substring(haystack: str, needle: str, max_distance: int = 2) -> Optional[Tuple[int, int, str]]:
    """
    Find fuzzy substring match allowing for minor differences.

    Returns (start, end, matched_text) or None.
    Used as fallback when exact match fails due to punctuation/whitespace differences.
    """
    if not needle or not haystack:
        return None

    # Normalize both strings
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
        s = ' '.join(s.split())  # Normalize whitespace
        return s

    needle_norm = normalize(needle)
    if not needle_norm:
        return None

    # Sliding window search
    words = haystack.split()
    needle_words = needle_norm.split()

    if not needle_words:
        return None

    for i in range(len(words) - len(needle_words) + 1):
        window = words[i:i + len(needle_words)]
        window_norm = [normalize(w) for w in window]

        # Check if normalized words match
        if window_norm == needle_words:
            # Find the actual span in the original text
            start_word = ' '.join(words[:i])
            matched_text = ' '.join(window)
            start_pos = len(start_word) + (1 if start_word else 0)
            return (start_pos, start_pos + len(matched_text), matched_text)

    return None


def certified_span_select(
    llm,
    question: str,
    winner_passages: List[Dict[str, Any]],
    max_chars_per_passage: int = 800
) -> CSSResult:
    """
    Certified Span Selection: Extract answer as exact substring from evidence.

    This is relation-agnostic and requires verbatim grounding.

    Args:
        llm: LLM interface with generate() method
        question: The question to answer
        winner_passages: List of certificate-winning passages [{pid, text, title}, ...]
        max_chars_per_passage: Max chars to include per passage (for context limits)

    Returns:
        CSSResult with answer span and verification status
    """
    import json as json_module
    import os

    debug = os.environ.get("TRIDENT_DEBUG_CSS", "0") == "1"

    if not winner_passages:
        return CSSResult(abstain=True, answer="", passage_id="", reason="NO_PASSAGES")

    # Build compact evidence
    evidence_parts = []
    for p in winner_passages:
        pid = p.get("pid", "")
        title = p.get("title", "")
        text = (p.get("text") or "")[:max_chars_per_passage]
        evidence_parts.append(f"[{pid}] {title}\n{text}")

    evidence_text = "\n\n".join(evidence_parts)

    prompt = f"""You must answer ONLY using the Evidence below.
Return exactly ONE JSON object with keys:
  "pid": string, the evidence passage id you copied from
  "answer": string, an EXACT substring copied verbatim from that passage
  "confidence": number between 0 and 1

Rules (hard constraints):
1) Output must be JSON ONLY. Do NOT include code fences, commentary, or extra text.
1b) The first character of your reply must be '{' and you must end immediately after the closing '}'.
2) The answer MUST be copied exactly from the evidence text (verbatim).
3) Copy the shortest complete answer span - usually a name, place, date, or short phrase.
4) If you cannot find an exact answer substring, output: {{"pid": "", "answer": "", "confidence": 0}}

Question: {question}

Evidence:
{evidence_text}

JSON:"""

    try:
        out, raw = strict_json_call(llm, prompt, max_new_tokens=160, temperature=0.0)
        raw_text = raw.text if hasattr(raw, 'text') else str(raw)
    except Exception as e:
        if debug:
            print(f"[CSS] JSON parse failed: {e}")
        return CSSResult(abstain=True, answer="", passage_id="", reason="PARSE_ERROR")

    if debug:
        print(f"[CSS] Raw LLM output: {raw_text[:200]}...")

    prompt_tokens = getattr(raw, "prompt_tokens", 0) or 0
    completion_tokens = getattr(raw, "completion_tokens", 0) or 0
    tokens_used = getattr(raw, "tokens_used", 0) or (prompt_tokens + completion_tokens)

    if not prompt_tokens and hasattr(llm, "compute_token_cost"):
        try:
            prompt_tokens = llm.compute_token_cost(prompt)
            tokens_used = tokens_used or prompt_tokens + completion_tokens
        except Exception:
            prompt_tokens = prompt_tokens

    pid = (out.get("pid") or "").strip()
    answer = (out.get("answer") or "").strip()
    confidence = float(out.get("confidence", 0))

    if debug:
        print(f"[CSS] Parsed: pid={pid}, answer={answer[:50]}..., conf={confidence}")

    # Check for empty response (LLM abstention)
    if not pid or not answer:
        return CSSResult(
            abstain=True,
            answer="",
            passage_id=pid,
            reason="NO_SPAN",
            confidence=confidence,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # Verify passage ID is in winners
    passage_map = {p.get("pid", ""): p for p in winner_passages}
    if pid not in passage_map:
        if debug:
            print(f"[CSS] PID {pid} not in winners: {list(passage_map.keys())}")
        return CSSResult(
            abstain=True,
            answer=answer,
            passage_id=pid,
            reason="PID_NOT_IN_WINNERS",
            confidence=confidence,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # Verify exact grounding
    passage_text = passage_map[pid].get("text", "")

    # Try exact match first
    match = _find_exact_substring(passage_text, answer)
    if match is not None:
        if debug:
            print(f"[CSS] Exact match found at {match}")
        return CSSResult(
            abstain=False,
            answer=answer,
            passage_id=pid,
            reason="OK",
            confidence=confidence,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # Try fuzzy match as fallback
    fuzzy = _find_fuzzy_substring(passage_text, answer)
    if fuzzy is not None:
        start, end, matched_text = fuzzy
        if debug:
            print(f"[CSS] Fuzzy match: '{answer}' -> '{matched_text}'")
        # Return the actual text from passage (properly grounded)
        return CSSResult(
            abstain=False,
            answer=matched_text,
            passage_id=pid,
            reason="OK_FUZZY",
            confidence=confidence,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    if debug:
        print(f"[CSS] No match found for '{answer}' in passage")
    return CSSResult(
        abstain=True,
        answer=answer,
        passage_id=pid,
        reason="SPAN_NOT_VERBATIM",
        confidence=confidence,
        tokens_used=tokens_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ==============================================================================
# CONSTRAINED CANDIDATE SELECTION
# ==============================================================================
# Instead of free-form extraction, build a candidate set and have LLM pick by
# index. This ensures verbatim answers and enables support-based ranking.
# ==============================================================================

@dataclass
class QuestionType:
    """Detected question type for constraining candidates."""
    category: str  # "either_or", "who", "what", "where", "when", "how_many", "yes_no", "other"
    expected_type: str  # "PERSON", "PLACE", "DATE", "NUMBER", "BINARY", "ENTITY", "OTHER"
    options: Optional[List[str]] = None  # For either/or questions


@dataclass
class ConstrainedSelectionResult:
    """Result from constrained candidate selection."""
    answer: str
    candidate_index: int
    confidence: float
    passage_id: str
    candidates: List[str]
    support_scores: List[float]
    reason: str  # OK, NO_PASSAGES, NO_VERIFIED_SPAN, etc.
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def detect_question_type(question: str) -> QuestionType:
    """
    Detect question type to constrain candidate extraction.

    Returns:
        QuestionType with category, expected_type, and options (for either/or)
    """
    q = question.strip()
    q_lower = q.lower()

    # Either/or questions: "X or Y?" pattern
    # Match patterns like "was it A or B", "is this X or Y", "A or B?"
    either_or_match = re.search(
        r'\b(is|was|are|were|did|does|do|can|could|would|should|will)\s+.{1,50}?\s+(\w+(?:\s+\w+){0,3})\s+or\s+(\w+(?:\s+\w+){0,3})\s*\??$',
        q_lower
    )
    if either_or_match:
        opt1 = either_or_match.group(2).strip()
        opt2 = either_or_match.group(3).strip("?. ")
        return QuestionType(
            category="either_or",
            expected_type="BINARY_CHOICE",
            options=[opt1, opt2]
        )

    # Simple "A or B?" at end
    simple_or = re.search(r'(\w+(?:\s+\w+){0,4})\s+or\s+(\w+(?:\s+\w+){0,4})\s*\??$', q_lower)
    if simple_or:
        opt1 = simple_or.group(1).strip()
        opt2 = simple_or.group(2).strip("?. ")
        # Make sure these look like answer options (not "this or that")
        if opt1.lower() not in {'this', 'that', 'it', 'he', 'she', 'they'}:
            return QuestionType(
                category="either_or",
                expected_type="BINARY_CHOICE",
                options=[opt1, opt2]
            )

    # Yes/No questions
    if q_lower.startswith(('is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ',
                           'can ', 'could ', 'should ', 'would ', 'has ', 'have ', 'had ')):
        return QuestionType(category="yes_no", expected_type="BINARY", options=["yes", "no"])

    # Who questions -> PERSON
    if q_lower.startswith('who ') or ' who ' in q_lower[:30]:
        return QuestionType(category="who", expected_type="PERSON")

    # Where questions -> PLACE
    if q_lower.startswith('where ') or ' where ' in q_lower[:30]:
        return QuestionType(category="where", expected_type="PLACE")

    # When questions -> DATE
    if q_lower.startswith('when ') or ' when ' in q_lower[:30]:
        return QuestionType(category="when", expected_type="DATE")

    # How many/much -> NUMBER
    if q_lower.startswith('how many ') or q_lower.startswith('how much '):
        return QuestionType(category="how_many", expected_type="NUMBER")

    # What questions -> ENTITY (generic)
    if q_lower.startswith('what ') or ' what ' in q_lower[:30]:
        # Check for more specific "what [type]" patterns
        what_person = re.match(r'what\s+(person|actor|actress|director|author|singer|player)', q_lower)
        if what_person:
            return QuestionType(category="what", expected_type="PERSON")
        what_place = re.match(r'what\s+(city|country|place|location|state|region)', q_lower)
        if what_place:
            return QuestionType(category="what", expected_type="PLACE")
        what_date = re.match(r'what\s+(year|date|time|day|month)', q_lower)
        if what_date:
            return QuestionType(category="what", expected_type="DATE")
        return QuestionType(category="what", expected_type="ENTITY")

    # Default: other
    return QuestionType(category="other", expected_type="OTHER")


def _normalize_for_dedup(s: str) -> str:
    """
    E2 FIX: Normalize string for substring deduplication.

    Handles punctuation variations so "D'Arcy Coulson" and "Darcy Coulson" compare correctly.
    """
    import re
    # Replace apostrophes, quotes, hyphens, periods with space
    s = re.sub(r"[\s''\.\-]+", " ", s.strip().lower())
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_candidates(
    winner_passages: List[Dict[str, Any]],
    question_type: QuestionType,
    max_candidates: int = 10
) -> List[Tuple[str, float, str]]:
    """
    Extract candidate answers from winner passages based on question type.

    Returns:
        List of (candidate_text, support_score, source_pid) tuples,
        sorted by support_score descending.
    """
    candidates: Dict[str, Tuple[float, str]] = {}  # text -> (score, pid)

    # For either/or questions, use the provided options
    if question_type.category == "either_or" and question_type.options:
        for opt in question_type.options:
            # Find which passage(s) support this option
            support = 0.0
            source_pid = ""
            for p in winner_passages:
                text_lower = (p.get("text") or "").lower()
                if opt.lower() in text_lower:
                    support += 1.0
                    if not source_pid:
                        source_pid = p.get("pid", "")
            if source_pid:
                candidates[opt] = (support, source_pid)
        # Return sorted by support
        return sorted(
            [(text, score, pid) for text, (score, pid) in candidates.items() if pid],
            key=lambda x: x[1],
            reverse=True
        )[:max_candidates]

    # For yes/no questions
    if question_type.category == "yes_no":
        # Count evidence for yes vs no
        yes_support = 0.0
        no_support = 0.0
        yes_pid = ""
        no_pid = ""

        for p in winner_passages:
            text = (p.get("text") or "").lower()
            pid = p.get("pid", "")
            # Look for affirmative/negative language
            if any(w in text for w in ['is a', 'was a', 'are the', 'were the', 'did ', 'does ', 'has ', 'had ']):
                yes_support += 0.5
                if not yes_pid:
                    yes_pid = pid
            if any(w in text for w in ['not ', "n't ", 'never ', 'no ', 'none ']):
                no_support += 0.5
                if not no_pid:
                    no_pid = pid

        if not yes_pid and winner_passages:
            yes_pid = winner_passages[0].get("pid", "")
        if not no_pid and winner_passages:
            no_pid = winner_passages[0].get("pid", "")

        return [
            ("yes", yes_support, yes_pid),
            ("no", no_support, no_pid)
        ]

    # For PERSON questions
    if question_type.expected_type == "PERSON":
        # CRITICAL FIX: Span-based extraction with full Unicode support
        # Old pattern missed Polish Ż (U+017B) and other extended Unicode
        # New approach: Extract until punctuation, handles all languages

        # Also include passage title as high-value candidate
        for p in winner_passages:
            title = p.get("title", "")
            text = p.get("text") or ""
            pid = p.get("pid", "")

            # Add title as a candidate (often the entity being discussed)
            if title and _looks_like_person(title):
                # High support score for title
                support = len(winner_passages) + 1.0
                candidates[title] = (support, pid)

            # Extract capitalized spans from text
            # Pattern: capital letter followed by non-punctuation until delimiter
            # Captures: "Xawery Żuławski", "J.K. Rowling", "D'Arcy", etc.
            for match in re.finditer(r'\b([A-Z][^.,;(\n]*?(?:\s+[A-Z][^.,;(\n]*?){0,5})\b', text):
                name = match.group(1).strip()
                # Clean trailing whitespace and partial punctuation
                name = re.sub(r'[\s,\.\(\)]+$', '', name)

                # Validate: looks like a person name
                if name and _looks_like_person(name):
                    # Compute support: count occurrences across all passages
                    support = sum(
                        1 for pp in winner_passages
                        if name.lower() in (pp.get("text") or "").lower()
                    )
                    # MULTI-TOKEN BONUS: Prefer longer spans (more specific)
                    token_count = len(name.split())
                    length_bonus = 0.5 * (token_count - 1)
                    adjusted_support = support + length_bonus
                    if name not in candidates or adjusted_support > candidates[name][0]:
                        candidates[name] = (adjusted_support, pid)

        # SUBSTRING DEDUPLICATION: Remove single-token candidates that are
        # substrings of higher-scoring multi-token candidates
        # But DON'T remove all single-token candidates (Plato, Madonna, Oregon exist!)
        if len(candidates) > 1:
            to_remove = set()
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
            for i, (name1, (score1, _)) in enumerate(sorted_cands):
                name1_norm = _normalize_for_dedup(name1)
                for name2, (score2, _) in sorted_cands[i+1:]:
                    name2_norm = _normalize_for_dedup(name2)
                    # Only remove if:
                    # 1. Shorter name is substring of longer name
                    # 2. Longer name has significantly better support
                    if (len(name2_norm) < len(name1_norm) and
                        name2_norm in name1_norm and
                        score1 > score2 * 1.2):  # 20% margin
                        to_remove.add(name2)
            for name in to_remove:
                if name in candidates:
                    del candidates[name]

    # For PLACE questions
    elif question_type.expected_type == "PLACE":
        # Extract capitalized spans that look like places
        place_pattern = re.compile(
            r'\b([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})\b'
        )
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for match in place_pattern.finditer(text):
                place = match.group(1).strip()
                # Filter out obvious non-places
                if len(place) >= 2 and place.lower() not in {
                    'the', 'a', 'an', 'he', 'she', 'they', 'it', 'his', 'her',
                    'their', 'was', 'were', 'is', 'are', 'has', 'had', 'have'
                }:
                    support = sum(
                        1 for pp in winner_passages
                        if place.lower() in (pp.get("text") or "").lower()
                    )
                    if place not in candidates or support > candidates[place][0]:
                        candidates[place] = (support, pid)

    # For DATE questions
    elif question_type.expected_type == "DATE":
        # Extract date patterns
        date_patterns = [
            # Year only: 1990, 2024
            re.compile(r'\b(1[0-9]{3}|20[0-2][0-9])\b'),
            # Month Day, Year: January 1, 2020
            re.compile(r'\b([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b'),
            # Day Month Year: 1 January 2020
            re.compile(r'\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b'),
            # Month Year: January 2020
            re.compile(r'\b([A-Z][a-z]+\s+\d{4})\b'),
        ]
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for pattern in date_patterns:
                for match in pattern.finditer(text):
                    date_str = match.group(1).strip()
                    support = sum(
                        1 for pp in winner_passages
                        if date_str in (pp.get("text") or "")
                    )
                    if date_str not in candidates or support > candidates[date_str][0]:
                        candidates[date_str] = (support, pid)

    # For NUMBER questions
    elif question_type.expected_type == "NUMBER":
        number_pattern = re.compile(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b')
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for match in number_pattern.finditer(text):
                num = match.group(1).strip()
                support = sum(
                    1 for pp in winner_passages
                    if num in (pp.get("text") or "")
                )
                if num not in candidates or support > candidates[num][0]:
                    candidates[num] = (support, pid)

    # For ENTITY or OTHER: extract all capitalized spans
    else:
        # E1 FIX: Punctuation-aware pattern for apostrophes, initials, etc.
        # Handles: D'Arcy, O'Connor, J.K. Rowling, C.S. Lewis
        # Increased max tokens from 4 to 6 for longer entity names
        entity_pattern = re.compile(
            r"\b([A-Z][a-zA-ZÀ-ÿ''\.­-]+(?:\s+[A-Z][a-zA-ZÀ-ÿ''\.­-]+){0,6})\b"
        )
        # Common stop-words to filter out
        stop_words = {'The', 'A', 'An', 'He', 'She', 'They', 'It', 'His', 'Her',
                      'Their', 'Was', 'Were', 'Is', 'Are', 'Has', 'Had', 'Have',
                      'This', 'That', 'These', 'Those', 'In', 'On', 'At', 'By'}
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for match in entity_pattern.finditer(text):
                entity = match.group(1).strip()
                # Skip single-token stop-words
                if entity in stop_words:
                    continue
                if len(entity) >= 2:
                    support = sum(
                        1 for pp in winner_passages
                        if entity.lower() in (pp.get("text") or "").lower()
                    )
                    # MULTI-TOKEN BONUS: Prefer longer spans (more specific)
                    token_count = len(entity.split())
                    length_bonus = 0.5 * (token_count - 1)
                    adjusted_support = support + length_bonus
                    if entity not in candidates or adjusted_support > candidates[entity][0]:
                        candidates[entity] = (adjusted_support, pid)

        # SUBSTRING DEDUPLICATION: Remove single-token candidates that are
        # substrings of higher-scoring multi-token candidates
        # E2 FIX: Use normalization so punctuation variants compare correctly
        if len(candidates) > 1:
            to_remove = set()
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
            for i, (name1, _) in enumerate(sorted_cands):
                name1_norm = _normalize_for_dedup(name1)
                for name2, _ in sorted_cands[i+1:]:
                    name2_norm = _normalize_for_dedup(name2)
                    # If shorter name is a substring of longer name, mark for removal
                    if len(name2_norm) < len(name1_norm) and name2_norm in name1_norm:
                        to_remove.add(name2)
            for name in to_remove:
                if name in candidates:
                    del candidates[name]

    # Sort by support score and return top candidates
    sorted_candidates = sorted(
        [
            (text, score, pid)
            for text, (score, pid) in candidates.items()
            if pid
        ],
        key=lambda x: x[1],
        reverse=True
    )

    # CRITICAL FIX (Change 7): REMOVED single-token rejection logic
    # Single-token entities ARE valid: "Plato", "Madonna", "Oregon", "Cher", "Prince"
    # The ranking/support scoring handles quality filtering, not hard rejection
    # If all candidates are single-token, that's the best we can do from the evidence

    return sorted_candidates[:max_candidates]


def compute_support_score(
    candidate: str,
    passages: List[Dict[str, Any]]
) -> float:
    """
    Compute deterministic support score for a candidate.

    Score = count of passages mentioning candidate, weighted by passage rank.
    """
    if not candidate or not passages:
        return 0.0

    candidate_lower = candidate.lower()
    score = 0.0

    for i, p in enumerate(passages):
        text_lower = (p.get("text") or "").lower()
        title_lower = (p.get("title") or "").lower()

        # Check if candidate appears
        if candidate_lower in text_lower or candidate_lower in title_lower:
            # Weight by rank (earlier passages = higher weight)
            weight = 1.0 / (1 + i * 0.1)
            score += weight

    return score


def get_answer_facet_passages(
    question: str,
    certificates: List[Dict[str, Any]],
    all_passages: List[Dict[str, Any]],
    facets: Optional[List[Dict[str, Any]]] = None,
    max_answer_facets: int = 3,
    max_answer_passages: int = 5,
) -> Tuple[List[Dict[str, Any]], List[str], str]:
    """
    Select passages corresponding to the *answer-determining* facets.

    Certified-only, fail-closed properties:
    - Only passages whose PID appears in certificates are eligible.
    - For each facet, only the best (lowest p-value) certificate is used to fetch evidence.
    - LLM (if used) may only choose from certified facet IDs; outputs are intersected.
    """
    import json
    import os

    debug = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"

    if not certificates:
        return [], [], "NO_CERTIFICATES"

    # -----------------------------
    # 1) Certified-only PID scope
    # -----------------------------
    certified_pids = _cert_pid(certificates)

    # Build passage lookup from certified-only pool
    pid_to_passage = {
        p.get("pid"): p
        for p in all_passages
        if p.get("pid") in certified_pids
    }

    # -----------------------------
    # 2) Best cert per facet (lowest p-value)
    # -----------------------------
    facet_to_certs: Dict[str, List[Dict[str, Any]]] = {}
    for cert in certificates:
        fid = cert.get("facet_id") or ""
        pid = cert.get("passage_id") or cert.get("pid") or cert.get("passage_pid") or cert.get("winning_pid") or ""
        if fid and pid and pid in pid_to_passage:
            facet_to_certs.setdefault(fid, []).append(cert)

    if not facet_to_certs:
        # If schema drift: we have certificates but couldn't map to passages cleanly
        return [], [], "NO_CERT_MAPPABLE_PASSAGES"

    best_cert_by_fid: Dict[str, Dict[str, Any]] = {}
    for fid, certs in facet_to_certs.items():
        best_cert_by_fid[fid] = min(certs, key=lambda c: float(c.get("p_value", 1.0)))

    certified_facet_ids = list(best_cert_by_fid.keys())

    # -----------------------------
    # 3) Choose answer facets deterministically
    # -----------------------------
    selected_facet_ids = [
        fid for fid, _ in sorted(
            ((fid, cert.get("p_value", 1.0)) for fid, cert in best_cert_by_fid.items()),
            key=lambda x: float(x[1])
        )
    ][:max_answer_facets]
    selection_reason = "TOP_PVALUE"

    if not selected_facet_ids:
        selected_facet_ids = certified_facet_ids[:max_answer_facets]
        selection_reason = f"HEURISTIC:{','.join(selected_facet_ids)}"

    # -----------------------------
    # 4) Fetch winning passages from best certs (strict provenance)
    # -----------------------------
    target_certs = [best_cert_by_fid[fid] for fid in selected_facet_ids if fid in best_cert_by_fid]
    target_certs.sort(key=lambda c: float(c.get("p_value", 1.0)))

    answer_passages: List[Dict[str, Any]] = []
    seen = set()
    for cert in target_certs:
        pid = cert.get("passage_id") or cert.get("pid") or cert.get("passage_pid") or cert.get("winning_pid") or ""
        if not pid or pid in seen:
            continue
        if pid in pid_to_passage:
            seen.add(pid)
            answer_passages.append(pid_to_passage[pid])
        if len(answer_passages) >= max_answer_passages:
            break

    if debug:
        apids = [p.get("pid") for p in answer_passages]
        print(f"[ANSWER-FACET] reason={selection_reason} facets={selected_facet_ids} pids={apids}")

    return answer_passages, selected_facet_ids, selection_reason

def constrained_span_select(
    llm,
    question: str,
    winner_passages: List[Dict[str, Any]],
    max_candidates: int = 10,
    max_chars_per_passage: int = 600,
    certificates: Optional[List[Dict[str, Any]]] = None,
) -> ConstrainedSelectionResult:
    """
    Primary: Frozen span extraction (pid + offsets + verbatim check) from winner_passages.
    Secondary: candidate-index selection (legacy) but still certified-only if certificates provided.

    Fail-closed: returns empty answer if nothing can be machine-verified.
    """
    import json
    import os
    import re

    debug = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"

    if not winner_passages:
        return ConstrainedSelectionResult(
            answer="",
            candidate_index=-1,
            confidence=0.0,
            passage_id="",
            candidates=[],
            support_scores=[],
            reason="NO_PASSAGES"
        )

    if certificates:
        certified_pids = _cert_pid(certificates)
        winner_passages = [p for p in winner_passages if p.get("pid") in certified_pids]
        if not winner_passages:
            return ConstrainedSelectionResult(
                answer="",
                candidate_index=-1,
                confidence=0.0,
                passage_id="",
                candidates=[],
                support_scores=[],
                reason="NO_CERTIFIED_PASSAGES"
            )

    # -----------------------------
    # Frozen span extraction prompt
    # -----------------------------
    snippet_map: Dict[str, str] = {}
    evidence_parts = []
    for p in winner_passages:
        pid = p.get("pid")
        if not pid:
            continue
        snip = (p.get("text") or "")[:max_chars_per_passage]
        snippet_map[pid] = snip
        evidence_parts.append(f"[{pid}] {snip}")

    evidence_text = "\n\n".join(evidence_parts)

    span_prompt = f"""Extract the answer as an EXACT substring from the evidence.

Question: {question}

Evidence:
{evidence_text}

Rules:
1. The answer_span must be copied VERBATIM from the evidence.
2. Provide start_char and end_char offsets relative to the snippet shown for that pid.
3. Return JSON only:
{{ "pid": "...", "start_char": 0, "end_char": 0, "answer_span": "...", "confidence": 0.0 }}
4. If not found, return {{ "answer_span": null }}

JSON:"""

    def _norm_ws(s: str) -> str:
        return " ".join((s or "").split())

    def _normalize_pid(pid: str) -> str:
        """Normalize PID format: [context_4] -> context_4"""
        if not pid:
            return pid
        # Strip brackets and whitespace
        return pid.strip().strip("[]")

    try:
        resp = llm.generate(span_prompt, temperature=0.0)
        raw = resp.text if hasattr(resp, "text") else str(resp)
        # Use robust JSON extraction to handle LLM commentary/fences
        json_str = _extract_first_json_object(raw)
        data = json.loads(json_str)

        span = data.get("answer_span")
        pid_raw = data.get("pid")
        start_char = data.get("start_char")
        end_char = data.get("end_char")
        conf = float(data.get("confidence", 0.0))

        # Normalize PID to handle [context_4] vs context_4 inconsistencies
        pid = _normalize_pid(pid_raw) if pid_raw else None

        # Also normalize snippet_map keys for comparison
        snippet_map_normalized = {_normalize_pid(k): v for k, v in snippet_map.items()}

        if span and pid in snippet_map_normalized and isinstance(start_char, int) and isinstance(end_char, int):
            snippet = snippet_map_normalized[pid]

            match = False
            if 0 <= start_char < end_char <= len(snippet):
                sliced = snippet[start_char:end_char]
                if _norm_ws(sliced) == _norm_ws(span):
                    match = True

            # Deterministic repair: exact find in original snippet if offsets wrong
            if not match:
                idx = snippet.find(span)
                if idx != -1:
                    start_char, end_char = idx, idx + len(span)
                    if _norm_ws(snippet[start_char:end_char]) == _norm_ws(span):
                        match = True

            if match:
                # Type sanity checks (lightweight)
                q_type = detect_question_type(question)
                ok = True
                if getattr(q_type, "expected_type", None) == "NUMBER":
                    ok = bool(re.search(r"\d", span)) and len(span) <= 50
                elif getattr(q_type, "expected_type", None) == "DATE":
                    ok = bool(re.search(r"\d{4}", span) or re.search(r"\d", span))
                elif getattr(q_type, "category", None) == "yes_no":
                    ok = _norm_ws(span).lower() in {"yes", "no"}

                if ok:
                    if debug:
                        print(f"[FROZEN-SPAN] pid={pid} span='{span}' [{start_char}:{end_char}] conf={conf:.2f}")
                    return ConstrainedSelectionResult(
                        answer=span,
                        candidate_index=0,
                        confidence=conf,
                        passage_id=pid,
                        candidates=[span],
                        support_scores=[1.0],
                        reason="OK_FROZEN_SPAN"
                    )
    except Exception as ex:
        if debug:
            print(f"[FROZEN-SPAN] Failed: {ex}")

    return ConstrainedSelectionResult(
        answer="",
        candidate_index=-1,
        confidence=0.0,
        passage_id="",
        candidates=[],
        support_scores=[],
        reason="NO_VERIFIED_SPAN"
    )

def bind_entity_via_css(
    llm,
    inner_question: str,
    hop1_passages: List[Dict[str, Any]],
    max_chars: int = 600,
    min_confidence: float = 0.55,
    certificates: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Canonical entity binder: deterministic candidate generation -> LLM index choice.

    Certified-only option:
    - If certificates provided, restrict hop1_passages to certified PIDs.
    Fail-closed:
    - If parsing fails, index out of bounds, or confidence < threshold => None.
    """
    import json
    import os
    from .nli_scorer import _normalize_text_unicode

    debug = os.environ.get("TRIDENT_DEBUG_CSS", "0") == "1"

    if not hop1_passages:
        return None

    if certificates:
        certified_pids = _cert_pid(certificates)
        hop1_passages = [p for p in hop1_passages if p.get("pid") in certified_pids]
        if not hop1_passages:
            return None

    # Candidate enumeration via existing extractor (still deterministic downstream)
    q_type = detect_question_type(inner_question)
    raw_candidates = extract_candidates(hop1_passages, q_type, max_candidates=20)

    # Build normalized -> (display_text, source_pid)
    cand_map: Dict[str, Tuple[str, str]] = {}
    for text, score, pid in raw_candidates:
        if not text or not pid:
            continue
        toks = text.split()
        ok = False
        if len(toks) >= 2:
            ok = True
        elif len(toks) == 1:
            t = toks[0]
            if len(t) >= 4 and t.lower() not in {"this", "that", "film", "movie", "city"}:
                ok = True
        if not ok:
            continue
        norm = _normalize_text_unicode(text)
        if norm and norm not in cand_map:
            cand_map[norm] = (text, pid)

    cand_items = list(cand_map.values())
    # Deterministic ordering: longer (more specific) first, then lexicographic, then pid
    cand_items.sort(key=lambda x: (-(len(x[0].split())), x[0].lower(), x[1]))
    candidate_list = cand_items[:8]  # (text, pid)

    if not candidate_list:
        if debug:
            print("[CSS-BIND] No candidates after filtering.")
        return None

    # Build evidence snippets only from relevant PIDs (for provenance)
    relevant_pids = {pid for _, pid in candidate_list}
    snippet_map: Dict[str, str] = {}
    evidence_parts = []
    for p in hop1_passages:
        pid = p.get("pid")
        if pid in relevant_pids:
            snip = (p.get("text") or "")[:max_chars]
            snippet_map[pid] = snip
            evidence_parts.append(f"[{pid}] {snip}")

    candidates_formatted = "\n".join([f"{i}: {c[0]}" for i, c in enumerate(candidate_list)])
    evidence_text = "\n".join(evidence_parts)

    prompt = "\n".join(
        (
            "Identify the entity that answers the question.",
            "Select ONE candidate from the list by index.",
            "",
            f"Question: {inner_question}",
            "",
            "Candidates:",
            candidates_formatted,
            "",
            "Evidence:",
            evidence_text,
            "",
            "Rules:",
            '1. Return JSON only: { "index": int, "confidence": float }',
            '2. If none match, return { "index": -1 }',
            "",
            "JSON:",
        )
    )

    try:
        resp = llm.generate(prompt, temperature=0.0)
        raw = resp.text if hasattr(resp, "text") else str(resp)
        # --- robust JSON extraction (fail-closed) ---
        def _first_json_obj(text: str) -> Optional[Dict[str, Any]]:
            """Parse the *first* JSON object from model output.
            Accepts extra commentary / code fences / multiple JSON blocks."""
            # Strip code fences (best-effort)
            cleaned = re.sub(r"```[\s\S]*?```", " ", text).strip()
            s0 = cleaned.find("{")
            if s0 == -1:
                return None
            try:
                dec = json.JSONDecoder()
                obj, _ = dec.raw_decode(cleaned[s0:])
                return obj if isinstance(obj, dict) else None
            except Exception:
                # Fallback: first non-greedy {...}
                m = re.search(r"\{[\s\S]*?\}", cleaned)
                if not m:
                    return None
                try:
                    obj = json.loads(m.group(0))
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None

        data = _first_json_obj(raw)
        if not data:
            if debug:
                print(f"[CSS-BIND] Failed: could not parse JSON from: {raw[:120]!r}")
            return None

        idx = int(data.get("index", -1)) if "index" in data and str(data.get("index")).strip() != "" else -1
        conf = float(data.get("confidence", 0.0) or 0.0)

        # Accept alternate schema: {"pid": "...", "answer": "..."} (some models do this).
        if idx < 0:
            pid_raw = str(data.get("pid", "") or "").strip().strip("[]")
            ans_raw = str(data.get("answer", "") or "").strip()
            if pid_raw:
                for j, (_txt, _pid) in enumerate(candidate_list):
                    if (_pid or "") == pid_raw:
                        idx = j
                        break
            if idx < 0 and ans_raw:
                norm_ans = _normalize_text_unicode(ans_raw)
                for j, (_txt, _pid) in enumerate(candidate_list):
                    if _normalize_text_unicode(_txt) == norm_ans:
                        idx = j
                        break

        if conf < min_confidence:
            return None
        if not (0 <= idx < len(candidate_list)):
            return None

        chosen_text, source_pid = candidate_list[idx]
        snippet = snippet_map.get(source_pid, "")

        # Provenance check (normalization-consistent containment)
        if _normalize_text_unicode(chosen_text) in _normalize_text_unicode(snippet):
            if debug:
                print(f"[CSS-BIND] idx={idx} conf={conf:.2f} -> {chosen_text} (pid={source_pid})")
            return chosen_text

    except Exception as ex:
        if debug:
            print(f"[CSS-BIND] Failed: {ex}")

    return None


def extract_object_from_certified_passage(
    passage_text: str,
    relation_kind: str
) -> Optional[str]:
    """
    FIX 2: Extract the object (answer) from a CERTIFIED relation passage.

    This is NOT heuristic - it extracts from passages that have already
    been certified by Safe-Cover. Uses simple, reliable patterns.

    Args:
        passage_text: The certified passage text
        relation_kind: The relation type (DIRECTOR, MOTHER, FATHER, etc.)

    Returns:
        The extracted entity (e.g., "Xawery Żuławski"), or None if not found
    """
    if not passage_text:
        return None

    import os
    debug = os.environ.get("TRIDENT_DEBUG_CSS", "0") == "1"

    t = passage_text
    t_lower = t.lower()

    # CRITICAL FIX (Change 8): Normalize relation_kind to handle variants
    # "DIRECTOR" vs "rel_director_hop1" vs "director"
    if relation_kind:
        relation_kind = relation_kind.upper().replace("REL_", "").replace("_HOP1", "").replace("_HOP2", "")

    # DIRECTOR: "directed by X" or "film directed by X"
    # P0-2 FIX: Run on raw text, fully case-insensitive, Unicode-safe patterns
    if relation_kind == "DIRECTOR":
        if "directed" not in t_lower and "director" not in t_lower:
            if debug:
                print(f"[CERTIFIED-EXTRACT] DIRECTOR: No 'directed'/'director' in passage")
            return None

        # P0-2 FIX: Removed [A-Z] gating - now matches regardless of capitalization
        # Uses re.IGNORECASE and captures any non-delimiter characters
        # This fixes failures on "directed by xawery żuławski" (lowercase) or mid-sentence

        # Pattern 1: "directed by Name" (most common)
        m = re.search(r"\bdirected\s+by\s+([^\n.,;()]+)", t, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Clean up trailing punctuation/whitespace
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] DIRECTOR: '{name}' from 'directed by' pattern")
                return name

        # Pattern 2: "film directed by Name" (Wikipedia common)
        m = re.search(r"\bfilm\s+directed\s+by\s+([^\n.,;()]+)", t, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] DIRECTOR: '{name}' from 'film directed by' pattern")
                return name

        # Pattern 3: "is a ... film directed by Name" (P0-2: explicit pattern for Wikipedia)
        # Handles "is a 1985 Polish horror film directed by Xawery Żuławski"
        m = re.search(r"is\s+a\s+[^.,;()]*?film\s+directed\s+by\s+([^\n.,;()]+)", t, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] DIRECTOR: '{name}' from 'is a ... film directed by' pattern")
                return name

        # Pattern 4: "director Name" (e.g., "director Xawery Żuławski")
        m = re.search(r"\bdirector\s+([^\n.,;()]+)", t, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] DIRECTOR: '{name}' from 'director X' pattern")
                return name

        # Pattern 5: "directed and produced by Name" (P1-1: Wikipedia common)
        # Handles "directed and produced by Xawery Żuławski"
        m = re.search(r"\bdirected\s+and\s+produced\s+by\s+([^\n.,;()]+)", t, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] DIRECTOR: '{name}' from 'directed and produced by' pattern")
                return name

        if debug:
            print(f"[CERTIFIED-EXTRACT] DIRECTOR: No pattern matched (passage has 'directed'/'director' but extraction failed)")
            print(f"[CERTIFIED-EXTRACT] DIRECTOR: Passage text (first 200 chars): {t[:200]}")

    # MOTHER: "mother of X" or "X is the son/daughter of"
    elif relation_kind in ["MOTHER", "PARENT"]:
        # Pattern 1: Direct "mother of X" or "mother X"
        m = re.search(r"(?i)\bmother\s+(?:of\s+)?([A-Z][^.,;(\n]+)", t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] MOTHER: '{name}' from 'mother of X' pattern")
                return name

        # Pattern 2: "X is the son/daughter of Y and Z" → extract female parent
        # CRITICAL: Handle "son of actress Y and director Z" → return Y
        m = re.search(r"(?:son|daughter|child)\s+of\s+(?:actress|actor)?\s*([A-Z][^.,;(\n]+?)\s+and\s+(?:actress|actor|director)?\s*([A-Z][^.,;(\n]+)", t)
        if m:
            parent1 = m.group(1).strip()
            parent2 = m.group(2).strip()
            parent1 = re.sub(r'[\s,\.\(\)]+$', '', parent1)
            parent2 = re.sub(r'[\s,\.\(\)]+$', '', parent2)

            # Look for gender/role indicators to identify mother
            parent1_context = t[max(0, m.start(1)-20):m.end(1)]
            parent2_context = t[max(0, m.start(2)-20):m.end(2)]

            # Actress/female indicators suggest mother
            if any(kw in parent1_context.lower() for kw in ['actress', 'mother', 'wife']):
                if _looks_like_person(parent1):
                    if debug:
                        print(f"[CERTIFIED-EXTRACT] MOTHER: '{parent1}' from 'son of actress X and Y' pattern")
                    return parent1
            elif any(kw in parent2_context.lower() for kw in ['actress', 'mother', 'wife']):
                if _looks_like_person(parent2):
                    if debug:
                        print(f"[CERTIFIED-EXTRACT] MOTHER: '{parent2}' from 'son of X and actress Y' pattern")
                    return parent2
            else:
                # If no gender indicator, return first parent (convention)
                if _looks_like_person(parent1):
                    if debug:
                        print(f"[CERTIFIED-EXTRACT] MOTHER: '{parent1}' from 'son of X and Y' pattern (first parent)")
                    return parent1

        # Pattern 3: "X is the son/daughter/child of Y" → get Y (object)
        m = re.search(r"(?:is|was)\s+(?:the\s+)?(?:son|daughter|child)\s+of\s+(?:actress\s+)?([A-Z][^.,;(\n]+)", t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            # Stop at "and" to avoid capturing "Y and Z"
            name = re.sub(r'\s+and\s+.*$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] MOTHER: '{name}' from 'is the son/daughter of X' pattern")
                return name

    # FATHER: "father of X" or "X is the son/daughter of"
    elif relation_kind == "FATHER":
        # Pattern 1: Direct "father of X" or "father X"
        m = re.search(r"(?i)\bfather\s+(?:of\s+)?([A-Z][^.,;(\n]+)", t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] FATHER: '{name}' from 'father of X' pattern")
                return name

        # Pattern 2: "X is the son/daughter of Y and Z" → extract male parent
        # CRITICAL: Handle "son of actress Y and director Z" → return Z
        m = re.search(r"(?:son|daughter|child)\s+of\s+(?:actress|actor)?\s*([A-Z][^.,;(\n]+?)\s+and\s+(?:actress|actor|director)?\s*([A-Z][^.,;(\n]+)", t)
        if m:
            parent1 = m.group(1).strip()
            parent2 = m.group(2).strip()
            parent1 = re.sub(r'[\s,\.\(\)]+$', '', parent1)
            parent2 = re.sub(r'[\s,\.\(\)]+$', '', parent2)

            # Look for gender/role indicators to identify father
            parent1_context = t[max(0, m.start(1)-20):m.end(1)]
            parent2_context = t[max(0, m.start(2)-20):m.end(2)]

            # Actor/director/male indicators suggest father
            if any(kw in parent1_context.lower() for kw in ['actor', 'director', 'father']):
                # Exclude 'actress' which contains 'actor'
                if 'actress' not in parent1_context.lower():
                    if _looks_like_person(parent1):
                        if debug:
                            print(f"[CERTIFIED-EXTRACT] FATHER: '{parent1}' from 'son of director X and Y' pattern")
                        return parent1
            elif any(kw in parent2_context.lower() for kw in ['actor', 'director', 'father']):
                if 'actress' not in parent2_context.lower():
                    if _looks_like_person(parent2):
                        if debug:
                            print(f"[CERTIFIED-EXTRACT] FATHER: '{parent2}' from 'son of X and director Y' pattern")
                        return parent2
            else:
                # If no gender indicator, return second parent (convention: mother first, father second)
                if _looks_like_person(parent2):
                    if debug:
                        print(f"[CERTIFIED-EXTRACT] FATHER: '{parent2}' from 'son of X and Y' pattern (second parent)")
                    return parent2

        # Pattern 3: "X is the son/daughter/child of Y" → get Y (object)
        m = re.search(r"(?:is|was)\s+(?:the\s+)?(?:son|daughter|child)\s+of\s+(?:actor|director)?\s*([A-Z][^.,;(\n]+)", t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            # Stop at "and" to avoid capturing "Y and Z"
            name = re.sub(r'\s+and\s+.*$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] FATHER: '{name}' from 'is the son/daughter of X' pattern")
                return name

    # SPOUSE: "married to X" or "spouse X"
    elif relation_kind == "SPOUSE":
        m = re.search(r"(?i)\bmarried\s+(?:to\s+)?([A-Z][^.,;(\n]+)", t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] SPOUSE: '{name}' from 'married to X' pattern")
                return name

    # AUTHOR: "written by X" or "author X"
    elif relation_kind == "AUTHOR":
        m = re.search(r"(?i)\bwritten\s+by\s+([A-Z][^.,;(\n]+)", t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] AUTHOR: '{name}' from 'written by X' pattern")
                return name

        m = re.search(r"(?i)\bauthor\s+([A-Z][^.,;(\n]+)", t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] AUTHOR: '{name}' from 'author X' pattern")
                return name

    # BIRTHPLACE: "born in X" or "birthplace X"
    elif relation_kind == "BIRTHPLACE":
        # Pattern: "born in Location" or "born at Location"
        m = re.search(r"(?i)\bborn\s+(?:in|at)\s+([A-Z][^.,;(\n]+)", t)
        if m:
            place = m.group(1).strip()
            place = re.sub(r'[\s,\.\(\)]+$', '', place)
            if debug:
                print(f"[CERTIFIED-EXTRACT] BIRTHPLACE: '{place}' from 'born in X' pattern")
            return place

        # Pattern: "birthplace: X"
        m = re.search(r"(?i)\bbirthplace[:\s]+([A-Z][^.,;(\n]+)", t)
        if m:
            place = m.group(1).strip()
            place = re.sub(r'[\s,\.\(\)]+$', '', place)
            if debug:
                print(f"[CERTIFIED-EXTRACT] BIRTHPLACE: '{place}' from 'birthplace X' pattern")
            return place

    # FOUNDER: "founded by X" or "founder X"
    elif relation_kind == "FOUNDER":
        # Pattern: "founded by Person" or "established by Person"
        m = re.search(r"(?i)\b(?:founded|established|created)\s+by\s+([A-Z][^.,;(\n]+)", t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] FOUNDER: '{name}' from 'founded by X' pattern")
                return name

        # Pattern: "founder: X"
        m = re.search(r"(?i)\bfounder[:\s]+([A-Z][^.,;(\n]+)", t)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\s,\.\(\)]+$', '', name)
            if _looks_like_person(name):
                if debug:
                    print(f"[CERTIFIED-EXTRACT] FOUNDER: '{name}' from 'founder X' pattern")
                return name

    # NATIONALITY: Extract the nationality/country from patterns
    elif relation_kind == "NATIONALITY":
        # Pattern: "is a Polish actress" / "was a French writer"
        m = re.search(r"(?:is|was)\s+a\s+([A-Z][a-z]+)\s+(?:actor|actress|writer|director|artist|politician|scientist)", t)
        if m:
            nationality = m.group(1).strip()
            if debug:
                print(f"[CERTIFIED-EXTRACT] NATIONALITY: '{nationality}' from 'is a X actor' pattern")
            return nationality

        # Pattern: "from Country" or "citizen of Country"
        m = re.search(r"(?i)\b(?:from|citizen\s+of|national\s+of)\s+([A-Z][^.,;(\n]+)", t)
        if m:
            nationality = m.group(1).strip()
            nationality = re.sub(r'[\s,\.\(\)]+$', '', nationality)
            if debug:
                print(f"[CERTIFIED-EXTRACT] NATIONALITY: '{nationality}' from 'from X' pattern")
            return nationality

    # DIED_ON: Extract death date/year from patterns
    elif relation_kind in ["DIED_ON", "DIED", "DEATH_DATE"]:
        # Pattern 1: "died on DATE" or "died in YEAR"
        m = re.search(r"(?i)\bdied\s+(?:on|in)\s+([A-Z][^.,;(\n]+)", t)
        if m:
            date = m.group(1).strip()
            date = re.sub(r'[\s,\.\(\)]+$', '', date)
            if debug:
                print(f"[CERTIFIED-EXTRACT] DIED_ON: '{date}' from 'died on/in X' pattern")
            return date

        # Pattern 2: "death: DATE" or "death date: DATE"
        m = re.search(r"(?i)\bdeath\s*(?:date)?[:\s]+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|\d{4})", t)
        if m:
            date = m.group(1).strip()
            if debug:
                print(f"[CERTIFIED-EXTRACT] DIED_ON: '{date}' from 'death: X' pattern")
            return date

        # Pattern 3: Just a year in parentheses after name (common in Wikipedia)
        # E.g., "John Smith (1920–1985)"
        m = re.search(r"\(\d{4}[–-](\d{4})\)", t)
        if m:
            year = m.group(1).strip()
            if debug:
                print(f"[CERTIFIED-EXTRACT] DIED_ON: '{year}' from '(birth–death)' pattern")
            return year

    # RELEASE_DATE: Extract release date/year for films, albums, etc.
    elif relation_kind in ["RELEASE_DATE", "RELEASED", "PREMIERE_DATE"]:
        # Pattern 1: "released in YEAR" or "released on DATE"
        m = re.search(r"(?i)\breleased\s+(?:on|in)\s+([A-Z][^.,;(\n]+|\d{4})", t)
        if m:
            date = m.group(1).strip()
            date = re.sub(r'[\s,\.\(\)]+$', '', date)
            if debug:
                print(f"[CERTIFIED-EXTRACT] RELEASE_DATE: '{date}' from 'released on/in X' pattern")
            return date

        # Pattern 2: "premiered in YEAR"
        m = re.search(r"(?i)\bpremiered\s+(?:on|in)\s+([A-Z][^.,;(\n]+|\d{4})", t)
        if m:
            date = m.group(1).strip()
            date = re.sub(r'[\s,\.\(\)]+$', '', date)
            if debug:
                print(f"[CERTIFIED-EXTRACT] RELEASE_DATE: '{date}' from 'premiered on/in X' pattern")
            return date

        # Pattern 3: Year in parentheses after title
        # E.g., "The Matrix (1999) is a film"
        m = re.search(r"\((\d{4})\)\s+(?:is|was)\s+a\s+(?:film|movie|album)", t)
        if m:
            year = m.group(1).strip()
            if debug:
                print(f"[CERTIFIED-EXTRACT] RELEASE_DATE: '{year}' from '(year) is a film' pattern")
            return year

    if debug:
        print(f"[CERTIFIED-EXTRACT] No match for {relation_kind} in passage")

    return None


def extract_answer_from_certified_facet(
    facet_id: str,
    certificates: List[Dict[str, Any]],
    all_passages: List[Dict[str, Any]],
    facets: List[Dict[str, Any]],
    llm: Any = None,
    question: str = ""
) -> Optional[str]:
    """
    Extract answer from the winning passage of a certified facet.

    CRITICAL: This enforces "answer from certifying passage only" contract.

    Strategy:
    1. Find the winning certificate for this facet (lowest p-value)
    2. Get the winning passage
    3. Try typed extraction first (deterministic, fast, reliable)
    4. If typed extraction fails, fall back to LLM (only on that passage)
    5. Verify extracted span appears in the winning passage

    Args:
        facet_id: The facet ID to extract answer for
        certificates: List of certificates from Safe-Cover
        all_passages: All available passages
        facets: Facet metadata (for relation_kind)
        llm: LLM interface for fallback extraction
        question: Original question (for LLM fallback)

    Returns:
        Extracted answer string, or None if extraction failed
    """
    import os
    debug = os.environ.get("TRIDENT_DEBUG_EXTRACT", "0") == "1"

    # 1. Find winning certificate for this facet
    facet_certs = [c for c in certificates if c.get("facet_id") == facet_id]
    if not facet_certs:
        if debug:
            print(f"[EXTRACT] No certificate for facet {facet_id[:8]}...")
        return None

    # Get best certificate (lowest p-value)
    best_cert = min(facet_certs, key=lambda c: float(c.get("p_value", 1.0)))
    winning_pid = best_cert.get("passage_id") or best_cert.get("pid") or ""

    if not winning_pid:
        if debug:
            print(f"[EXTRACT] Certificate has no passage_id for facet {facet_id[:8]}...")
        return None

    # 2. Get winning passage
    pid_to_passage = {p.get("pid"): p for p in all_passages}
    if winning_pid not in pid_to_passage:
        if debug:
            print(f"[EXTRACT] Winning passage {winning_pid[:12]}... not found")
        return None

    winning_passage = pid_to_passage[winning_pid]
    passage_text = winning_passage.get("text", "")

    if not passage_text:
        if debug:
            print(f"[EXTRACT] Winning passage has no text")
        return None

    # 3. Get relation_kind from facet metadata
    facet_dict = None
    for f in facets:
        if f.get("facet_id") == facet_id:
            facet_dict = f
            break

    relation_kind = None
    if facet_dict:
        tpl = facet_dict.get("template", {})
        # Try multiple field names for relation type
        relation_kind = (
            tpl.get("outer_relation_type") or
            tpl.get("inner_relation_type") or
            tpl.get("relation_kind") or
            tpl.get("relation_type")
        )

    if debug:
        print(f"\n[EXTRACT] Facet {facet_id[:8]}...")
        print(f"  Winning PID: {winning_pid[:12]}...")
        print(f"  Relation kind: {relation_kind}")
        print(f"  Passage length: {len(passage_text)} chars")

    # 4. Try typed extraction first (PRIMARY PATH)
    extracted = None
    extraction_method = "none"

    if relation_kind:
        extracted = extract_object_from_certified_passage(passage_text, relation_kind)
        if extracted:
            extraction_method = "typed"
            if debug:
                print(f"  ✓ Typed extraction: '{extracted}'")

    # 5. Fallback to LLM (SECONDARY PATH) - only if typed extraction failed
    if not extracted and llm and question:
        if debug:
            print(f"  Typed extraction failed, trying LLM fallback...")

        # Use constrained selection with ONLY the winning passage
        from .chain_builder import constrained_span_select
        result = constrained_span_select(
            llm=llm,
            question=question,
            winner_passages=[winning_passage],
            max_candidates=5,
            max_chars_per_passage=600
        )

        if result.reason == "OK" and result.answer:
            extracted = result.answer
            extraction_method = "llm_fallback"
            if debug:
                print(f"  ✓ LLM fallback: '{extracted}'")
        elif debug:
            print(f"  ✗ LLM fallback failed: {result.reason}")

    # 6. Verify extracted span appears in winning passage (provenance check)
    if extracted:
        # Normalize for verification (handle Unicode, punctuation variations)
        # CRITICAL: Use robust normalization to avoid rejecting correct answers
        import unicodedata
        import string

        def normalize_for_verification(text: str) -> str:
            """
            Robust normalization for span verification.

            Steps:
            1. Unicode normalize (NFKC) - handles diacritics, combining chars
            2. Lowercase
            3. Strip surrounding punctuation
            4. Collapse whitespace

            This prevents false rejections like:
            - "Małgorzata Braunek" vs "Małgorzata Braunek,"
            - "D'Arcy" vs "D'Arcy"
            - "1985  " vs "1985"
            """
            if not text:
                return ""

            # 1. Unicode normalize (NFKC)
            text = unicodedata.normalize('NFKC', text)

            # 2. Lowercase
            text = text.lower()

            # 3. Strip surrounding punctuation and whitespace
            text = text.strip()
            text = text.strip(string.punctuation)
            text = text.strip()

            # 4. Collapse whitespace
            text = ' '.join(text.split())

            return text

        extracted_norm = normalize_for_verification(extracted)
        passage_norm = normalize_for_verification(passage_text)

        # Primary check: normalized match
        if extracted_norm in passage_norm:
            if debug:
                print(f"  ✓ Provenance verified (normalized): '{extracted}' found in passage")
                print(f"  Method: {extraction_method}")
            return extracted

        # Fallback check: raw substring match (in case normalization was too aggressive)
        if extracted.strip() in passage_text:
            if debug:
                print(f"  ✓ Provenance verified (raw): '{extracted}' found in passage")
                print(f"  Method: {extraction_method}")
            return extracted

        # Failed both checks
        if debug:
            print(f"  ✗ Extracted '{extracted}' not found in passage (provenance check failed)")
            print(f"    Normalized: '{extracted_norm}' not in passage")
        return None

    if debug:
        print(f"  ✗ No extraction succeeded")

    return None


def get_llm_answer_certificate(
    llm: Any,
    question: str,
    passages: List[Dict[str, Any]],
    max_passages: int = 10,
    max_chars_per_passage: int = 600
) -> Optional[AnswerCertificate]:
    """
    🟨 TIER 2: LLM HEURISTIC ANSWER (not a formal certificate!).

    Get LLM to extract answer with provenance (quote from passage).
    This is used when Safe-Cover has no answer facets certified (Tier 1 fails).

    ⚠️  CRITICAL: This is a HEURISTIC, not a CERTIFIED answer:
    - Answer is plausible, evidence-quoted, but NOT formally verified
    - Must be labeled separately from Tier 1 (Safe-Cover/Deterministic)
    - Used to improve EM/F1, but not counted as "certified" in research metrics

    Acceptance rules:
    - Confidence ≥ 0.85
    - Answer type matches question intent
    - Answer found in at least one passage (consistency check)
    - Preferably found in ≥2 passages (higher confidence)

    Args:
        llm: LLM interface
        question: Original question
        passages: Top passages (already retrieved/scored)
        max_passages: Maximum passages to send to LLM
        max_chars_per_passage: Truncate passages to this length

    Returns:
        AnswerCertificate (misnomer - actually HeuristicAnswer) if verification passes, None otherwise
    """
    import json
    import os
    import unicodedata

    debug = os.environ.get("TRIDENT_DEBUG_LLM_CERT", "0") == "1"

    # STEP 1: Print version banner to prove we're running patched code
    if debug:
        import trident.chain_builder
        print(f"\n{'='*80}")
        print(f"[TIER 2: LLM HEURISTIC] USING PATCHED CODE")
        print(f"  File: {trident.chain_builder.__file__}")
        print(f"  Patch ID: 2026-01-01-P0-FINAL")
        print(f"  Backend: {llm.__class__.__name__}")
        print(f"  NOTE: This is HEURISTIC, not certified")
        print(f"{'='*80}")

    # CHANGE 2: Abstain-friendly - no passages means abstain, not error
    if not passages:
        if debug:
            print(f"[TIER 2: LLM HEURISTIC] No passages provided - abstaining")
        return None

    # P0-1 FIX: More aggressive retry strategy (up to 3 attempts total)
    # Attempt 1: Standard parameters
    # Attempt 2: Hard prompt cut (max total context 3000 chars)
    # Attempt 3: Extreme cut + different sampling params

    def _try_llm_extraction(max_chars: int, max_tokens: int, stop_tokens: list = None,
                           attempt_label: str = "", temperature: float = 0.0,
                           top_p: float = None, max_total_chars: int = None):
        """Try LLM extraction with specific truncation/token limits."""
        # Prepare passages for LLM (truncate and format)
        evidence_parts = []
        pid_to_passage_local = {}

        for i, p in enumerate(passages[:max_passages]):
            pid_local = p.get("pid", f"p{i}")
            text_local = p.get("text", "")
            title_local = p.get("title", "")

            # Truncate text
            if len(text_local) > max_chars:
                text_local = text_local[:max_chars] + "..."

            pid_to_passage_local[pid_local] = p
            evidence_parts.append(f"[{pid_local}] {title_local}\n{text_local}")

        evidence_str_local = "\n\n".join(evidence_parts)

        # P0-1: Hard cut on total context length if specified
        if max_total_chars and len(evidence_str_local) > max_total_chars:
            evidence_str_local = evidence_str_local[:max_total_chars] + "\n..."

        # Build LLM prompt requesting structured answer certificate with confidence
        prompt_local = f"""Question: {question}

Evidence passages:
{evidence_str_local}

TASK: Extract the answer to the question with provenance and confidence.

OUTPUT FORMAT (JSON):
{{
  "answer": "the extracted answer",
  "passage_id": "pid of supporting passage",
  "quote": "relevant quote from that passage supporting the answer",
  "answer_type": "entity or date or number or yesno or choice",
  "confidence": 0.95
}}

INSTRUCTIONS:
1. The answer should be directly stated or strongly implied in the passage
2. Provide a supporting quote (does not need to be exact substring, paraphrase ok)
3. Set confidence 0.0-1.0 based on how certain you are:
   - 0.95-1.0: Answer is explicitly stated in passage
   - 0.85-0.94: Answer is strongly implied or requires minor inference
   - 0.70-0.84: Answer requires moderate inference
   - Below 0.70: Uncertain or speculative
4. If no answer can be found, return {{"answer": "", "reason": "no_verified_answer", "confidence": 0.0}}

JSON:"""

        if debug:
            print(f"\n[TIER 2: LLM HEURISTIC] {attempt_label}")
            print(f"  Backend: {llm.__class__.__name__}")
            print(f"  Prompt length: {len(prompt_local)} chars")
            print(f"  Max chars/passage: {max_chars}")
            print(f"  Max total context: {max_total_chars if max_total_chars else 'unlimited'}")
            print(f"  Max new tokens: {max_tokens}")
            print(f"  Temperature: {temperature}")
            print(f"  Top-p: {top_p if top_p else 'default'}")
            print(f"  Stop tokens: {stop_tokens}")

        # Call LLM with specified parameters
        try:
            kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
            }
            if top_p is not None:
                kwargs['top_p'] = top_p
            if stop_tokens:
                kwargs['stop'] = stop_tokens

            response_local = llm.generate(prompt_local, **kwargs)
            return response_local, pid_to_passage_local
        except Exception as e:
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Exception during generation: {e}")
            return None, pid_to_passage_local

    # Attempt 1: Standard parameters
    response, pid_to_passage = _try_llm_extraction(
        max_chars=max_chars_per_passage,
        max_tokens=512,
        temperature=0.0,
        attempt_label="Attempt 1/3 (standard prompt)"
    )

    # Parse LLM response (try JSON, then fallback format)
    answer = ""
    pid = ""
    quote = ""
    answer_type = "entity"
    confidence = 0.0  # P1: Add confidence field

    # P0-2: Track backend used for metrics
    backend_class = llm.__class__.__name__ if llm else "Unknown"
    backend_used = "vllm" if backend_class == "VLLMInterface" else backend_class.lower()
    fail_reason = None

    try:
        # P0-1 FIX: Aggressive retry strategy (up to 3 attempts)
        attempt_count = 1

        if response is None or not hasattr(response, 'text') or not response.text or not response.text.strip():
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Attempt 1 failed (None or empty response)")
                print(f"  Retrying with hard prompt cut...")

            # Attempt 2: Hard prompt cut (max total context 3000 chars)
            response, pid_to_passage = _try_llm_extraction(
                max_chars=400,
                max_tokens=64,
                max_total_chars=3000,  # P0-1: Hard cut on total context
                stop_tokens=["\n\n", "Question:", "Evidence:"],
                temperature=0.0,
                attempt_label="Attempt 2/3 (hard prompt cut)"
            )
            attempt_count = 2

            # Check again after Attempt 2
            if response is None or not hasattr(response, 'text') or not response.text or not response.text.strip():
                if debug:
                    print(f"[TIER 2: LLM HEURISTIC] Attempt 2 failed")
                    print(f"  Trying Attempt 3 with different sampling params...")

                # Attempt 3: Extreme cut + different sampling params
                # Some backends choke on temperature=0.0, try small non-zero
                response, pid_to_passage = _try_llm_extraction(
                    max_chars=250,
                    max_tokens=64,
                    max_total_chars=2500,  # Even harder cut
                    stop_tokens=["\n\n", "Question:", "Evidence:"],
                    temperature=0.1,  # P0-1: Non-zero temperature
                    top_p=0.95,  # P0-1: Add top_p sampling
                    attempt_label="Attempt 3/3 (extreme cut + sampling)"
                )
                attempt_count = 3

                # P0-2: Backend fallback - try transformers if vLLM keeps failing
                if response is None or not hasattr(response, 'text') or not response.text or not response.text.strip():
                    backend_name = llm.__class__.__name__ if llm else "Unknown"

                    if debug:
                        print(f"[TIER 2: LLM HEURISTIC] ✗ All 3 vLLM attempts failed")
                        print(f"  Backend: {backend_name}")

                    # P0-2: Try HF transformers backend as fallback if vLLM failed
                    if backend_name == "VLLMInterface":
                        if debug:
                            print(f"[TIER 2: HF FALLBACK] Switching from vLLM to HuggingFace transformers...")

                        try:
                            # Import LLMInterface (HF backend)
                            from trident.llm_interface import LLMInterface

                            # Get model name from vLLM backend (or use default)
                            model_name = getattr(llm, 'model_name', 'meta-llama/Llama-3.1-8B-Instruct')

                            if debug:
                                print(f"[TIER 2: HF FALLBACK] Instantiating HF backend with model: {model_name}")

                            # Instantiate HF backend (8-bit quantization for memory efficiency)
                            hf_llm = LLMInterface(
                                model_name=model_name,
                                device="cuda:0",
                                temperature=0.0,
                                max_new_tokens=64,
                                load_in_8bit=True  # Use 8-bit to save memory
                            )

                            if debug:
                                print(f"[TIER 2: HF FALLBACK] HF backend loaded successfully")
                                print(f"[TIER 2: HF FALLBACK] Retrying with short prompt (3000 chars max, 64 tokens)...")

                            # Retry with HF backend (attempt-2 style: short prompt)
                            # Use a temporary closure to capture hf_llm
                            def _try_hf_extraction():
                                """Try extraction with HF backend."""
                                evidence_parts = []
                                pid_to_passage_local = {}

                                for i, p in enumerate(passages[:max_passages]):
                                    pid_local = p.get("pid", f"p{i}")
                                    text_local = p.get("text", "")
                                    title_local = p.get("title", "")

                                    # Truncate to 400 chars per passage
                                    if len(text_local) > 400:
                                        text_local = text_local[:400] + "..."

                                    pid_to_passage_local[pid_local] = p
                                    evidence_parts.append(f"[{pid_local}] {title_local}\n{text_local}")

                                evidence_str_local = "\n\n".join(evidence_parts)

                                # Hard cut at 3000 chars total
                                if len(evidence_str_local) > 3000:
                                    evidence_str_local = evidence_str_local[:3000] + "\n..."

                                # Build prompt (same format as vLLM attempts)
                                prompt_local = f"""Question: {question}

Evidence passages:
{evidence_str_local}

TASK: Extract the answer to the question with provenance and confidence.

OUTPUT FORMAT (JSON):
{{
  "answer": "the extracted answer",
  "passage_id": "pid of supporting passage",
  "quote": "relevant quote from that passage supporting the answer",
  "answer_type": "entity or date or number or yesno or choice",
  "confidence": 0.95
}}

INSTRUCTIONS:
1. The answer should be directly stated or strongly implied in the passage
2. Provide a supporting quote (does not need to be exact substring, paraphrase ok)
3. Set confidence 0.0-1.0 based on how certain you are:
   - 0.95-1.0: Answer is explicitly stated in passage
   - 0.85-0.94: Answer is strongly implied or requires minor inference
   - 0.70-0.84: Answer requires moderate inference
   - Below 0.70: Uncertain or speculative
4. If no answer can be found, return {{"answer": "", "reason": "no_verified_answer", "confidence": 0.0}}

JSON:"""

                                if debug:
                                    print(f"  Prompt length: {len(prompt_local)} chars")

                                # Call HF backend
                                try:
                                    response_local = hf_llm.generate(
                                        prompt_local,
                                        max_new_tokens=64,
                                        temperature=0.0,
                                        stop=["\n\n", "Question:", "Evidence:"]
                                    )
                                    return response_local, pid_to_passage_local
                                except Exception as e:
                                    if debug:
                                        print(f"[TIER 2: HF FALLBACK] Exception during HF generation: {e}")
                                    return None, pid_to_passage_local

                            response, pid_to_passage = _try_hf_extraction()
                            attempt_count = 4  # Mark as attempt 4 (HF fallback)

                            if response and hasattr(response, 'text') and response.text and response.text.strip():
                                if debug:
                                    print(f"[TIER 2: HF FALLBACK] ✓ HF backend succeeded!")
                                # P0-2: Track that HF backend was used
                                backend_used = "hf"
                                fail_reason = "vllm_returned_none_3x"
                                # Continue to parse the response (don't return here)
                            else:
                                if debug:
                                    print(f"[TIER 2: HF FALLBACK] ✗ HF backend also failed")
                                    print(f"  Will try deterministic fallback if available")
                                # P0-3: Caller will try deterministic fallback before abstaining
                                return None

                        except Exception as e:
                            if debug:
                                print(f"[TIER 2: HF FALLBACK] Failed to instantiate HF backend: {e}")
                                print(f"  Will try deterministic fallback if available")
                            # P0-3: Caller will try deterministic fallback before abstaining
                            return None
                    else:
                        # Not vLLM backend, just return None
                        if debug:
                            print(f"  Will try deterministic fallback if available")
                        # P0-3: Caller will try deterministic fallback before abstaining
                        return None

        response_text = response.text.strip()

        if debug and attempt_count > 1:
            print(f"[TIER 2: LLM HEURISTIC] ✓ Succeeded on attempt {attempt_count}")

        if debug:
            print(f"\n[TIER 2: LLM HEURISTIC] Raw LLM response:")
            print(f"  Length: {len(response_text)} chars")
            print(f"  First 300 chars: {response_text[:300]}")
            if len(response_text) > 300:
                print(f"  ... (truncated)")

        # Parse JSON (try to extract if wrapped in markdown)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Extracted from ```json block")
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Extracted from ``` block")

        # Try JSON parsing first
        json_parse_success = False

        try:
            cert_dict = json.loads(response_text)
            json_parse_success = True

            answer = cert_dict.get("answer", "").strip()
            pid = cert_dict.get("passage_id", "").strip()
            quote = cert_dict.get("quote", "").strip()
            answer_type = cert_dict.get("answer_type", "entity").strip()
            confidence = float(cert_dict.get("confidence", 0.0))  # P1: Extract confidence

            if debug:
                print(f"\n[TIER 2: LLM HEURISTIC] JSON parsed successfully")
                print(f"  Answer: '{answer}'")
                print(f"  PID: {pid}")
                print(f"  Quote: '{quote[:100]}...'")
                print(f"  Type: {answer_type}")
                print(f"  Confidence: {confidence:.2f}")

        except json.JSONDecodeError as e:
            # CHANGE 2: Abstain-friendly - truncated JSON means abstain
            if "Unterminated string" in str(e):
                if debug:
                    print(f"[TIER 2: LLM HEURISTIC] JSON truncated (hit token limit) - abstaining")
                return None

            # P0 FIX: Handle "Extra data" error (trailing garbage after JSON)
            # Try to extract just the JSON object by finding matching braces
            if "Extra data" in str(e) and '{' in response_text:
                if debug:
                    print(f"[TIER 2: LLM HEURISTIC] JSON parse error (Extra data) - extracting JSON objects")

                # Find ALL complete JSON objects in the response
                json_objects = []
                search_start = 0
                while True:
                    start_idx = response_text.find('{', search_start)
                    if start_idx == -1:
                        break

                    brace_count = 0
                    end_idx = -1
                    for i in range(start_idx, len(response_text)):
                        if response_text[i] == '{':
                            brace_count += 1
                        elif response_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break

                    if end_idx != -1:
                        json_str = response_text[start_idx:end_idx]
                        try:
                            parsed = json.loads(json_str)
                            json_objects.append(parsed)
                        except json.JSONDecodeError:
                            pass  # Skip invalid JSON
                        search_start = end_idx
                    else:
                        break

                if debug:
                    print(f"  Found {len(json_objects)} valid JSON objects")

                # Prefer non-empty JSON objects (skip all-empty fields)
                selected_dict = None
                for obj in json_objects:
                    ans = obj.get("answer", "").strip()
                    p = obj.get("passage_id", "").strip()
                    q = obj.get("quote", "").strip()
                    # If this object has at least one non-empty field, prefer it
                    if ans or p or q:
                        selected_dict = obj
                        if debug:
                            print(f"  Selected non-empty JSON: answer='{ans[:50] if ans else ''}', pid={p}, quote_len={len(q)}")
                        break

                # If all objects are empty, just use the first one
                if not selected_dict and json_objects:
                    selected_dict = json_objects[0]
                    if debug:
                        print(f"  All JSON objects empty, using first one")

                if selected_dict:
                    json_parse_success = True
                    cert_dict = selected_dict

                    answer = cert_dict.get("answer", "").strip()
                    pid = cert_dict.get("passage_id", "").strip()
                    quote = cert_dict.get("quote", "").strip()
                    answer_type = cert_dict.get("answer_type", "entity").strip()
                    confidence = float(cert_dict.get("confidence", 0.0))  # P1: Extract confidence

                    if debug:
                        print(f"[TIER 2: LLM HEURISTIC] ✓ JSON extracted successfully (removed trailing garbage)")
                        print(f"  Answer: '{answer}'")
                        print(f"  PID: {pid}")
                        print(f"  Quote: '{quote[:100]}...'")
                        print(f"  Confidence: {confidence:.2f}")

            # If JSON parsing failed (including extraction), try line-by-line fallback
            if not json_parse_success:
                if debug and "Extra data" not in str(e):
                    print(f"[TIER 2: LLM HEURISTIC] JSON parse error: {e}")
                    print(f"  Trying line-by-line fallback...")

                # Parse line-by-line
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line.startswith("ANSWER:"):
                        answer = line.split("ANSWER:", 1)[1].strip()
                    elif line.startswith("PID:") or line.startswith("PASSAGE_ID:"):
                        pid = line.split(":", 1)[1].strip()
                        # Clean up "context_X" to just the PID
                        pid = pid.replace("context_", "")
                    elif line.startswith("QUOTE:"):
                        quote = line.split("QUOTE:", 1)[1].strip()
                        # Remove surrounding quotes if present
                        quote = quote.strip('"\'')
                    elif line.startswith("TYPE:") or line.startswith("ANSWER_TYPE:"):
                        answer_type = line.split(":", 1)[1].strip()

                if debug:
                    print(f"[TIER 2: LLM HEURISTIC] Fallback parsed:")
                    print(f"  Answer: '{answer}'")
                    print(f"  PID: {pid}")
                    print(f"  Quote: '{quote[:100]}...'")

                # CHANGE 2: Abstain-friendly - incomplete parsing means abstain
                if not (answer and quote and pid):
                    if debug:
                        print(f"[TIER 2: LLM HEURISTIC] Fallback parsing incomplete - abstaining")
                        print(f"  Parsed answer: '{answer}', pid: '{pid}', quote length: {len(quote) if quote else 0}")
                    return None

        # Check for empty answer - could be abstention or error
        if not answer or not quote or not pid:
            # Check if this is a valid abstention (has "reason" field)
            # Parse cert_dict if we haven't already (might only have answer/pid/quote vars)
            abstention_reason = None
            try:
                # Try to get reason from already-parsed dict if available
                if json_parse_success and 'cert_dict' in locals():
                    abstention_reason = cert_dict.get("reason", "").strip()
                elif not json_parse_success:
                    # Try parsing response_text one more time to check for reason field
                    try:
                        temp_dict = json.loads(response_text)
                        abstention_reason = temp_dict.get("reason", "").strip()
                    except:
                        pass
            except:
                pass

            # If LLM explicitly provided a reason for abstention, return None gracefully
            if abstention_reason:
                if debug:
                    print(f"[TIER 2: LLM HEURISTIC] LLM abstained (reason: '{abstention_reason}')")
                    print(f"  This is expected for unanswerable questions")
                return None

            # CHANGE 2: Abstain-friendly - incomplete response means abstain
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Empty fields in response - abstaining")
                print(f"  answer={bool(answer)}, quote={bool(quote)}, pid={bool(pid)}")
            return None

        # CHANGE 2: Abstain-friendly - invalid PID means abstain (hallucination)
        if pid not in pid_to_passage:
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Invalid PID '{pid}' - abstaining (LLM hallucinated)")
                print(f"  Available PIDs: {list(pid_to_passage.keys())[:5]}...")
            return None

        passage_text = pid_to_passage[pid].get("text", "")

        # CHANGE 2: Abstain-friendly - no passage text means abstain
        if not passage_text:
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Passage has no text - abstaining")
                print(f"  PID: {pid}")
            return None

        # P1: RELAXED VERIFICATION - 2-tier certification policy
        # Type B: LLM-Confident Certificate (Soft-Verified)
        # Requirements: confidence >= 0.85 AND answer found in passage

        def normalize_for_match(text: str) -> str:
            """Normalize for substring matching (Unicode + whitespace)."""
            text = unicodedata.normalize('NFKC', text)
            text = ' '.join(text.split())
            return text

        # STEP 3: Introduce Tier-2 confidence bands (instead of hard reject)
        # ≥ 0.85 → accept (high confidence)
        # 0.70–0.85 → accept but mark low_confidence
        # < 0.70 → abstain
        low_confidence_flag = False
        if confidence < 0.70:
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Confidence too low: {confidence:.2f} < 0.70")
                print(f"  Rejecting (below minimum threshold)")
            return None
        elif confidence < 0.85:
            low_confidence_flag = True
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] ⚠ Low confidence: {confidence:.2f} (0.70-0.85 band)")
                print(f"  Will accept but mark as low_confidence")

        # CHANGE 1: Gate on question-intent → answer-type consistency
        intent = detect_question_intent(question)
        allowed_types = get_allowed_answer_types(intent)
        if answer_type not in allowed_types:
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Answer type mismatch")
                print(f"  Question intent: '{intent}'")
                print(f"  Answer type: '{answer_type}'")
                print(f"  Allowed types: {allowed_types}")
                print(f"  Rejecting certificate (type mismatch)")
            return None

        answer_norm = normalize_for_match(answer)
        passage_norm = normalize_for_match(passage_text)
        quote_norm = normalize_for_match(quote)

        # STEP 2: RELAXED acceptance criteria (safely)
        # Changed from: answer ⊂ quote ⊂ passage
        # To: answer ⊂ passage AND quote_length ≥ 15 chars
        # Why: Many valid answers appear without clean contiguous quotes
        #      Especially comparisons and yes/no questions
        # We still prevent hallucination and keep provenance

        # Check 1: Answer must appear in passage (prevents hallucination)
        answer_in_passage = (
            answer_norm in passage_norm or
            answer_norm.lower() in passage_norm.lower()
        )

        if not answer_in_passage:
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] ✗ FAILED: Answer not in passage")
                print(f"  Answer (normalized): '{answer_norm}'")
                print(f"  Passage (first 200): '{passage_text[:200]}'")
            return None

        # Check 2: Quote must be meaningful (≥15 chars)
        if len(quote_norm) < 15:
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] ✗ FAILED: Quote too short ({len(quote_norm)} chars < 15)")
                print(f"  Quote: '{quote}'")
            return None

        if debug:
            print(f"[TIER 2: LLM HEURISTIC] ✓ Verification passed")
            print(f"  Answer in passage: YES")
            print(f"  Quote length: {len(quote_norm)} chars (≥15)")
            print(f"  Confidence: {confidence:.2f}")

        # CHANGE 3: STRENGTHENED self-consistency checks
        # Don't trust self-reported confidence blindly - verify answer appears in multiple passages
        passages_with_answer = 0
        answer_occurrences = 0
        passages_with_answer_list = []
        for p in passages:
            p_text = p.get("text", "")
            p_id = p.get("pid", "")
            if not p_text:
                continue
            p_norm = normalize_for_match(p_text).lower()
            if answer_norm.lower() in p_norm:
                passages_with_answer += 1
                passages_with_answer_list.append(p_id)
                # Count how many times answer appears in this passage
                answer_occurrences += p_norm.count(answer_norm.lower())

        # STRENGTHENED: Check if answer appears in passages OTHER than the supporting passage
        other_passages_with_answer = [p_id for p_id in passages_with_answer_list if p_id != pid]

        # Calibrate confidence based on consistency
        calibrated_confidence = confidence
        if passages_with_answer >= 3:
            # Answer found in many passages - strong boost
            calibrated_confidence = min(1.0, confidence + 0.08)
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Strong confidence boost: answer in {passages_with_answer} passages")
        elif passages_with_answer >= 2:
            # Answer found in multiple passages - modest boost
            calibrated_confidence = min(1.0, confidence + 0.05)
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Confidence boost: answer in {passages_with_answer} passages")
        elif passages_with_answer == 1 and not other_passages_with_answer:
            # STRENGTHENED: Answer ONLY in supporting passage, not corroborated elsewhere
            # This is suspicious - lower confidence
            calibrated_confidence = max(0.0, confidence - 0.10)
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] ⚠ Confidence penalty: answer only in supporting passage")
                print(f"  No corroboration from other passages")
            # If calibrated confidence drops below threshold, reject
            # STEP 3: Updated threshold to 0.70 to match confidence bands
            if calibrated_confidence < 0.70:
                if debug:
                    print(f"[TIER 2: LLM HEURISTIC] ✗ FAILED: Calibrated confidence {calibrated_confidence:.2f} < 0.70")
                    print(f"  Answer lacks cross-passage validation")
                return None
            elif calibrated_confidence < 0.85:
                low_confidence_flag = True  # Mark as low confidence after calibration
        elif passages_with_answer == 0:
            # Answer not found in ANY passage - major red flag
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] ✗ FAILED: Answer not found in any passage (confidence check)")
                print(f"  Answer: '{answer}'")
                print(f"  This is likely hallucination")
            return None

        if answer_occurrences >= 5:
            # Answer appears many times across passages - strong signal
            calibrated_confidence = min(1.0, calibrated_confidence + 0.05)
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Confidence boost: answer appears {answer_occurrences} times")
        elif answer_occurrences >= 3:
            # Answer appears multiple times across passages
            calibrated_confidence = min(1.0, calibrated_confidence + 0.03)
            if debug:
                print(f"[TIER 2: LLM HEURISTIC] Confidence boost: answer appears {answer_occurrences} times")

        if debug and calibrated_confidence != confidence:
            print(f"[TIER 2: LLM HEURISTIC] Confidence calibrated: {confidence:.2f} → {calibrated_confidence:.2f}")

        # STEP 3: Determine final confidence band
        if calibrated_confidence >= 0.85:
            confidence_band = "high (≥0.85)"
        elif calibrated_confidence >= 0.70:
            confidence_band = "low (0.70-0.85)"
        else:
            confidence_band = "too low (<0.70)"

        # All verification passed!
        cert = AnswerCertificate(
            answer=answer,
            pid=pid,
            quote=quote,
            answer_type=answer_type,
            confidence=calibrated_confidence,  # CHANGE 3: Use calibrated confidence
            verified=True,
            tier2_backend=backend_used,  # P0-2: Track backend used
            tier2_fail_reason=fail_reason  # P0-2: Track why vLLM failed (if HF was used)
        )

        if debug:
            print(f"[TIER 2: LLM HEURISTIC] ✓ VERIFIED Answer Certificate")
            print(f"  Answer: '{answer}'")
            print(f"  PID: {pid[:12]}...")
            print(f"  Quote length: {len(quote)} chars")
            print(f"  Confidence: {calibrated_confidence:.2f} [{confidence_band}]")
            print(f"  Backend: {backend_used}")  # P0-2: Show backend used
            if fail_reason:
                print(f"  Fail reason: {fail_reason}")  # P0-2: Show why vLLM failed
            if low_confidence_flag:
                print(f"  ⚠ LOW CONFIDENCE FLAG: Answer accepted but marked for review")

        return cert

    except Exception as e:
        # P0: Don't swallow exceptions - ALWAYS re-raise to get full traceback
        if debug:
            import traceback
            print(f"\n[TIER 2: LLM HEURISTIC] ✗ EXCEPTION:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {e}")
            print(f"  Traceback:")
            traceback.print_exc()
        else:
            # In production, still log before re-raising
            print(f"[TIER 2: LLM HEURISTIC] Exception: {type(e).__name__}: {e}")
        # Re-raise the exception to force hard failure with traceback
        raise


def looks_generic(entity: str) -> bool:
    """
    FIX 3: Check if bound entity is generic/invalid for hop-2.

    Returns True if entity looks like:
    - Generic nouns: "film", "movie", "Polish film"
    - Common words in lowercase
    - Too short or invalid

    Returns False if entity looks like a proper named entity.
    """
    if not entity or len(entity.strip()) < 2:
        return True

    entity = entity.strip()

    # Lowercase common nouns are generic
    if entity.lower() == entity:
        return True

    # Generic words that should never be hop-2 anchors
    generic_words = {
        'film', 'movie', 'book', 'song', 'album', 'series',
        'city', 'country', 'place', 'person', 'thing',
        'war', 'battle', 'event', 'show', 'game'
    }

    entity_lower = entity.lower()

    # Check if it's just a generic word
    if entity_lower in generic_words:
        return True

    # Check if it's a generic phrase like "Polish film", "American movie"
    tokens = entity_lower.split()
    if len(tokens) == 2 and tokens[1] in generic_words:
        return True

    # If it contains only generic words, it's generic
    if all(tok in generic_words for tok in tokens if tok):
        return True

    return False


def build_inner_question_from_facet(facet: Dict[str, Any]) -> str:
    """
    Build the inner question from a hop-1 facet for entity binding.

    CRITICAL: Normalizes subject="?" to proper WH words to avoid malformed prompts.
    Example: "? directed Polish-Russian War?" → "Who directed Polish-Russian War?"

    Example facet template:
        {"subject": "?", "predicate": "directed", "object": "Polish-Russian War",
         "inner_relation_type": "DIRECTOR"}

    Returns:
        "Who is the director of Polish-Russian War?"
    """
    template = facet.get("template", {})
    subject = template.get("subject", "Who")
    predicate = template.get("predicate", "")
    obj = template.get("object", "")
    relation_type = template.get("inner_relation_type") or template.get("relation_kind", "")

    # CRITICAL FIX: Normalize subject="?" to proper WH word
    # This prevents malformed binding questions like "? directed Polish-Russian War?"
    if subject in {"?", "", "who", "what", "which", "where", "when"}:
        # Infer WH word from relation type
        if relation_type:
            rel_upper = relation_type.upper()
            if rel_upper in {"DIRECTOR", "MOTHER", "FATHER", "SPOUSE", "AUTHOR", "FOUNDER", "CREATOR"}:
                subject = "Who"
            elif rel_upper in {"BIRTHPLACE", "BORN", "LOCATION", "CAPITAL", "LOCATED_IN"}:
                subject = "Where"
            elif rel_upper in {"DATE", "BORN_DATE", "DIED_DATE", "FOUNDED_DATE"}:
                subject = "When"
            elif "COMPARISON" in rel_upper or "CHOICE" in rel_upper:
                subject = "Which"
            else:
                subject = "What"
        else:
            # Default based on predicate if no relation type
            pred_lower = predicate.lower()
            if any(kw in pred_lower for kw in ["director", "mother", "father", "spouse", "author", "founder", "creator"]):
                subject = "Who"
            elif any(kw in pred_lower for kw in ["born", "birthplace", "location", "capital", "located"]):
                subject = "Where"
            elif any(kw in pred_lower for kw in ["date", "when", "year"]):
                subject = "When"
            else:
                subject = "What"

    # CRITICAL FIX: Ensure predicate is grammatical
    # Prefer "is the director of" over bare verb "directed"
    if relation_type and predicate:
        rel_upper = relation_type.upper()
        pred_lower = predicate.lower().strip()

        # Normalize common bare verbs to grammatical forms
        if rel_upper == "DIRECTOR" and pred_lower in {"directed", "direct"}:
            predicate = "is the director of"
        elif rel_upper == "AUTHOR" and pred_lower in {"wrote", "written", "authored"}:
            predicate = "is the author of"
        elif rel_upper == "FOUNDER" and pred_lower in {"founded", "established", "created"}:
            predicate = "is the founder of"
        elif rel_upper in {"MOTHER", "FATHER"} and pred_lower in {"parent", "parents"}:
            if rel_upper == "MOTHER":
                predicate = "is the mother of"
            else:
                predicate = "is the father of"
        elif rel_upper == "SPOUSE" and pred_lower in {"married", "spouse"}:
            predicate = "is the spouse of"
        elif rel_upper in {"BIRTHPLACE", "BORN"} and pred_lower in {"born", "birthplace"}:
            predicate = "was born in"

    # Build natural question
    question = f"{subject} {predicate} {obj}".strip()

    # Ensure it ends with question mark
    if not question.endswith("?"):
        question += "?"

    return question


def get_winner_passages_only(
    certificates: List[Dict[str, Any]],
    passages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract only the certificate-winning passages (deduplicated).

    CERTIFIED-ONLY INVARIANT:
    - Only includes passages that have valid certificates
    - Filters out passages with deny-listed titles
    - Orders by certificate p-value (best first)

    Args:
        certificates: List of certificates with passage_id
        passages: List of all selected passages

    Returns:
        List of unique winning passages (ordered by certificate p-value)
    """
    if not certificates:
        return []

    # CERTIFIED-ONLY: Get set of certified passage IDs
    certified_pids = _cert_pid(certificates)

    # Build passage lookup (only certified passages, excluding deny-listed)
    pid_to_passage = {}
    for p in passages:
        pid = p.get("pid", "")
        if pid and pid in certified_pids:
            title = p.get("title", "")
            if not _is_deny_title(title):
                pid_to_passage[pid] = p

    # Sort certificates by p-value (best first)
    sorted_certs = sorted(certificates, key=lambda c: c.get("p_value", 1.0))

    # Collect unique winning passages
    seen_pids = set()
    winners = []

    for cert in sorted_certs:
        pid = cert.get("passage_id", "")
        # CERTIFIED-ONLY: Must be in pid_to_passage (certified and not deny-listed)
        if pid and pid not in seen_pids and pid in pid_to_passage:
            seen_pids.add(pid)
            winners.append(pid_to_passage[pid])

    return winners
