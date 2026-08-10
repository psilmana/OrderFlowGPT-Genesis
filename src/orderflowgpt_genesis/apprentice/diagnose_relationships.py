#!/usr/bin/env python3
"""
diagnose_relationships.py

Analyzes a transcript to find actual relationship-indicating phrases
that Fabio uses, so we can tune the RelationshipExtractor patterns.

Usage:
    python diagnose_relationships.py --transcript assets/fabio/transcripts/Lesson01.txt --ontology fabio_ontology.json
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter


def load_ontology(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_concept_pairs(text: str, concept_names: list) -> list:
    """Find all pairs of concepts that appear in the same sentence."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'\(])', text)
    pairs = []
    for sent in sentences:
        sent_lower = sent.lower()
        found = []
        for name in concept_names:
            if name.lower() in sent_lower:
                found.append(name)
        if len(found) >= 2:
            for i, a in enumerate(found):
                for b in found[i+1:]:
                    pairs.append((a, b, sent.strip()[:120]))
    return pairs


def find_relationship_verbs(text: str, concept_names: list) -> Counter:
    """Find verbs between concept names that might indicate relationships."""
    text_lower = text.lower()
    verbs = Counter()

    # Common relationship-indicating verbs/phrases
    verb_patterns = [
        r'\b(is|are|was|were)\b',
        r'\b(means?|indicates?|shows?|signals?|suggests?)\b',
        r'\b(requires?|needs?|demands?|depends?\s+on)\b',
        r'\b(leads?\s+to|results?\s+in|gives?\s+rise\s+to|creates?)\b',
        r'\b(precedes?|comes?\s+before|follows?|comes?\s+after)\b',
        r'\b(triggers?|initiates?|starts?|causes?)\b',
        r'\b(confirms?|validates?|verifies?)\b',
        r'\b(contradicts?|negates?|opposes?)\b',
        r'\b(opposite\s+of|vs\.?|versus)\b',
        r'\b(form\s+of|type\s+of|kind\s+of)\b',
        r'\b(and|with|plus|together\s+with)\b',
        r'\b(but|however|yet|although)\b',
        r'\b(because|since|as|due\s+to)\b',
        r'\b(if|when|whenever|once)\b',
        r'\b(see|look\s+at|notice|observe)\b',
        r'\b(like|similar\s+to|same\s+as)\b',
    ]

    for pattern in verb_patterns:
        for match in re.finditer(pattern, text_lower):
            # Get surrounding context
            start = max(0, match.start() - 60)
            end = min(len(text_lower), match.end() + 60)
            context = text_lower[start:end]

            # Check if any concept names appear in context
            concepts_in_context = [c for c in concept_names if c.lower() in context]
            if len(concepts_in_context) >= 2:
                verb = match.group(0)
                verbs[(verb, tuple(sorted(concepts_in_context[:3])))] += 1

    return verbs


def find_fabio_phrases(text: str) -> Counter:
    """Find Fabio's characteristic teaching phrases that link concepts."""
    phrases = Counter()

    # Fabio-specific patterns
    patterns = [
        r'\b(this\s+is\s+(?:what\s+)?(?:we\s+call|called))\b',
        r'\b(this\s+means?)\b',
        r'\b(what\s+this\s+means?)\b',
        r'\b(this\s+is\s+(?:the\s+)?(?:same\s+as|equivalent\s+to))\b',
        r'\b(this\s+is\s+(?:a|an)\s+(?:form\s+of|type\s+of))\b',
        r'\b(you\s+see)\b',
        r'\b(as\s+you\s+can\s+see)\b',
        r'\b(look\s+at)\b',
        r'\b(notice\s+(?:that|how))\b',
        r'\b(remember\s+(?:that)?)\b',
        r'\b(the\s+reason)\b',
        r'\b(because\s+(?:of)?)\b',
        r'\b(therefore)\b',
        r'\b(so\s+(?:what|when|if))\b',
        r'\b(which\s+means?)\b',
        r'\b(that\s+(?:is|means?))\b',
        r'\b(in\s+other\s+words)\b',
        r'\b(what\s+happens?)\b',
        r'\b(the\s+result)\b',
        r'\b(the\s+outcome)\b',
        r'\b(we\s+have)\b',
        r'\b(this\s+gives\s+us)\b',
        r'\b(this\s+is\s+where)\b',
        r'\b(this\s+is\s+why)\b',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text.lower()):
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 80)
            context = text[start:end].replace('\n', ' ')
            phrases[match.group(0)] += 1

    return phrases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", "-t", required=True)
    parser.add_argument("--ontology", "-o", required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    ontology = load_ontology(args.ontology)
    concept_names = [c["name"] for c in ontology.get("concepts", [])]

    with open(args.transcript, "r", encoding="utf-8") as f:
        text = f.read()

    print("=" * 70)
    print("RELATIONSHIP DIAGNOSTIC")
    print("=" * 70)
    print(f"Transcript length: {len(text)} chars")
    print(f"Ontology concepts: {len(concept_names)}")

    # 1. Find concept pairs in same sentences
    print("\n[1] CONCEPT PAIRS IN SAME SENTENCES")
    pairs = find_concept_pairs(text, concept_names)
    print(f"Found {len(pairs)} concept co-occurrences")
    pair_counts = Counter((a, b) for a, b, _ in pairs)
    for (a, b), count in pair_counts.most_common(args.max_examples):
        print(f"  {a} + {b}: {count} times")

    # 2. Find relationship verbs
    print("\n[2] RELATIONSHIP VERBS BETWEEN CONCEPTS")
    verbs = find_relationship_verbs(text, concept_names)
    for (verb, concepts), count in verbs.most_common(args.max_examples):
        print(f"  '{verb}' between {concepts}: {count} times")

    # 3. Fabio's teaching phrases
    print("\n[3] FABIO'S TEACHING PHRASES")
    phrases = find_fabio_phrases(text)
    for phrase, count in phrases.most_common(20):
        print(f"  '{phrase}': {count}")

    # 4. Sample sentences with multiple concepts
    print("\n[4] SAMPLE SENTENCES WITH MULTIPLE CONCEPTS")
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'\(])', text)
    examples = 0
    for sent in sentences:
        sent_lower = sent.lower()
        found = [c for c in concept_names if c.lower() in sent_lower]
        if len(found) >= 2 and examples < args.max_examples:
            print(f"  • {sent.strip()[:150]}...")
            print(f"    Concepts: {found}")
            examples += 1

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    if not pairs:
        print("No concept pairs found in same sentences.")
        print("→ Fabio may refer to concepts separately, not in connected phrases.")
    elif not verbs:
        print("Concepts co-occur but no explicit relationship verbs found.")
        print("→ Fabio may imply relationships through context, not explicit grammar.")
        print("→ Consider co-occurrence-based relationship inference instead of pattern matching.")
    else:
        print(f"Found {len(verbs)} potential relationship patterns.")
        print("→ Update RelationshipExtractor.PATTERNS with the verbs above.")


if __name__ == "__main__":
    main()