#!/usr/bin/env python3
"""
validate_ontology.py

Validates a Fabio ontology JSON file for structural integrity,
dangling references, and relationship consistency.

Usage:
    python validate_ontology.py fabio_ontology.json
    python validate_ontology.py fabio_ontology.json --fix --output fixed.json
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Set, Dict, List, Any


class OntologyValidator:
    def __init__(self, ontology: Dict[str, Any]):
        self.ontology = ontology
        self.concepts = {c["name"]: c for c in ontology.get("concepts", [])}
        self.concept_names = set(self.concepts.keys())
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Run all validation checks. Returns True if valid."""
        self._validate_concept_ids_unique()
        self._validate_concept_names_unique()
        self._validate_relationships()
        self._validate_parent_concepts()
        self._validate_related_concepts()
        self._validate_opposite_concepts()
        self._validate_decision_hierarchy()
        self._validate_extraction_rules()
        self._validate_orphan_concepts()
        self._validate_confidence_weights()
        return len(self.errors) == 0

    def _validate_concept_ids_unique(self):
        ids = [c.get("concept_id", "") for c in self.ontology.get("concepts", [])]
        seen = set()
        for cid in ids:
            if cid in seen:
                self.errors.append(f"Duplicate concept_id: {cid}")
            seen.add(cid)

    def _validate_concept_names_unique(self):
        names = [c.get("name", "") for c in self.ontology.get("concepts", [])]
        seen = set()
        for name in names:
            if name in seen:
                self.errors.append(f"Duplicate concept name: {name}")
            seen.add(name)

    def _validate_relationships(self):
        for i, rel in enumerate(self.ontology.get("relationships", [])):
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src not in self.concept_names:
                self.errors.append(f"Relationship[{i}] source not found: '{src}'")
            if tgt not in self.concept_names:
                self.errors.append(f"Relationship[{i}] target not found: '{tgt}'")
            if rel.get("strength", 0) < 0 or rel.get("strength", 0) > 1:
                self.warnings.append(
                    f"Relationship[{i}] strength out of range [0,1]: {rel.get('strength')}"
                )

    def _validate_parent_concepts(self):
        for c in self.ontology.get("concepts", []):
            for parent in c.get("parent_concepts", []):
                if parent not in self.concept_names:
                    self.errors.append(
                        f"Concept '{c['name']}' parent_concepts references missing: '{parent}'"
                    )

    def _validate_related_concepts(self):
        for c in self.ontology.get("concepts", []):
            for related in c.get("related_concepts", []):
                if related not in self.concept_names:
                    self.errors.append(
                        f"Concept '{c['name']}' related_concepts references missing: '{related}'"
                    )

    def _validate_opposite_concepts(self):
        for c in self.ontology.get("concepts", []):
            for opp in c.get("opposite_concepts", []):
                if opp not in self.concept_names:
                    self.errors.append(
                        f"Concept '{c['name']}' opposite_concepts references missing: '{opp}'"
                    )

    def _validate_decision_hierarchy(self):
        dh = self.ontology.get("decision_hierarchy", {})
        for step in dh.get("steps", []):
            for concept in step.get("concepts", []):
                if concept not in self.concept_names:
                    self.warnings.append(
                        f"Decision hierarchy references missing concept: '{concept}'"
                    )

    def _validate_extraction_rules(self):
        rules = self.ontology.get("extraction_rules", {})
        if rules.get("minimum_confidence", 0.5) < 0 or rules.get("minimum_confidence", 0.5) > 1:
            self.warnings.append("extraction_rules.minimum_confidence out of range [0,1]")
        if not rules.get("ngram_sizes"):
            self.warnings.append("extraction_rules.ngram_sizes is empty")

    def _validate_orphan_concepts(self):
        """Warn about concepts with no relationships at all."""
        related = set()
        for rel in self.ontology.get("relationships", []):
            related.add(rel["source"])
            related.add(rel["target"])
        for c in self.ontology.get("concepts", []):
            name = c["name"]
            if name not in related:
                has_refs = (
                    c.get("parent_concepts")
                    or c.get("related_concepts")
                    or c.get("opposite_concepts")
                )
                if not has_refs:
                    self.warnings.append(
                        f"Concept '{name}' is an orphan: no relationships or references"
                    )

    def _validate_confidence_weights(self):
        for c in self.ontology.get("concepts", []):
            cw = c.get("confidence_weight", 1.0)
            if cw < 0 or cw > 1:
                self.warnings.append(
                    f"Concept '{c['name']}' confidence_weight out of range [0,1]: {cw}"
                )

    def report(self):
        print("=" * 60)
        print(f"ONTOLOGY VALIDATION REPORT")
        print("=" * 60)
        print(f"Concepts:      {len(self.concept_names)}")
        print(f"Relationships: {len(self.ontology.get('relationships', []))}")
        print(f"Categories:    {len(self.ontology.get('categories', {}))}")
        print("-" * 60)
        if self.errors:
            print(f"ERRORS ({len(self.errors)}):")
            for e in self.errors:
                print(f"  [ERROR] {e}")
        else:
            print("  No errors found.")
        if self.warnings:
            print(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  [WARN]  {w}")
        else:
            print("  No warnings.")
        print("=" * 60)
        return len(self.errors) == 0

    def auto_fix(self) -> Dict[str, Any]:
        """Return a copy of the ontology with auto-fixes applied."""
        fixed = json.loads(json.dumps(self.ontology))
        concept_names = {c["name"] for c in fixed["concepts"]}

        # Fix dangling parent_concepts
        for c in fixed["concepts"]:
            c["parent_concepts"] = [
                p for p in c.get("parent_concepts", []) if p in concept_names
            ]
            c["related_concepts"] = [
                r for r in c.get("related_concepts", []) if r in concept_names
            ]
            c["opposite_concepts"] = [
                o for o in c.get("opposite_concepts", []) if o in concept_names
            ]

        # Fix dangling relationships
        fixed["relationships"] = [
            r for r in fixed.get("relationships", [])
            if r["source"] in concept_names and r["target"] in concept_names
        ]

        # Fix decision hierarchy
        for step in fixed.get("decision_hierarchy", {}).get("steps", []):
            step["concepts"] = [
                c for c in step.get("concepts", []) if c in concept_names
            ]

        return fixed


def main():
    parser = argparse.ArgumentParser(description="Validate Fabio ontology JSON")
    parser.add_argument("ontology", help="Path to ontology JSON file")
    parser.add_argument("--fix", action="store_true", help="Auto-fix and save")
    parser.add_argument("--output", "-o", help="Output path for fixed ontology")
    args = parser.parse_args()

    path = Path(args.ontology)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    validator = OntologyValidator(ontology)
    is_valid = validator.validate()
    validator.report()

    if args.fix:
        fixed = validator.auto_fix()
        out_path = args.output or path.with_stem(path.stem + "_fixed")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixed, f, indent=2, ensure_ascii=False)
        print(f"\nFixed ontology saved to: {out_path}")

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()