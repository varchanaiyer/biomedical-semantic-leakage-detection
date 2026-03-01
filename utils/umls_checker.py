# utils/umls_checker.py — Validate concepts and relations using UMLS ontologies
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional, Sequence, Tuple
from dataclasses import dataclass

# Map UMLS semantic type *names* (as returned by the REST API) to broad buckets.
# The UMLS API returns semantic types as human-readable names like "Disease or Syndrome",
# NOT as TUI codes like "T047".  We also keep TUI codes for backward compatibility.
SEMTYPE_BUCKETS: Dict[str, str] = {
    # Disease / Condition
    "Disease or Syndrome": "Disease",
    "Neoplastic Process": "Disease",
    "Pathologic Function": "Disease",
    "Sign or Symptom": "Disease",
    "Congenital Abnormality": "Disease",
    "Acquired Abnormality": "Disease",
    "Injury or Poisoning": "Disease",
    "Mental or Behavioral Dysfunction": "Disease",
    "Cell or Molecular Dysfunction": "Disease",
    # Pharmacological
    "Pharmacologic Substance": "Pharmaco",
    "Antibiotic": "Pharmaco",
    "Immunologic Factor": "Pharmaco",
    "Biomedical or Dental Material": "Pharmaco",
    "Hormone": "Pharmaco",
    "Enzyme": "Pharmaco",
    "Vitamin": "Pharmaco",
    "Amino Acid, Peptide, or Protein": "Pharmaco",
    "Biologically Active Substance": "Pharmaco",
    "Organic Chemical": "Pharmaco",
    # Clinical Drug
    "Clinical Drug": "ClinicalDrug",
    # Anatomy
    "Body Part, Organ, or Organ Component": "Anatomy",
    "Tissue": "Anatomy",
    "Cell": "Anatomy",
    "Cell Component": "Anatomy",
    "Body System": "Anatomy",
    "Body Space or Junction": "Anatomy",
    # Physiology / Function
    "Organ or Tissue Function": "Physiology",
    "Physiologic Function": "Physiology",
    "Molecular Function": "Physiology",
    "Cell Function": "Physiology",
    "Genetic Function": "Physiology",
    "Mental Process": "Physiology",
    # Diagnostic / Lab
    "Laboratory Procedure": "Diagnostic",
    "Diagnostic Procedure": "Diagnostic",
    "Laboratory or Test Result": "Diagnostic",
    "Therapeutic or Preventive Procedure": "Procedure",
    # Backward-compat TUI codes
    "T047": "Disease",
    "T191": "Disease",
    "T046": "Disease",
    "T184": "Disease",
    "T019": "Disease",
    "T020": "Disease",
    "T037": "Disease",
    "T048": "Disease",
    "T049": "Disease",
    "T121": "Pharmaco",
    "T129": "Pharmaco",
    "T195": "Pharmaco",
    "T125": "Pharmaco",
    "T126": "Pharmaco",
    "T116": "Pharmaco",
    "T127": "Pharmaco",
    "T123": "Pharmaco",
    "T200": "ClinicalDrug",
    "T023": "Anatomy",
    "T024": "Anatomy",
    "T025": "Anatomy",
    "T026": "Anatomy",
    "T022": "Anatomy",
    "T030": "Anatomy",
    "T042": "Physiology",
    "T039": "Physiology",
    "T044": "Physiology",
    "T043": "Physiology",
    "T045": "Physiology",
    "T041": "Physiology",
    "T059": "Diagnostic",
    "T060": "Diagnostic",
    "T034": "Diagnostic",
    "T061": "Procedure",
}

# Keep old name for backward compat
TUI_BUCKETS = SEMTYPE_BUCKETS

# Allowed relation pairs between type buckets
ALLOWED_RELATIONS: Dict[Tuple[str, str], str] = {
    ("Pharmaco", "Disease"): "treats",
    ("ClinicalDrug", "Disease"): "treats",
    ("Pharmaco", "Pharmaco"): "interacts",
    ("Pharmaco", "Physiology"): "modulates",
    ("Pharmaco", "Anatomy"): "targets",
    ("Disease", "Anatomy"): "affects",
    ("Disease", "Physiology"): "disrupts",
    ("Diagnostic", "Disease"): "diagnoses",
    ("Procedure", "Disease"): "treats",
    ("Procedure", "Anatomy"): "targets",
}
SYMMETRIC_KEYS = {
    ("Pharmaco", "Pharmaco"),
    ("Disease", "Anatomy"),
    ("Pharmaco", "Disease"),
    ("Pharmaco", "Physiology"),
}


def _parse_semtype(st: Any) -> str:
    """Extract a semantic type string from either a dict or a plain string."""
    if isinstance(st, dict):
        return st.get("name") or st.get("value") or ""
    return str(st) if st else ""


def _best_bucket_from_stypes(semantic_types: List[str]) -> Optional[str]:
    """Find the best matching bucket for a list of semantic type strings."""
    for st in semantic_types:
        if st in SEMTYPE_BUCKETS:
            return SEMTYPE_BUCKETS[st]
    return None


@dataclass
class CheckerConfig:
    allowed_sources: Iterable[str] = ()
    main_sources: Iterable[str] = ()
    secondary_sources: Iterable[str] = ()
    allowed_tuis: Iterable[str] = ()
    min_score: float = 0.50
    enable_relation_check: bool = False
    require_main_source: bool = False
    ban_generic: bool = False
    upgrade_bioprocess_terms: bool = False
    allow_missing_score: bool = False

class ConceptRecord:
    """Internal wrapper for a concept dict for easier validation."""
    def __init__(self, cdict: Dict[str, Any]):
        self.text = cdict.get("text", "")
        self.cui = cdict.get("cui", "").strip()
        self.canonical = cdict.get("canonical", "")
        # Handle both dict and string formats for semantic_types
        raw_stypes = cdict.get("semantic_types") or []
        self.semantic_types: List[str] = [_parse_semtype(st) for st in raw_stypes]
        self.kb_sources = [s for s in (cdict.get("kb_sources") or [])]
        self.score: Optional[float] = None
        scores = cdict.get("scores") or {}
        if "confidence" in scores:
            self.score = float(scores["confidence"])
        elif "api" in scores:
            self.score = float(scores["api"])
        self.valid = False
        self.reasons: List[str] = []


class UMLSChecker:
    def __init__(self, config: CheckerConfig = CheckerConfig()):
        self.cfg = config

    def validate_concept(self, cdict: Dict[str, Any]) -> Dict[str, Any]:
        c = ConceptRecord(cdict)
        reasons: List[str] = []
        # Allowed source check
        src_ok = True
        if self.cfg.allowed_sources:
            if not any(src.upper() in self.cfg.allowed_sources for src in c.kb_sources):
                src_ok = False
                reasons.append("source not allowed")
        # Semantic type filter
        allowed_tui = True
        if self.cfg.allowed_tuis:
            matched = [t for t in c.semantic_types if t in SEMTYPE_BUCKETS or t in self.cfg.allowed_tuis]
            if not matched:
                allowed_tui = False
                reasons.append("semantic type not allowed")
        # Main source requirement
        if self.cfg.require_main_source:
            main_ok = any(src.upper() in (s.upper() for s in self.cfg.main_sources) for src in c.kb_sources)
            if not main_ok:
                reasons.append("missing main source")
        # Generic term ban
        if self.cfg.ban_generic and len(c.text.split()) <= 1:
            reasons.append("generic/uninformative span")
        # Bioprocess upgrade
        if self.cfg.upgrade_bioprocess_terms and "generic/uninformative span" in reasons:
            reasons = [r for r in reasons if r != "generic/uninformative span"]
        # Score threshold check
        score_ok = (c.score is None and self.cfg.allow_missing_score) or (c.score is not None and c.score >= self.cfg.min_score)

        # Check for recognized biomedical semantic type (if types are available)
        bucket = _best_bucket_from_stypes(c.semantic_types)
        has_biomedical_type = bucket is not None
        # The UMLS /search endpoint does NOT return semantic types — only /content/CUI does.
        # So semantic_types is usually empty.  Don't penalise when types are simply absent.
        if not has_biomedical_type and c.semantic_types:
            reasons.append("no recognized biomedical semantic type")
        types_ok = has_biomedical_type or (not c.semantic_types)  # empty types = OK

        c.valid = bool(score_ok and src_ok and allowed_tui and c.cui and types_ok)
        cdict_out = {**cdict}
        cdict_out["valid"] = c.valid
        cdict_out["reasons"] = sorted(set(reasons)) if not c.valid else ["ok"]
        # Add bucket info for downstream use
        cdict_out["bucket"] = bucket
        return cdict_out

    def validate_step_concepts(self, step_concepts: Iterable[Iterable[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = []
        for group in step_concepts:
            validated = [self.validate_concept(c) for c in (group or [])]
            out.append(validated)
        return out

    def _best_bucket(self, concept: Dict[str, Any]) -> Optional[str]:
        # Check pre-computed bucket first
        if concept.get("bucket"):
            return concept["bucket"]
        stypes = concept.get("semantic_types") or []
        parsed = [_parse_semtype(st) for st in stypes]
        return _best_bucket_from_stypes(parsed)

    def _pair_allowed(self, a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        ba = self._best_bucket(a)
        bb = self._best_bucket(b)
        if not ba or not bb:
            return False, None
        verb = ALLOWED_RELATIONS.get((ba, bb))
        if verb:
            return True, verb
        # symmetric support
        if (bb, ba) in ALLOWED_RELATIONS and (ba, bb) in SYMMETRIC_KEYS:
            return True, ALLOWED_RELATIONS[(bb, ba)]
        # try reverse
        verb_rev = ALLOWED_RELATIONS.get((bb, ba))
        if verb_rev and (bb, ba) in SYMMETRIC_KEYS:
            return True, verb_rev
        return False, None

    def validate_relations_adjacent(self, step_concepts: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not self.cfg.enable_relation_check:
            return []
        diagnostics: List[Dict[str, Any]] = []
        for s in range(len(step_concepts) - 1):
            left = step_concepts[s] or []
            right = step_concepts[s + 1] or []
            for ca in left:
                if not ca.get("valid"):
                    continue
                for cb in right:
                    if not cb.get("valid"):
                        continue
                    ok, verb = self._pair_allowed(ca, cb)
                    diagnostics.append({
                        "i": s, "j": s + 1,
                        "a_cui": ca.get("cui", ""),
                        "a_name": ca.get("canonical", ca.get("text", "")),
                        "a_bucket": self._best_bucket(ca),
                        "b_cui": cb.get("cui", ""),
                        "b_name": cb.get("canonical", cb.get("text", "")),
                        "b_bucket": self._best_bucket(cb),
                        "allowed": bool(ok),
                        "verb": (verb if ok else None),
                        "reason": (f"type-compatible: {verb}" if ok else "no supported relation between types"),
                    })
        return diagnostics

# Convenience top-level functions
_DEFAULT_CHECKER = UMLSChecker()

def validate_concepts(per_step_concepts: Sequence[Sequence[Dict[str, Any]]],
                      checker: Optional[UMLSChecker] = None) -> List[List[Dict[str, Any]]]:
    ch = checker or _DEFAULT_CHECKER
    return ch.validate_step_concepts(per_step_concepts)

def validate_step_concepts(per_step_concepts: Sequence[Sequence[Dict[str, Any]]],
                           checker: Optional[UMLSChecker] = None) -> List[List[Dict[str, Any]]]:
    return validate_concepts(per_step_concepts, checker)

def validate_relations(per_step_concepts: Sequence[Sequence[Dict[str, Any]]],
                       checker: Optional[UMLSChecker] = None) -> List[Dict[str, Any]]:
    ch = checker or _DEFAULT_CHECKER
    return ch.validate_relations_adjacent(per_step_concepts)

def make_checker(
    allowed_sources: Optional[Iterable[str]] = None,
    main_sources: Optional[Iterable[str]] = None,
    secondary_sources: Optional[Iterable[str]] = None,
    allowed_tuis: Optional[Iterable[str]] = None,
    min_score: Optional[float] = None,
    enable_relation_check: Optional[bool] = None,
    require_main_source: Optional[bool] = None,
    ban_generic: Optional[bool] = None,
    upgrade_bioprocess_terms: Optional[bool] = None,
) -> UMLSChecker:
    cfg = CheckerConfig()
    if allowed_sources is not None:
        cfg.allowed_sources = {s.upper() for s in allowed_sources}
    if main_sources is not None:
        cfg.main_sources = {s.upper() for s in main_sources}
    if secondary_sources is not None:
        cfg.secondary_sources = {s.upper() for s in secondary_sources}
    if allowed_tuis is not None:
        cfg.allowed_tuis = {str(t).upper() for t in allowed_tuis}
    if min_score is not None:
        cfg.min_score = float(min_score)
    if enable_relation_check is not None:
        cfg.enable_relation_check = bool(enable_relation_check)
    if require_main_source is not None:
        cfg.require_main_source = bool(require_main_source)
    if ban_generic is not None:
        cfg.ban_generic = bool(ban_generic)
    if upgrade_bioprocess_terms is not None:
        cfg.upgrade_bioprocess_terms = bool(upgrade_bioprocess_terms)
    return UMLSChecker(cfg)

def _best_bucket_for_concept(concept: Dict[str, Any]) -> Optional[str]:
    stypes = concept.get("semantic_types") or []
    parsed = [_parse_semtype(st) for st in stypes]
    return _best_bucket_from_stypes(parsed)

def has_supported_relation(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ba = _best_bucket_for_concept(a)
    bb = _best_bucket_for_concept(b)
    if not ba or not bb:
        return False
    if (ba, bb) in ALLOWED_RELATIONS:
        return True
    if (bb, ba) in ALLOWED_RELATIONS and (ba, bb) in SYMMETRIC_KEYS:
        return True
    if (bb, ba) in ALLOWED_RELATIONS and (bb, ba) in SYMMETRIC_KEYS:
        return True
    return False

def provisional_support(concept_a: Dict[str, Any], concept_b: Dict[str, Any]) -> Dict[str, Any]:
    return {"allowed": False, "evidence": []}
