"""Check quote provenance, not truth, entailment or source trustworthiness."""
from collections.abc import Iterable
import re

from app.schemas import ResearchDraft, ResearchSource


class EvidenceValidationError(ValueError):
    pass


def align_evidence_fields(draft: ResearchDraft, sources: Iterable[ResearchSource]) -> ResearchDraft:
    """Resolve a real quote's field within its cited source; never alter quote text."""
    by_id = {source.id: source for source in sources}
    aligned = draft.model_copy(deep=True)
    for finding in aligned.findings:
        for item in finding.evidence:
            source = by_id.get(item.source_id)
            if source is None:
                continue
            excerpt = " ".join(item.excerpt.split())
            if not excerpt:
                continue
            matches = [field for field in ("title", "description")
                       if excerpt in " ".join((getattr(source, field) or "").split())]
            if item.field not in matches and len(matches) == 1:
                item.field = matches[0]
    return aligned


def validate_evidence(draft: ResearchDraft, sources: Iterable[ResearchSource]) -> set[str]:
    by_id = {source.id: source for source in sources}
    cited_ids = set()
    for finding in draft.findings:
        finding_ids = set(finding.source_ids)
        if not finding_ids.issubset(by_id):
            raise EvidenceValidationError("Answer cites unknown sources")
        evidence_ids = {item.source_id for item in finding.evidence}
        if evidence_ids != finding_ids:
            raise EvidenceValidationError("Evidence must match every cited source")
        for item in finding.evidence:
            source_text = getattr(by_id[item.source_id], item.field) or ""
            # Accept layout differences, but never change words, case or punctuation.
            excerpt = " ".join(item.excerpt.split())
            available = " ".join(source_text.split())
            if not excerpt or excerpt not in available:
                raise EvidenceValidationError(
                    f"Evidence excerpt does not match {item.source_id}.{item.field}"
                )
        cited_ids.update(finding_ids)
    return cited_ids


def expand_description_excerpts(draft: ResearchDraft, sources: Iterable[ResearchSource]) -> ResearchDraft:
    """Add sentence context to real description fragments, without inventing words.

    Validate before calling: this changes presentation, not provenance requirements.
    Sentence splitting is a conservative heuristic, not a claim-support check.
    """
    by_id = {source.id: source for source in sources}
    expanded = draft.model_copy(deep=True)
    for finding in expanded.findings:
        for item in finding.evidence:
            if item.field != "description" or item.source_id not in by_id:
                continue
            description = " ".join((by_id[item.source_id].description or "").split())
            excerpt = " ".join(item.excerpt.split())
            if not excerpt or excerpt not in description:
                continue
            # Retain existing excerpts spanning sentences or very long sentences.
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", description):
                if excerpt in sentence and len(sentence) <= 500:
                    item.excerpt = sentence
                    break
    return expanded
