"""Compact model-only contract. Public responses still contain exact source quotes."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import CorroborationReview, ResearchEvidence
from app.services.quality import AnswerQualityError

MAX_REVIEW_FINDINGS = 5


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    stance: Literal["supports", "challenges", "context"]
    explanation: str = Field(min_length=1, max_length=240)
    evidence_ids: list[str] = Field(min_length=1, max_length=2)


class OriginSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    other_source_id: str
    origin_name: str = Field(min_length=3, max_length=150)
    evidence_ids: list[str] = Field(min_length=1, max_length=2)


class CompactFindingReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_index: int = Field(ge=0, lt=MAX_REVIEW_FINDINGS)
    assessed_claim: str = Field(min_length=1, max_length=1000)
    revised_text: str = Field(min_length=1, max_length=1000)
    evidence_status: Literal["reported", "announced", "speculative"]
    sources: list[EvidenceSelection] = Field(max_length=10)
    explanation: str = Field(min_length=1, max_length=400)
    conflict_explanation: Optional[str] = Field(default=None, min_length=1, max_length=400)
    origins: list[OriginSelection] = Field(default_factory=list, max_length=3)


class CompactCorroborationReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessments: list[CompactFindingReview] = Field(max_length=MAX_REVIEW_FINDINGS)
    follow_up_query: Optional[str] = Field(default=None, min_length=1, max_length=200)
    follow_up_reason: Optional[str] = Field(default=None, min_length=1, max_length=240)


def evidence_catalog(sources, draft=None):
    """Assign stable IDs to complete fields; never ask the model to transcribe them."""
    catalog, articles = {}, []
    for source in sources.values():
        passages = []
        for field, suffix in (("title", "T"), ("description", "D")):
            text = getattr(source, field)
            if not text or not text.strip():
                continue
            key = source.id + ":" + suffix
            catalog[key] = ResearchEvidence(source_id=source.id, field=field, excerpt=text)
            passages.append(dict(id=key, field=field, text=text))
        articles.append(dict(source_id=source.id, publisher=source.source,
                             published_at=source.published_at.isoformat() if source.published_at else None,
                             evidence=passages))
    # Preserve already-validated, self-contained draft excerpts without recopying
    # them in every candidate. These remain literal substrings of this source.
    by_id = {article["source_id"]: article for article in articles}
    for finding in draft.findings if draft else []:
        for quote in finding.evidence:
            source = sources.get(quote.source_id)
            if source is None or " ".join(quote.excerpt.split()) not in " ".join((getattr(source, quote.field) or "").split()):
                raise AnswerQualityError("Draft passage must match its source")
            if any(item == quote for item in catalog.values()):
                continue
            key = quote.source_id + ":Q" + str(len(by_id[quote.source_id]["evidence"]))
            catalog[key] = quote.model_copy(deep=True)
            by_id[quote.source_id]["evidence"].append(dict(id=key, field=quote.field, text=quote.excerpt))
    return catalog, articles


def expand_review(compact, catalog):
    """Resolve exact IDs from this request only, then reuse the strict public validator."""
    def quotes(ids, allowed):
        if len(ids) != len(set(ids)):
            raise AnswerQualityError("Duplicate evidence IDs")
        result = []
        for key in ids:
            item = catalog.get(key)
            if item is None or item.source_id not in allowed:
                raise AnswerQualityError("Unknown or wrong-source evidence ID")
            result.append(item.model_dump())
        return result

    assessments = []
    for item in compact.assessments:
        relations = []
        for relation in item.sources:
            relations.append(dict(source_id=relation.source_id, stance=relation.stance,
                                  explanation=relation.explanation,
                                  evidence=quotes(relation.evidence_ids, {relation.source_id}),
                                  shared_origin_with=[], origin_evidence=[]))
        by_id = {relation["source_id"]: relation for relation in relations}
        for origin in item.origins:
            if (origin.source_id not in by_id or origin.other_source_id not in by_id
                    or origin.source_id == origin.other_source_id):
                raise AnswerQualityError("Origin links must join two assessed sources")
            relation = by_id[origin.source_id]
            if origin.other_source_id not in relation["shared_origin_with"]:
                relation["shared_origin_with"].append(origin.other_source_id)
            relation["origin_evidence"].append(dict(
                other_source_id=origin.other_source_id, origin_name=origin.origin_name,
                evidence=quotes(origin.evidence_ids, {origin.source_id, origin.other_source_id}),
            ))
        assessments.append(dict(
            finding_index=item.finding_index, assessed_claim=item.assessed_claim,
            revised_text=item.revised_text, evidence_status=item.evidence_status,
            sources=relations, explanation=item.explanation,
            conflict_explanation=item.conflict_explanation,
        ))
    return CorroborationReview(assessments=assessments,
                               follow_up_query=compact.follow_up_query,
                               follow_up_reason=compact.follow_up_reason)
