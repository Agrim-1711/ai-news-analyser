from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, HttpUrl, Field, ConfigDict, field_validator


class ArticleInput(BaseModel):
    title: str
    description: Optional[str] = None
    url: HttpUrl
    source: Optional[str] = None
    published_at: Optional[datetime] = None


class AIAnalysis(BaseModel):
    summary: str
    sentiment: Literal["positive", "neutral", "negative"]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=200)


class ResearchEvidence(BaseModel):
    source_id: str
    field: Literal["title", "description"]
    excerpt: str = Field(min_length=1, max_length=2000,
                         description="A verbatim substring from the cited source field, never a paraphrase.")

    @field_validator("excerpt")
    @classmethod
    def reject_blank_excerpt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence excerpts cannot be blank")
        return value


class ResearchFinding(BaseModel):
    text: str = Field(min_length=1, max_length=2000, description=(
        "A concise factual answer to the user's question, written in your own words. "
        "No copied teaser text, rhetorical questions or conclusions absent from the evidence."
    ))
    source_ids: list[str] = Field(min_length=1, max_length=10)
    evidence_status: Literal["reported", "announced", "speculative"]
    evidence: list[ResearchEvidence] = Field(min_length=1, max_length=10)


class ResearchDraft(BaseModel):
    findings: list[ResearchFinding] = Field(max_length=8)
    limitations: str = Field(min_length=1, max_length=2000)


class ResearchSource(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    url: HttpUrl
    source: Optional[str] = None
    published_at: Optional[datetime] = None


class SourceAssessment(BaseModel):
    source_id: str
    relevance: Literal["direct", "background", "irrelevant", "duplicate"]
    reason: str = Field(min_length=1, max_length=240)


class SourceReview(BaseModel):
    assessments: list[SourceAssessment] = Field(max_length=30)
    follow_up_query: Optional[str] = Field(default=None, min_length=1, max_length=200)


class ResearchCoverage(BaseModel):
    retrieved_articles: int = Field(ge=0)
    selected_articles: int = Field(ge=0)
    cited_articles: int = Field(ge=0)
    publisher_count: int = Field(ge=0)


class SharedOriginEvidence(BaseModel):
    other_source_id: str
    origin_name: str = Field(min_length=3, max_length=150)
    evidence: list[ResearchEvidence] = Field(min_length=1, max_length=2)


class ClaimSourceAssessment(BaseModel):
    source_id: str
    stance: Literal["supports", "challenges", "context"]
    explanation: str = Field(min_length=1, max_length=500)
    evidence: list[ResearchEvidence] = Field(min_length=1, max_length=3)
    shared_origin_with: list[str] = Field(default_factory=list, max_length=10)
    origin_reason: Optional[str] = Field(default=None, min_length=1, max_length=500)
    origin_evidence: list[SharedOriginEvidence] = Field(default_factory=list, max_length=10)


class FindingAssessment(BaseModel):
    finding_index: int = Field(ge=0, le=7)
    assessed_claim: str = Field(min_length=1, max_length=2000)
    revised_text: str = Field(min_length=1, max_length=2000)
    evidence_status: Literal["reported", "announced", "speculative"]
    sources: list[ClaimSourceAssessment] = Field(max_length=10)
    explanation: str = Field(min_length=1, max_length=1000)
    conflict_explanation: Optional[str] = Field(default=None, min_length=1, max_length=1000)


class CorroborationReview(BaseModel):
    assessments: list[FindingAssessment] = Field(max_length=8)
    follow_up_query: Optional[str] = Field(default=None, min_length=1, max_length=200)
    follow_up_reason: Optional[str] = Field(default=None, min_length=1, max_length=500)


class ClaimSupport(BaseModel):
    assessed_claim: str
    status: Literal["single_source_reporting", "multiple_source_support", "conflicting_reports"]
    supporting_source_ids: list[str]
    challenging_source_ids: list[str]
    context_source_ids: list[str]
    supporting_groups: list[list[str]]
    possible_shared_origin: bool
    grouping_notes: list[str] = Field(default_factory=list)
    explanation: str
    conflict_explanation: Optional[str] = None
    source_assessments: list[ClaimSourceAssessment]


class ResearchResultFinding(ResearchFinding):
    corroboration: Optional[ClaimSupport] = None


class ResearchResponse(ResearchDraft):
    findings: list[ResearchResultFinding] = Field(max_length=8)
    sources: list[ResearchSource]
    searches: list[str]
    coverage: Optional[ResearchCoverage] = None
