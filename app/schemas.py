from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, HttpUrl, Field, ConfigDict


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


class ResearchFinding(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    source_ids: list[str] = Field(min_length=1, max_length=10)


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


class ResearchResponse(ResearchDraft):
    sources: list[ResearchSource]
    searches: list[str]