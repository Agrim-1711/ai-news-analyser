from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, HttpUrl


class ArticleInput(BaseModel):
    title: str
    description: Optional[str] = None
    url: HttpUrl
    source: Optional[str] = None
    published_at: Optional[datetime] = None


class AIAnalysis(BaseModel):
    summary: str
    sentiment: Literal["positive", "neutral", "negative"]