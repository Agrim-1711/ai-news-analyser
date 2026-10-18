import asyncio
import logging
import httpx
from fastapi import Depends, FastAPI, HTTPException, Query
from openai import APITimeoutError, OpenAIError, RateLimitError
from pydantic import ValidationError
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.database import Base, engine, get_db
from app.models import Analysis
from app.schemas import ArticleInput, ResearchRequest, ResearchResponse
from app.services.research import ResearchError, research_question
from app.services.ai import analyse_article
from app.services.news import search_articles

from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

Base.metadata.create_all(bind=engine)

app = FastAPI(title="NewsAnalyser")
logger = logging.getLogger(__name__)

@app.post("/api/research", response_model=ResearchResponse)
async def create_research(request: ResearchRequest):
    question = request.question.strip()
    if len(question) < 3:
        raise HTTPException(status_code=422, detail="Please enter a research question")
    try:
        return await research_question(question)
    except asyncio.TimeoutError:
        logger.warning("Research failed: overall_timeout")
        raise HTTPException(status_code=504, detail="Research timed out. Please try again")
    except APITimeoutError:
        logger.warning("Research failed: model_request_timeout")
        raise HTTPException(status_code=504, detail="The AI service timed out. Please try again")
    except RateLimitError:
        logger.warning("Research failed: model_rate_limited")
        raise HTTPException(status_code=503, detail="The AI service is busy or its usage limit was reached. Please try again later")
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="News provider returned an error")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Unable to reach news provider")
    except ResearchError as exc:
        logger.warning("Research failed: %s", exc.code)
        raise HTTPException(status_code=502, detail="Unable to produce a reliable research answer")
    except (OpenAIError, ValidationError) as exc:
        # Never log exception text: provider errors can contain request data.
        logger.warning("Research failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Unable to produce a reliable research answer")


app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/articles")
async def get_articles(
    q: str = Query(..., min_length=1, description="News search query")
):
    try:
        articles = await search_articles(q)
        return {"articles": articles}

    except httpx.HTTPStatusError:
        raise HTTPException(
            status_code=502,
            detail="News provider returned an error",
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Unable to reach news provider",
        )


@app.post("/api/analyses")
async def create_analysis(
    article: ArticleInput,
    db: Session = Depends(get_db),
):
    try:
        ai_analysis = await analyse_article(article)

        analysis = Analysis(
            title=article.title,
            description=article.description,
            url=str(article.url),
            source=article.source,
            published_at=article.published_at,
            summary=ai_analysis.summary,
            sentiment=ai_analysis.sentiment,
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return analysis

    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        raise HTTPException(
            status_code=502,
            detail="AI analysis failed",
        )

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to save analysis",
        )


@app.get("/api/analyses")
def get_analyses(db: Session = Depends(get_db)):
    return (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .limit(10)
        .all()
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )