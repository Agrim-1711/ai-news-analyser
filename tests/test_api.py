import httpx
from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models import Analysis
from app.schemas import AIAnalysis
from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_articles_requires_query():
    response = client.get("/api/articles")

    assert response.status_code == 422


def test_articles_rejects_empty_query():
    response = client.get("/api/articles", params={"q": ""})

    assert response.status_code in (400, 422)


def test_create_analysis_requires_article():
    response = client.post("/api/analyses", json={})

    assert response.status_code == 422


def test_search_articles(monkeypatch):
    async def mock_search_articles(query):
        assert query == "artificial intelligence"

        return [
            {
                "title": "Test article",
                "description": "A test description",
                "url": "https://example.com/article",
                "image": None,
                "published_at": "2026-09-19T00:00:00Z",
                "source": "Test News",
            }
        ]

    monkeypatch.setattr(
        "app.main.search_articles",
        mock_search_articles,
    )

    response = client.get(
        "/api/articles",
        params={"q": "artificial intelligence"},
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["articles"]) == 1
    assert data["articles"][0]["title"] == "Test article"
    assert data["articles"][0]["source"] == "Test News"


def test_news_provider_connection_error(monkeypatch):
    async def mock_search_articles(query):
        request = httpx.Request(
            "GET",
            "https://example.com",
        )

        raise httpx.RequestError(
            "Connection failed",
            request=request,
        )

    monkeypatch.setattr(
        "app.main.search_articles",
        mock_search_articles,
    )

    response = client.get(
        "/api/articles",
        params={"q": "technology"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Unable to reach news provider"
    }


def test_news_provider_http_error(monkeypatch):
    async def mock_search_articles(query):
        request = httpx.Request(
            "GET",
            "https://example.com",
        )
        response = httpx.Response(
            500,
            request=request,
        )

        raise httpx.HTTPStatusError(
            "Server error",
            request=request,
            response=response,
        )

    monkeypatch.setattr(
        "app.main.search_articles",
        mock_search_articles,
    )

    response = client.get(
        "/api/articles",
        params={"q": "technology"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "News provider returned an error"
    }


def test_create_analysis_and_persist(monkeypatch):
    # Create an isolated in-memory database for this test.
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine,
    )

    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    async def mock_analyse_article(article):
        return AIAnalysis(
            summary="This is a test AI summary.",
            sentiment="positive",
        )

    app.dependency_overrides[get_db] = override_get_db

    monkeypatch.setattr(
        "app.main.analyse_article",
        mock_analyse_article,
    )

    try:
        response = client.post(
            "/api/analyses",
            json={
                "title": "AI company launches new product",
                "description": "The company announced a new AI product.",
                "url": "https://example.com/article",
                "source": "Test News",
                "published_at": "2026-09-19T00:00:00Z",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["title"] == "AI company launches new product"
        assert data["summary"] == "This is a test AI summary."
        assert data["sentiment"] == "positive"
        assert data["id"] == 1

        # Verify that it was actually persisted.
        db = TestSessionLocal()

        try:
            saved_analysis = db.query(Analysis).first()

            assert saved_analysis is not None
            assert saved_analysis.title == "AI company launches new product"
            assert saved_analysis.summary == "This is a test AI summary."
            assert saved_analysis.sentiment == "positive"
        finally:
            db.close()

        # Verify that persisted analyses can be retrieved via the API.
        history_response = client.get("/api/analyses")
        
        assert history_response.status_code == 200

        history = history_response.json()

        assert len(history) == 1
        assert history[0]["title"] == "AI company launches new product"
        assert history[0]["sentiment"] == "positive"

    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=test_engine)