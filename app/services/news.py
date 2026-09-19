import os

import httpx
from dotenv import load_dotenv

load_dotenv()

GNEWS_API_URL = "https://gnews.io/api/v4/search"
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")


async def search_articles(query: str) -> list[dict]:
    params = {
        "q": query,
        "lang": "en",
        "max": 10,
        "apikey": GNEWS_API_KEY,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            GNEWS_API_URL,
            params=params,
            timeout=10.0,
        )
        response.raise_for_status()

    data = response.json()

    return [
        {
            "title": article["title"],
            "description": article.get("description"),
            "url": article["url"],
            "image": article.get("image"),
            "published_at": article.get("publishedAt"),
            "source": article.get("source", {}).get("name"),
        }
        for article in data.get("articles", [])
    ]