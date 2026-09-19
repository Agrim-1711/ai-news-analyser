import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.schemas import AIAnalysis, ArticleInput

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def analyse_article(article: ArticleInput) -> AIAnalysis:
    article_text = f"""
Title: {article.title}
Description: {article.description or "No description available."}
"""

    response = await client.responses.parse(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "system",
                "content": (
                    "You analyse news articles. "
                    "Provide a concise 2-3 sentence summary based only on "
                    "the supplied information, and classify its overall "
                    "sentiment as positive, neutral, or negative."
                ),
            },
            {
                "role": "user",
                "content": article_text,
            },
        ],
        text_format=AIAnalysis,
    )

    return response.output_parsed