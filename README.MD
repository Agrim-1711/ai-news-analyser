# NewsAnalyser

A small full-stack application for searching recent news articles and using AI to generate a summary and sentiment analysis.

**Live application:** https://news-analyser-9bp0.onrender.com

## Features

- Search for recent news articles by topic
- View article metadata including title, source, publication date and description
- Generate an AI summary and sentiment classification for an article
- Persist completed analyses in a PostgreSQL database
- View recent analyses when returning to the application
- Responsive frontend for desktop and mobile

## Tech Stack

### Backend

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL

### Frontend

- HTML
- CSS
- Vanilla JavaScript
- Jinja2

### External services

- GNews API for news search
- OpenAI API for article analysis
- Render for application and PostgreSQL hosting

## Architecture

The application uses a simple client-server architecture:

1. The frontend sends a search query to the FastAPI backend.
2. The backend requests recent articles from GNews.
3. Search results are returned to the frontend.
4. When the user chooses to analyse an article, its metadata is sent to the backend.
5. The backend sends the article information to OpenAI for summarisation and sentiment analysis.
6. The resulting analysis and article metadata are persisted using SQLAlchemy.
7. Previous analyses are retrieved from the database and displayed in the UI.

The backend separates external integrations into service modules:

```text
app/
├── main.py              # FastAPI routes and application setup
├── database.py          # Database configuration and sessions
├── models.py            # SQLAlchemy models
├── schemas.py           # Request/response models
├── services/
│   ├── ai.py            # OpenAI integration
│   └── news.py          # GNews integration
├── static/
│   ├── app.js           # Frontend behaviour
│   └── styles.css       # Styling
└── templates/
    └── index.html       # Main application page
```

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Agrim-1711/aries-news-analyser.git
cd aries-news-analyser
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GNEWS_API_KEY=your_gnews_api_key
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=sqlite:///./news_analyser.db
```

SQLite is used here for simple local development. The deployed application uses PostgreSQL.

### 5. Start the application

```bash
python -m uvicorn app.main:app --reload
```

Then visit:

```text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## API

### `GET /api/articles?q={query}`

Searches for recent news articles matching a query.

### `POST /api/analyses`

Generates an AI summary and sentiment classification for an article and persists the result.

Sentiment is classified as:

- `positive`
- `neutral`
- `negative`

### `GET /api/analyses`

Returns recent previously generated analyses.

### `GET /api/health`

Basic application health check.

## Design Decisions and Trade-offs

### Simple frontend

I used vanilla JavaScript and server-served HTML rather than introducing a frontend framework. The interface is small and interaction requirements are limited, so this keeps the application lightweight and avoids unnecessary build tooling.

### Database abstraction

SQLAlchemy allows the application to use SQLite for local development while using PostgreSQL in production without changing the application logic.

### Structured AI output

The AI integration requests a defined summary and sentiment structure rather than parsing free-form model output. This makes the API response predictable and easier to validate and display.

### Persistence

Analyses are stored server-side rather than only in browser state, allowing previous analyses to remain available between sessions. The production application uses PostgreSQL so this state persists independently of application instances.

### Scope

Given the time-boxed nature of the project, I prioritised a complete and reliable core workflow over additional features. In a production system I would consider adding:

- cursor-based pagination for analysis history
- duplicate-analysis handling or caching
- authentication and per-user history
- database migrations using Alembic
- rate limiting and more extensive request validation
- more comprehensive automated testing and observability

## Error Handling

The application handles failures from external services separately from database failures. Failed database transactions are rolled back before the session is reused, and API failures return appropriate HTTP error responses rather than exposing implementation details.

API keys and database credentials are supplied through environment variables and are not committed to the repository.

## Deployment

The application is deployed on Render.

The production architecture consists of:

- a Render Python web service running FastAPI/Uvicorn
- a Render PostgreSQL database
- GNews and OpenAI as external API services

The application binds to Render's provided `PORT` environment variable when deployed.

## Future Improvements

With more time, I would focus on:

1. Fetching or extracting fuller article content where permitted, allowing the AI analysis to use more than the news API metadata.
2. Adding database migrations rather than creating the schema directly at application startup.
3. Adding integration tests around external API failures and persistence.
4. Introducing pagination and filtering for larger analysis histories.
5. Adding caching to avoid repeated external API calls for identical requests.
