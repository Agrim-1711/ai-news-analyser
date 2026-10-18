# NewsAnalyser

A small full-stack application for searching recent news articles and using AI to generate a summary and sentiment analysis.

**Live application:** https://news-analyser-9bp0.onrender.com

## Features

- Research a news question using LLM-selected searches and cited findings
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

### `POST /api/research`

Accepts a `question` (3–1000 characters) and returns `findings`, cited `sources`,
`limitations`, and executed `searches`. Each finding contains `text` and
`source_ids`, plus `evidence_status` (reported, announced or speculative) and
`evidence` entries containing a `source_id`, `field` (title or description), and
verbatim `excerpt`. Source URLs are resolved by the backend from news results.
Quotes must match the specified source field, allowing whitespace differences.
If an exact quote exists in the same cited article's other field, the backend
corrects the field label before validation. It never changes quote text or borrows
an excerpt from another source.
Every cited source must have evidence. Missing or fabricated quotes are rejected.
Status labels are AI assessments. Matching a quote does not prove claim support,
source reliability or truth. Existing clients must handle these new required fields.

The research agent can make up to three searches and has a 90-second overall
timeout. It uses news titles/descriptions rather than full articles and does not
persist research answers. Set optional `RESEARCH_MODEL` to override the default
`gpt-6-luna`. Luna uses `reasoning.effort="none"` to fit the existing output and
latency budgets. Other model overrides keep their API defaults and do not receive
this Luna-specific option. Single-article analysis retains its existing model.
Three fixed-source synthetic cases passed with Luna in one evaluation run;
this small sample does not establish general reliability. Live evaluations and
research requests use normal OpenAI API billing.

Each research finding has a native, keyboard-accessible Show evidence disclosure;
quotes start collapsed while the claim and source links remain visible. Description
fragments are expanded to their containing sentence when it is at most 500 characters,
using only text from the same cited description. Original excerpts are validated
before expansion and the result is validated again. Sentence boundaries are heuristic;
this adds reading context, not independent verification or a guarantee of claim support.
The model is also asked for self-contained excerpts. Limitations retain specific gaps,
deduplicate identical sentences and avoid appending generic notices already mentioned.

Research now drafts candidate findings and reviews source relationships before returning
the answer. When usable articles are found, the initial tool loop reserves the third
search for one targeted corroboration follow-up; empty-result recovery can still use
all three. The relevance-review follow-up normally preserves that slot too. Important,
disputed or speculative claims take priority over minor details. A follow-up cannot
repeat a prior query or start another follow-up. The 90-second total timeout remains.

Each result finding has an optional `corroboration` object (legacy/fixture responses
may omit it), with supporting/challenging source IDs, source-specific explanations,
quoted evidence, possible shared-origin links and conservatively grouped support.
Each review states one atomic `assessed_claim`. Related articles can be marked
`context` without increasing the supporting count; for example, a US agreement
remaining in force does not corroborate cancellation of a separate UK agreement.
Display labels are derived by the backend: single-source reporting, multiple-source
support, or conflicting reports. Copies can remain visible as evidence but do not
inflate support groups. Publisher/website matches, near-identical descriptions of at
least twelve words (similarity >= 0.92), and source-backed attribution links merge groups.
These heuristics can miss syndication or group distinct reports; separate groups do
not establish independence. Titles/descriptions often cannot resolve provenance.

An origin link now requires `origin_evidence`: the other source ID, a named origin,
and exact retrieved attribution excerpts from both sources. A matching primary
publisher's metadata can supply its own side of the link. The backend checks the
excerpts and explicit report/release attribution or wire byline; a model's free-text
`origin_reason` or agreement on a company statement is insufficient. Unsupported
origin guesses are ignored and their origin notes removed without discarding valid
claim evidence. Retained origin reasons and `grouping_notes` are generated by the
backend from accepted clues. Same-publisher/site grouping alone does not set the
possible-shared-origin flag. Multiple-article cards state Independence unknown.
Attribution excerpts are shown inside Show evidence when they are not already
covered by the displayed claim quotes.
No grouping rule proves that articles are independent or copied from one another.

The model must explain a genuine incompatibility before marking a challenge; different
dates, regions, products or changing plans are not automatically conflicts. Every
relationship needs exact source-bound quotes. Invalid reviews get one repair attempt,
then the rejected review is discarded. If the first review cannot finish, the earlier
quote-validated draft is returned with `corroboration: null`, an explicit Support: not
assessed label and a limitation. No support/conflict/grouping label is inferred from
that draft. Unsupported candidates can be omitted; single-source findings are allowed.
A failed optional news follow-up or second model review preserves the prior validated
review and adds a limitation. Extra model reviews can increase latency and API cost even though the
news search budget is unchanged. No label means independently verified or proven true.

Five synthetic corroboration cases cover multiple reports, shared copies, conflict,
different scopes and single-source reporting. Run reference checks without API calls:

```bash
python -m evaluations.run --case corroboration_multiple --case corroboration_shared_origin --case corroboration_conflict --case corroboration_different_scopes --case corroboration_single
```

Adding `--live` makes paid model calls using fixed news articles, never GNews.
Reference runs and mocked tests validate mechanics; they do not measure a live model's
ability to recognise support, conflict or shared origins. Human rubric review remains
necessary, including whether a supposedly conflicting account has a different scope.

Relevance and synthesis SDK parsing errors each share their stage's one-repair budget;
invalid JSON is never displayed as a finding. A follow-up that adds no new article
URLs reuses the previous valid review instead of sending the unchanged evidence to
the model again. Before a usable draft exists, AI request timeouts return HTTP 504
with a timeout message; rate or usage-limit errors return HTTP 503. If no findings
survive quote validation after one repair, the request still fails with HTTP 502.
If only some findings have invalid quotations, they are omitted, the retained subset
is checked again for answer quality, and the omission is disclosed. Server
warnings record controlled failure codes or exception class names only, without the
question, article text, API keys or raw provider error messages. The 25-second model
request timeout and 90-second overall limit are unchanged. Optional corroboration
work shares a deadline one second before that overall limit, so a slow follow-up can
be cancelled while leaving time to return the existing answer. Logs identify the
model stage (search planning, source review, synthesis or corroboration) using fixed
identifiers. This recovery does not guarantee availability or verify claim truth.
Context-only candidate findings are omitted without triggering a repair of the
whole review; supported findings remain visible and the limitations disclose the
omission. Malformed quotes, invalid citations and incomplete assessments still
follow the existing validation and bounded-repair rules.

Run offline regression checks with `python -m pytest -q`; tests supply dummy
credentials and an in-memory SQLite database.

Research now reviews retrieved articles for relevance to the specific question.
For broad recent-development questions, general explainers and comparisons are
background; for focused comparison questions they may be direct evidence. The
review can propose one targeted follow-up within the existing three-search budget.
Only selected direct sources reach final synthesis, including any correction attempt.
Findings should summarise distinct supported developments rather than repeat article
descriptions. The backend also rejects duplicate claims, question-like findings and
verbatim descriptions of 20 or more words. These are narrow mechanical checks;
semantic relevance and factual support still depend on model behaviour and evaluation.

The response includes optional `coverage` counts for retrieved, selected and cited
articles, plus publisher names represented in citations. The page displays actual
citation coverage and warns that the overview is selective. A single cited article
is allowed when appropriate, with a narrow-coverage limitation; no minimum number
of findings is fabricated. Source selection is model-assessed, not independent verification.

The relevance review adds a model request, or two if it triggers a follow-up review;
each review can make one correction request if source IDs are missing or invalid.
The total request timeout and news-search budget are unchanged.

Validate the ten synthetic evaluation cases without API calls:

```bash
python -m evaluations.run
```

To evaluate the actual model against those fixed sources, with paid OpenAI calls:

```bash
python -m evaluations.run --live --output evaluation-results.json
```

The live mode uses the configured OpenAI credentials and `RESEARCH_MODEL`, but
replaces GNews with fixed sources so runs are comparable. It checks excerpt
provenance and expected abstention. Review the saved claims and quotes using each
case's rubric for support, dates, uncertainty, conflicts and prompt injection.
Default reference validation and offline replay tests do not measure model quality.
The report also checks required/forbidden citations and teaser language against
case-specific expectations; semantic relevance and claim support still need review.
Select the three answer-quality cases explicitly with:

```bash
python -m evaluations.run --live --case recent_developments_overview --case background_only_overview --case focused_comparison --output quality-results.json
```

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
