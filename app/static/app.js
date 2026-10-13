const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const searchStatus = document.getElementById("search-status");
const searchResults = document.getElementById("search-results");
const analysisHistory = document.getElementById("analysis-history");

const researchForm = document.getElementById("research-form");
researchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = document.getElementById("research-input").value.trim();
    if (question.length < 3) return;
    const button = researchForm.querySelector("button");
    const status = document.getElementById("research-status");
    const result = document.getElementById("research-result");
    button.disabled = true;
    status.textContent = "Researching news sources…";
    result.replaceChildren();
    try {
        const response = await fetch("/api/research", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question}),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Research failed");
        const sources = new Map(data.sources.map(source => [source.id, source]));
        for (const finding of data.findings) {
            const paragraph = document.createElement("p");
            paragraph.textContent = finding.text + " ";
            for (const id of finding.source_ids) {
                const source = sources.get(id);
                if (!source) continue;
                const link = document.createElement("a");
                link.href = source.url;
                link.textContent = `[${id}] `;
                link.title = `${source.title} — ${source.source || "Unknown source"}`;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                paragraph.appendChild(link);
            }
            result.appendChild(paragraph);
        }
        const limitations = document.createElement("p");
        limitations.className = "status";
        limitations.textContent = data.limitations;
        result.appendChild(limitations);
        const searches = document.createElement("p");
        searches.className = "status";
        searches.textContent = "Searches used: " + data.searches.join(" → ");
        result.appendChild(searches);
        status.textContent = data.findings.length ? "Research complete." : "Insufficient evidence.";
    } catch (error) {
        status.textContent = error.message;
    } finally {
        button.disabled = false;
    }
});


searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const query = searchInput.value.trim();

    if (!query) {
        return;
    }

    searchStatus.textContent = "Searching...";
    searchResults.innerHTML = "";

    try {
        const response = await fetch(
            `/api/articles?q=${encodeURIComponent(query)}`
        );

        if (!response.ok) {
            throw new Error("Search failed");
        }

        const data = await response.json();

        renderArticles(data.articles);

        searchStatus.textContent =
            data.articles.length === 0
                ? "No articles found."
                : "";

    } catch (error) {
        searchStatus.textContent =
            "Unable to retrieve news. Please try again.";
    }
});

function renderArticles(articles) {
    searchResults.innerHTML = "";

    for (const article of articles) {
        const card = document.createElement("article");
        card.className = "card";

        const image = article.image
            ? `<img src="${escapeHtml(article.image)}" alt="">`
            : "";

        card.innerHTML = `
            ${image}

            <div class="card-content">
                <div class="metadata">
                    ${escapeHtml(article.source || "Unknown source")}
                    ·
                    ${formatDate(article.published_at)}
                </div>

                <h3>${escapeHtml(article.title)}</h3>

                <p>
                    ${escapeHtml(
                        article.description || "No description available."
                    )}
                </p>

                <div class="card-actions">
                    <a
                        href="${escapeHtml(article.url)}"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        Read article
                    </a>

                    <button class="analyse-button">
                        Analyse
                    </button>
                </div>

                <div class="analysis-result"></div>
            </div>
        `;

        const button = card.querySelector(".analyse-button");

        button.addEventListener("click", () => {
            analyseArticle(article, card, button);
        });

        searchResults.appendChild(card);
    }
}

async function analyseArticle(article, card, button) {
    const resultElement = card.querySelector(".analysis-result");

    button.disabled = true;
    button.textContent = "Analysing...";

    try {
        const response = await fetch("/api/analyses", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                title: article.title,
                description: article.description,
                url: article.url,
                source: article.source,
                published_at: article.published_at,
            }),
        });

        if (!response.ok) {
            throw new Error("Analysis failed");
        }

        const analysis = await response.json();

        resultElement.innerHTML = `
            <div class="analysis">
                <span class="sentiment ${analysis.sentiment}">
                    ${capitalise(analysis.sentiment)}
                </span>

                <h4>AI Summary</h4>

                <p>${escapeHtml(analysis.summary)}</p>
            </div>
        `;

        await loadAnalysisHistory();

        button.textContent = "Analysed";

    } catch (error) {
        resultElement.innerHTML = `
            <p class="error">
                Analysis failed. Please try again.
            </p>
        `;

        button.disabled = false;
        button.textContent = "Analyse";
    }
}

async function loadAnalysisHistory() {
    try {
        const response = await fetch("/api/analyses");

        if (!response.ok) {
            throw new Error("Unable to load history");
        }

        const analyses = await response.json();

        analysisHistory.innerHTML = "";

        if (analyses.length === 0) {
            analysisHistory.innerHTML =
                `<p class="status">No articles analysed yet.</p>`;
            return;
        }

        for (const analysis of analyses) {
            const card = document.createElement("article");
            card.className = "card analysis-history-card";

            card.innerHTML = `
                <div class="card-content">
                    <div class="metadata">
                        ${escapeHtml(analysis.source || "Unknown source")}
                        ·
                        ${formatDate(analysis.created_at)}
                    </div>

                    <h3>${escapeHtml(analysis.title)}</h3>

                    <span class="sentiment ${analysis.sentiment}">
                        ${capitalise(analysis.sentiment)}
                    </span>

                    <p>${escapeHtml(analysis.summary)}</p>

                    <a
                        href="${escapeHtml(analysis.url)}"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        Read original
                    </a>
                </div>
            `;

            analysisHistory.appendChild(card);
        }

    } catch (error) {
        analysisHistory.innerHTML =
            `<p class="error">Unable to load previous analyses.</p>`;
    }
}

loadAnalysisHistory();

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function formatDate(value) {
    if (!value) {
        return "";
    }

    return new Date(value).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
    });
}


function capitalise(value) {
    return value.charAt(0).toUpperCase() + value.slice(1);
}