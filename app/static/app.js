const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const searchStatus = document.getElementById("search-status");
const searchResults = document.getElementById("search-results");
const analysisHistory = document.getElementById("analysis-history");

function appendResearchQuote(parent, evidence, source, prefix = "") {
    const quote = document.createElement("blockquote");
    const text = document.createElement("p");
    text.textContent = evidence.excerpt;
    const attribution = document.createElement("cite");
    attribution.textContent = `${prefix}[${evidence.source_id}] ${source.title}`
        + ` — ${source.source || "Unknown source"}`
        + (source.published_at ? ` · ${formatDate(source.published_at)}` : "")
        + ` (${evidence.field})`;
    quote.append(text, attribution);
    parent.appendChild(quote);
}

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
        if (data.coverage) {
            const coverage = document.createElement("p");
            coverage.className = "status";
            coverage.textContent = `Reviewed ${data.coverage.retrieved_articles} retrieved articles; `
                + `cited ${data.coverage.cited_articles} from ${data.coverage.publisher_count} publishers. `
                + "This is a selected overview, not complete coverage.";
            result.appendChild(coverage);
        }
        for (const finding of data.findings) {
            const card = document.createElement("article");
            card.className = "research-finding";
            const label = document.createElement("p");
            label.className = "evidence-status";
            label.textContent = "Evidence status (AI-assessed): " + finding.evidence_status;
            card.appendChild(label);
            const support = finding.corroboration;
            if (support) {
                const supportLabel = document.createElement("p");
                supportLabel.className = "support-status";
                supportLabel.dataset.status = support.status;
                const names = {
                    single_source_reporting: "Single-source reporting",
                    multiple_source_support: "Multiple-source support",
                    conflicting_reports: "Conflicting reports",
                };
                supportLabel.textContent = "Support (AI-assessed): " + names[support.status];
                card.appendChild(supportLabel);
                const counts = document.createElement("p");
                counts.className = "status support-counts";
                const supportingCount = support.supporting_source_ids.length;
                const groupCount = support.supporting_groups.length;
                const challengingCount = support.challenging_source_ids.length;
                const contextCount = support.context_source_ids.length;
                counts.textContent = `${supportingCount} supporting ${supportingCount === 1 ? "article" : "articles"} in `
                    + `${groupCount} ${groupCount === 1 ? "source group" : "source groups"}`
                    + (support.challenging_source_ids.length
                        ? `; ${challengingCount} challenging ${challengingCount === 1 ? "article" : "articles"}.` : ".")
                    + (contextCount ? ` ${contextCount} additional ${contextCount === 1 ? "article provides" : "articles provide"} context.` : "")
                    + (support.possible_shared_origin ? " Possible shared origin." : "");
                if (support.supporting_source_ids.length + support.challenging_source_ids.length
                    + support.context_source_ids.length > 1) {
                    counts.textContent += " Independence unknown.";
                }
                card.appendChild(counts);
                if (support.conflict_explanation) {
                    const conflict = document.createElement("p");
                    conflict.className = "conflict-explanation";
                    conflict.textContent = support.conflict_explanation;
                    card.appendChild(conflict);
                }
            }
            if (!support) {
                const unchecked = document.createElement("p");
                unchecked.className = "status";
                unchecked.textContent = "Support: not assessed — the additional source-support review could not be completed.";
                card.appendChild(unchecked);
            }
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
            card.appendChild(paragraph);
            const details = document.createElement("details");
            details.className = "research-evidence";
            const summary = document.createElement("summary");
            summary.textContent = "Show evidence";
            details.appendChild(summary);
            details.addEventListener("toggle", () => {
                summary.textContent = details.open ? "Hide evidence" : "Show evidence";
            });
            if (support) {
                const assessedClaim = document.createElement("p");
                assessedClaim.textContent = "Claim assessed: " + support.assessed_claim;
                details.appendChild(assessedClaim);
                const explanation = document.createElement("p");
                explanation.textContent = support.explanation
                    + " Reports from the same publisher or website, near copies, and reports with a possible common origin are grouped together. Different groups do not establish independence.";
                details.appendChild(explanation);
                for (const note of support.grouping_notes || []) {
                    const grouping = document.createElement("p");
                    grouping.textContent = note;
                    details.appendChild(grouping);
                }
            }
            const relations = new Map((support?.source_assessments || []).map(item => [item.source_id, item]));
            const explained = new Set();
            for (const evidence of finding.evidence) {
                const source = sources.get(evidence.source_id);
                const relationship = relations.get(evidence.source_id);
                if (relationship && !explained.has(evidence.source_id)) {
                    const explanation = document.createElement("p");
                    explanation.className = "source-relationship";
                    explanation.textContent = `[${evidence.source_id}] `
                        + ({supports: "Supports: ", challenges: "Challenges: ", context: "Context: "}[relationship.stance])
                        + relationship.explanation
                        + (relationship.origin_reason ? " Possible shared origin: " + relationship.origin_reason : "");
                    details.appendChild(explanation);
                    explained.add(evidence.source_id);
                }
                appendResearchQuote(details, evidence, source);
            }
            const shownOrigins = new Set();
            const normaliseQuote = text => text.replace(/\s+/g, " ").trim();
            for (const relationship of relations.values()) {
                for (const clue of relationship.origin_evidence || []) {
                    for (const evidence of clue.evidence) {
                        const key = JSON.stringify([evidence.source_id, evidence.field, normaliseQuote(evidence.excerpt)]);
                        const alreadyShown = finding.evidence.some(item => item.source_id === evidence.source_id
                            && item.field === evidence.field
                            && normaliseQuote(item.excerpt).includes(normaliseQuote(evidence.excerpt)));
                        if (alreadyShown || shownOrigins.has(key)) continue;
                        const source = sources.get(evidence.source_id);
                        if (!source) continue;
                        if (!shownOrigins.size) {
                            const heading = document.createElement("p");
                            heading.textContent = "Shared-origin attribution";
                            details.appendChild(heading);
                        }
                        appendResearchQuote(details, evidence, source, "Attribution ");
                        shownOrigins.add(key);
                    }
                }
            }
            card.appendChild(details);
            result.appendChild(card);
        }
        if (data.findings.length) {
            const note = document.createElement("p");
            note.className = "status";
            note.textContent = "Quotes match retrieved source text. Claim support and evidence status still need checking.";
            result.appendChild(note);
        }
        const limitations = document.createElement("p");
        limitations.className = "status";
        limitations.textContent = data.limitations;
        result.appendChild(limitations);
        const searches = document.createElement("p");
        searches.className = "status";
        searches.textContent = "Searches used: " + data.searches.join(" → ");
        result.appendChild(searches);
        status.textContent = data.findings.length
            ? (data.findings.some(finding => !finding.corroboration)
                ? "Research returned with limitations: source support was not assessed." : "Research complete.")
            : "Insufficient evidence.";
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
