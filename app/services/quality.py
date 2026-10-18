"""Mechanical checks complement model-assessed relevance; they do not prove meaning."""
import re

from app.schemas import ResearchCoverage, ResearchDraft, ResearchSource, SourceReview


class AnswerQualityError(ValueError):
    pass


def select_sources(review: SourceReview, sources: dict[str, ResearchSource]) -> dict[str, ResearchSource]:
    assessed = [item.source_id for item in review.assessments]
    if len(assessed) != len(set(assessed)) or set(assessed) != set(sources):
        raise AnswerQualityError("Source review must assess every known source exactly once")
    return {item.source_id: sources[item.source_id]
            for item in review.assessments if item.relevance == "direct"}


def validate_synthesis(draft: ResearchDraft, sources: dict[str, ResearchSource]) -> None:
    seen = set()
    for finding in draft.findings:
        text = " ".join(finding.text.split()).casefold()
        identity = text.rstrip(".!?")
        if identity in seen:
            raise AnswerQualityError("Findings repeat the same claim")
        seen.add(identity)
        if text.endswith("?"):
            raise AnswerQualityError("A finding must answer the question, not repeat a headline question")
        # Short factual statements can naturally match their source. Only reject
        # long verbatim descriptions; this is a narrow heuristic, not semantic grading.
        if len(text.split()) >= 20:
            for source in sources.values():
                description = " ".join((source.description or "").split()).casefold()
                if description and text == description:
                    raise AnswerQualityError("A finding copies a long article description instead of synthesising")


def coverage_for(sources: dict[str, ResearchSource], selected: dict[str, ResearchSource],
                 cited_ids: set[str]) -> ResearchCoverage:
    cited = [selected[sid] for sid in cited_ids]
    publishers = {(source.source or source.url.host or "").strip().casefold()
                  for source in cited}
    publishers.discard("")
    return ResearchCoverage(retrieved_articles=len(sources), selected_articles=len(selected),
                            cited_articles=len(cited), publisher_count=len(publishers))


def format_limitations(notes: str, cited_count: int, search_limit_reached: bool,
                       warnings: list[str]) -> str:
    """Preserve case-specific caveats and avoid appending repeated boilerplate."""
    sentences = re.split(r"(?<=[.!?])\s+", notes.strip()) if notes.strip() else []
    unique = []
    seen = set()
    for sentence in sentences:
        key = " ".join(sentence.casefold().split()).rstrip(".!?")
        if key not in seen:
            unique.append(sentence)
            seen.add(key)
    text = " ".join(unique)
    lowered = text.casefold()
    additions = []
    if not ("title" in lowered and "description" in lowered):
        additions.append("Evidence is limited to news titles and descriptions.")
    if cited_count < 2 and not any(phrase in lowered for phrase in (
        "coverage is narrow", "single source", "single article", "one source", "one article",
        "no findings", "no relevant", "insufficient evidence",
    )):
        additions.append("Coverage is narrow: fewer than two articles support this answer.")
    if search_limit_reached and not ("search" in lowered and any(
        phrase in lowered for phrase in ("limit", "budget")
    )):
        additions.append("The search limit was reached.")
    for warning in dict.fromkeys(warnings):
        if warning.casefold() not in lowered:
            additions.append(warning)
    return " ".join([text] + additions).strip()
