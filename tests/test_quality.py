import pytest

from app.schemas import ResearchDraft, ResearchFinding, ResearchSource, SourceReview
from app.services.quality import (
    AnswerQualityError, coverage_for, format_limitations, select_sources, validate_synthesis,
)


LONG_DESCRIPTION = ("Apple's M-series chips have transformed Mac performance, but Intel CPUs still "
                    "have advantages. Here's how the two compare in speed, graphics, and flexibility.")


def source(sid="S1", publisher="Example News"):
    return ResearchSource(id=sid, title="A chip comparison", description=LONG_DESCRIPTION,
                          source=publisher, url=f"https://example.com/{sid}")


def answer(text):
    return ResearchDraft(findings=[ResearchFinding(
        text=text, source_ids=["S1"], evidence_status="reported", evidence=[{
            "source_id": "S1", "field": "title", "excerpt": "A chip comparison",
        }],
    )], limitations="Fixture.")


def review(items):
    return SourceReview(assessments=[{"source_id": sid, "relevance": relevance,
                                      "reason": "Fixture classification."} for sid, relevance in items])


def test_only_direct_sources_selected():
    sources = {f"S{i}": source(f"S{i}") for i in range(1, 5)}
    result = select_sources(review([
        ("S1", "background"), ("S2", "direct"), ("S3", "irrelevant"), ("S4", "duplicate"),
    ]), sources)
    assert list(result) == ["S2"]


@pytest.mark.parametrize("items", [[], [("S99", "direct")], [("S1", "direct"), ("S1", "background")]])
def test_missing_unknown_or_duplicate_assessments_rejected(items):
    with pytest.raises(AnswerQualityError, match="every known source"):
        select_sources(review(items), {"S1": source()})


def test_no_direct_sources_is_valid():
    assert select_sources(review([("S1", "background")]), {"S1": source()}) == {}


def test_copied_long_description_rejected():
    with pytest.raises(AnswerQualityError, match="copies a long article description"):
        validate_synthesis(answer(LONG_DESCRIPTION), {"S1": source()})


def test_short_matching_fact_is_allowed():
    article = source().model_copy(update={"description": "Apple announced a chip."})
    validate_synthesis(answer("Apple announced a chip."), {"S1": article})


def test_question_like_finding_rejected():
    with pytest.raises(AnswerQualityError, match="answer the question"):
        validate_synthesis(answer("Are these chips faster?"), {"S1": source()})


def test_repeated_claim_rejected():
    draft = answer("A new chip was announced.")
    draft.findings.append(draft.findings[0].model_copy(update={"text": "A NEW CHIP WAS ANNOUNCED"}))
    with pytest.raises(AnswerQualityError, match="repeat the same claim"):
        validate_synthesis(draft, {"S1": source()})


def test_coverage_counts_cited_articles_and_publishers():
    all_sources = {"S1": source(), "S2": source("S2"), "S3": source("S3", "Another News")}
    coverage = coverage_for(all_sources, all_sources, {"S1", "S2"})
    assert coverage.retrieved_articles == 3
    assert coverage.selected_articles == 3
    assert coverage.cited_articles == 2
    assert coverage.publisher_count == 1


def test_checks_do_not_prove_semantic_relevance():
    # A paraphrased irrelevant claim can pass; source review and live evaluation matter.
    validate_synthesis(answer("This unrelated product seems interesting."), {"S1": source()})


def test_limitations_preserve_specific_gaps_and_add_generic_notices_once():
    notes = "No benchmark data was provided. No benchmark data was provided."
    result = format_limitations(notes, 1, True, ["Some results were excluded."] * 2)
    assert result.count("No benchmark data was provided.") == 1
    assert result.count("titles and descriptions") == 1
    assert result.count("Coverage is narrow") == 1
    assert result.count("search limit") == 1
    assert result.count("Some results were excluded.") == 1


def test_model_caveats_do_not_get_equivalent_app_notices_appended():
    notes = ("Only article titles and descriptions were available; benchmark details are missing. "
             "Only one source supports this answer. The search budget was exhausted.")
    assert format_limitations(notes, 1, True, []) == notes


def test_broad_coverage_does_not_get_narrow_or_search_limit_notices():
    result = format_limitations("No independent confirmation was available.", 3, False, [])
    assert "No independent confirmation" in result
    assert "titles and descriptions" in result
    assert "Coverage is narrow" not in result
    assert "search limit" not in result
