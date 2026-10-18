import asyncio

from evaluations.run import evaluate


def test_reference_evaluation_is_explicitly_not_live():
    report = asyncio.run(evaluate())
    assert report["mode"] == "reference_fixture_validation"
    assert report["model_quality_evaluated"] is False
    assert report["semantic_review_complete"] is False
    assert len(report["cases"]) == 15
    assert all(row["quote_provenance_pass"] for row in report["cases"])
    assert all(row["abstention_matches_reference"] for row in report["cases"])
    assert all(row["required_sources_cited"] for row in report["cases"])
    assert all(row["forbidden_sources_excluded"] for row in report["cases"])
    assert all(row["no_teaser_language"] for row in report["cases"])
    assert all(row["human_review"]["claim_support"] is None for row in report["cases"])


def test_evaluation_reports_errors_without_private_details(monkeypatch):
    def fail(*_):
        raise ValueError("private provider details")
    monkeypatch.setattr("evaluations.run.validate_evidence", fail)
    report = asyncio.run(evaluate())
    assert all(row["error_type"] == "ValueError" for row in report["cases"])
    assert "private" not in str(report)


def test_live_evaluation_uses_agent_and_fixed_sources(monkeypatch):
    from app.schemas import ResearchResponse
    from app.services import research

    async def fake_agent(question):
        # Exercise the runner wiring with the real injected news function.
        articles = await research.search_articles("fixture query")
        return ResearchResponse(findings=[], sources=[], searches=["fixture query"],
                                limitations=f"Received {len(articles)} fixed articles.")

    monkeypatch.setattr(research, "research_question", fake_agent)
    report = asyncio.run(evaluate(live=True))
    assert report["mode"] == "live_model_fixed_news"
    assert len(report["cases"]) == 15
    assert not any("error_type" in row for row in report["cases"])
    assert report["cases"][0]["result"]["limitations"] == "Received 1 fixed articles."
    assert report["cases"][0]["abstention_matches_reference"] is False


def test_evaluation_can_select_only_three_quality_cases():
    from evaluations.cases import QUALITY_CASE_IDS
    report = asyncio.run(evaluate(case_ids=QUALITY_CASE_IDS))
    assert [case["id"] for case in report["cases"]] == QUALITY_CASE_IDS


def test_live_report_detects_missing_relevant_sources_and_irrelevant_citations(monkeypatch):
    from app.schemas import ResearchResponse
    from app.services import research
    from evaluations.cases import APPLE_BACKGROUND, CASES, finding
    from evaluations.run import fixture_sources

    async def poor_answer(_):
        return ResearchResponse(findings=[finding(
            "Here's how they compare.", APPLE_BACKGROUND["description"], "reported",
        )], sources=fixture_sources(CASES[7]), searches=["Apple"], limitations="Fixture.")

    monkeypatch.setattr(research, "research_question", poor_answer)
    row = asyncio.run(evaluate(live=True, case_ids=["recent_developments_overview"]))["cases"][0]
    assert row["required_sources_cited"] is False
    assert row["forbidden_sources_excluded"] is False
    assert row["no_teaser_language"] is False


def test_live_report_records_controlled_error_code_not_raw_details(monkeypatch):
    from app.services import research

    async def fail(_):
        raise research.ResearchError("PRIVATE_PROVIDER_RESPONSE", code="source_review_invalid")

    monkeypatch.setattr(research, "research_question", fail)
    row = asyncio.run(evaluate(live=True, case_ids=["recent_developments_overview"]))["cases"][0]
    assert row["error_code"] == "source_review_invalid"
    assert "PRIVATE" not in str(row)


def test_curated_corroboration_references_are_explicitly_offline():
    from evaluations.corroboration_cases import CORROBORATION_CASE_IDS
    report = asyncio.run(evaluate(case_ids=CORROBORATION_CASE_IDS))
    assert len(report["cases"]) == 5
    assert report["model_quality_evaluated"] is False
    assert all(row["support_labels_match_reference"] for row in report["cases"])
    assert all(row["shared_origin_matches_reference"] for row in report["cases"])
    assert all(row["human_review"]["relationship_accuracy"] is None for row in report["cases"])


def test_schema_diagnostics_export_only_locations_and_types(monkeypatch):
    import pytest
    from pydantic import ValidationError
    from app.schemas import ResearchEvidence

    with pytest.raises(ValidationError) as caught:
        ResearchEvidence.model_validate({"source_id": "S1", "field": "PRIVATE_INPUT", "excerpt": "PRIVATE_EXCERPT"})

    def fail(*_):
        raise caught.value

    monkeypatch.setattr("evaluations.run.validate_evidence", fail)
    row = asyncio.run(evaluate(case_ids=["announcement"]))["cases"][0]
    assert row["schema_errors"] == [{"location": ["field"], "type": "literal_error"}]
    assert "PRIVATE" not in str(row)
