import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from openai import APITimeoutError, AsyncOpenAI, OpenAIError, RateLimitError

from app.main import app
from app.schemas import (
    CorroborationReview, ResearchDraft, ResearchFinding, ResearchEvidence, SourceReview, SourceAssessment,
)
from app.services import research
from app.services.corroboration_contract import CompactCorroborationReview
from evaluations.cases import CASES

ARTICLE = {
    "title": "Nvidia announces a chip", "description": "A new chip was announced.",
    "url": "https://example.com/chip", "source": "Example News",
    "published_at": "2026-10-12T12:00:00Z",
}


def call(query="Nvidia", name="search_news", arguments=None):
    return SimpleNamespace(type="function_call", name=name, call_id="call_1",
                           arguments=arguments or json.dumps({"query": query}))


def response(*items, status="completed"):
    return SimpleNamespace(output=list(items), status=status)


def compact_fixture(review, data):
    """Adapt curated public-format answers to the new model-only fixture format."""
    passages = {source["source_id"]: source["evidence"] for source in data["sources"]}
    def refs(quotes):
        ids = []
        for quote in quotes:
            matches = [p["id"] for p in passages.get(quote.source_id, [])
                       if p["field"] == quote.field
                       and " ".join(quote.excerpt.split()) in " ".join(p["text"].split())]
            ids.append(matches[0] if matches else "UNKNOWN_EVIDENCE")
        return ids
    values = []
    for item in review.assessments:
        value = item.model_dump(exclude={"sources"})
        value["sources"] = [dict(source_id=r.source_id, stance=r.stance,
                                  explanation=r.explanation, evidence_ids=refs(r.evidence))
                            for r in item.sources]
        value["origins"] = [dict(source_id=r.source_id, other_source_id=clue.other_source_id,
                                  origin_name=clue.origin_name, evidence_ids=refs(clue.evidence))
                            for r in item.sources for clue in r.origin_evidence]
        values.append(value)
    return CompactCorroborationReview(assessments=values,
        follow_up_query=review.follow_up_query, follow_up_reason=review.follow_up_reason)


def setup_agent(monkeypatch, turns, articles=None, draft=None):
    responses = SimpleNamespace(
        create=AsyncMock(side_effect=turns),
        parse=AsyncMock(return_value=SimpleNamespace(status="completed", output_parsed=draft)),
    )
    async def default_review(**kwargs):
        sources = json.loads(kwargs["input"][0]["content"])["sources"]
        review = SourceReview(assessments=[SourceAssessment(
            source_id=s["id"], relevance="direct", reason="Relevant fixture evidence.",
        ) for s in sources])
        return SimpleNamespace(status="completed", output_parsed=review)

    responses.review = AsyncMock(side_effect=default_review)

    async def default_corroboration(**kwargs):
        data = json.loads(kwargs["input"][0]["content"])
        source_map = {s["source_id"]: s for s in data["sources"]}
        return SimpleNamespace(status="completed", output_parsed=CompactCorroborationReview(
            assessments=[{
                "finding_index": finding["finding_index"], "revised_text": finding["text"],
                "assessed_claim": finding["text"],
                "evidence_status": finding["evidence_status"], "explanation": "Fixture support review.",
                "sources": [{"source_id": sid, "stance": "supports", "explanation": "Fixture support.",
                             "evidence_ids": [key for key in finding["evidence_ids"] if key.startswith(sid + ":")]}
                            for sid in finding["source_ids"]],
            } for finding in data["candidate_findings"]],
        ))

    responses.corroborate = AsyncMock(side_effect=default_corroboration)

    async def parse_dispatch(**kwargs):
        if kwargs["text_format"] is SourceReview:
            return await responses.review(**kwargs)
        if kwargs["text_format"] is CompactCorroborationReview:
            result = await responses.corroborate(**kwargs)
            if isinstance(result.output_parsed, CorroborationReview):
                result = SimpleNamespace(status=result.status, output_parsed=compact_fixture(
                    result.output_parsed, json.loads(kwargs["input"][0]["content"])))
            return result
        return await responses.parse(**kwargs)

    wire = SimpleNamespace(create=responses.create, parse=parse_dispatch)
    fake = SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace(responses=wire))
    monkeypatch.setattr(research, "client", fake)
    search = AsyncMock(return_value=[ARTICLE] if articles is None else articles)
    monkeypatch.setattr(research, "search_articles", search)
    return responses, search


def answer(ids=None):
    return ResearchDraft(findings=[ResearchFinding(
        text="Nvidia announced a new chip.", source_ids=ids or ["S1"],
        evidence_status="announced", evidence=[ResearchEvidence(
            source_id=(ids or ["S1"])[0], field="description",
            excerpt="A new chip was announced.",
        )],
    )], limitations="Coverage is limited.")


def run():
    return asyncio.run(research.research_question("What happened at Nvidia?"))


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-4.1-nano"])
def test_model_override_options_apply_to_every_stage(monkeypatch, model):
    monkeypatch.setenv("RESEARCH_MODEL", model)
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    run()
    calls = (responses.create.await_args_list + responses.review.await_args_list
             + responses.parse.await_args_list + responses.corroborate.await_args_list)
    assert len(calls) == 5
    for request in calls:
        assert request.kwargs["model"] == model
        if model == "gpt-6-luna":
            assert request.kwargs["reasoning"] == {"effort": "none"}
        else:
            assert "reasoning" not in request.kwargs


def test_default_research_model_is_luna(monkeypatch):
    monkeypatch.delenv("RESEARCH_MODEL", raising=False)
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    run()
    for request in (responses.create.await_args_list + responses.review.await_args_list
                    + responses.parse.await_args_list + responses.corroborate.await_args_list):
        assert request.kwargs["model"] == "gpt-6-luna"
        assert request.kwargs["reasoning"] == {"effort": "none"}


def test_tool_output_roundtrip_and_citations(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    result = run()
    search.assert_awaited_once_with("Nvidia")
    assert str(result.sources[0].url) == ARTICLE["url"]
    assert result.findings[0].source_ids == ["S1"]
    inputs = responses.create.call_args.kwargs["input"]
    output = next(item for item in inputs if isinstance(item, dict) and item.get("type") == "function_call_output")
    assert output["call_id"] == "call_1"
    assert json.loads(output["output"])["articles"][0]["id"] == "S1"
    assert "descriptions" in result.limitations


def test_research_returns_contextual_quote_without_duplicate_limitations(monkeypatch):
    article = dict(ARTICLE, description="Microsoft said it blocked phishing as it disrupted YAM Store.")
    draft = answer()
    draft.findings[0].text = "Microsoft said it disrupted YAM Store."
    draft.findings[0].evidence[0].excerpt = "as it disrupted YAM Store."
    draft.limitations = "Only titles and descriptions were available. No operational details were given."
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], articles=[article], draft=draft)
    result = run()
    assert result.findings[0].evidence[0].excerpt == article["description"]
    assert result.limitations.count("titles and descriptions") == 1
    assert "No operational details were given." in result.limitations
    assert responses.parse.await_count == 1


def test_refines_search_and_deduplicates_sources(monkeypatch):
    _, search = setup_agent(monkeypatch, [response(call()), response(call("Nvidia chip")), response()], draft=answer())
    result = run()
    assert [c.args[0] for c in search.await_args_list] == ["Nvidia", "Nvidia chip"]
    assert len(result.sources) == 1


def test_initial_search_reserves_last_slot_for_corroboration(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call())] * 3, draft=answer())
    result = run()
    assert search.await_count == 2
    assert responses.create.await_count == 2
    assert "search limit" not in result.limitations


def test_no_evidence_abstains(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call())] * 3, articles=[])
    assert run().findings == []
    assert search.await_count == 3
    assert all(c.kwargs["tool_choice"] == "required" for c in responses.create.await_args_list)
    responses.parse.assert_not_awaited()


def test_empty_search_recovers_with_broader_query(monkeypatch):
    responses, search = setup_agent(monkeypatch, [
        response(call("recent developments affecting Nvidia")),
        response(call("Nvidia")), response(),
    ], draft=answer())
    search.side_effect = [[], [ARTICLE]]
    result = run()
    assert result.findings[0].source_ids == ["S1"]
    assert result.searches == ["recent developments affecting Nvidia", "Nvidia"]
    assert responses.create.await_args_list[1].kwargs["tool_choice"] == "required"
    messages = responses.create.call_args.kwargs["input"]
    tool_outputs = [m for m in messages if isinstance(m, dict) and m.get("type") == "function_call_output"]
    assert "broader query" in json.loads(tool_outputs[0]["output"])["guidance"]


def test_fabricated_citation_rejected(monkeypatch):
    setup_agent(monkeypatch, [response(call()), response()], draft=answer(["S99"]))
    with pytest.raises(research.ResearchError, match="unknown sources"):
        run()


def test_fabricated_excerpt_rejected_through_endpoint(monkeypatch):
    invalid = answer()
    invalid.findings[0].evidence[0].excerpt = "Invented supporting evidence."
    setup_agent(monkeypatch, [response(call()), response()], draft=invalid)
    result = TestClient(app).post("/api/research", json={"question": "Nvidia news?"})
    assert result.status_code == 502
    assert result.json() == {"detail": "Unable to produce a reliable research answer"}


def test_uncited_sources_not_returned(monkeypatch):
    setup_agent(monkeypatch, [response(call()), response()], draft=answer(), articles=[
        ARTICLE, {**ARTICLE, "url": "https://example.com/other"},
    ])
    result = run()
    assert [s.id for s in result.sources] == ["S1"]


def reviewed(*items, query=None):
    return SimpleNamespace(status="completed", output_parsed=SourceReview(
        assessments=[SourceAssessment(source_id=sid, relevance=kind, reason="Fixture.")
                     for sid, kind in items], follow_up_query=query,
    ))


def test_background_is_removed_from_synthesis_and_coverage_reported(monkeypatch):
    background = {**ARTICLE, "title": "Evergreen chip comparison",
                  "description": "BACKGROUND_SHOULD_NOT_REACH_SYNTHESIS", "url": "https://example.com/background"}
    valid = answer(["S2"])
    responses, _ = setup_agent(monkeypatch, [response(call()), response()],
                               articles=[background, ARTICLE], draft=valid)
    responses.review.side_effect = [reviewed(("S1", "background"), ("S2", "direct"))]
    result = run()
    assert [s.id for s in result.sources] == ["S2"]
    final_input = json.dumps(responses.parse.call_args.kwargs["input"])
    assert "BACKGROUND_SHOULD_NOT_REACH_SYNTHESIS" not in final_input
    assert "Evergreen chip comparison" not in final_input
    assert result.coverage.retrieved_articles == 2
    assert result.coverage.selected_articles == 1
    assert result.coverage.cited_articles == 1
    assert "Coverage is narrow" in result.limitations


def test_background_only_abstains_without_synthesis(monkeypatch):
    responses, _ = setup_agent(monkeypatch, [response(call()), response()])
    responses.review.side_effect = [reviewed(("S1", "background"))]
    result = run()
    assert result.findings == []
    assert "directly answered" in result.limitations
    responses.parse.assert_not_awaited()


def test_source_review_requests_one_targeted_followup(monkeypatch):
    background = {**ARTICLE, "url": "https://example.com/background", "title": "A generic comparison"}
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer(["S2"]))
    search.side_effect = [[background], [ARTICLE]]
    responses.review.side_effect = [
        reviewed(("S1", "background"), query="Nvidia chip launch"),
        reviewed(("S1", "background"), ("S2", "direct"), query="Nvidia earnings"),
    ]
    result = run()
    assert result.searches == ["Nvidia", "Nvidia chip launch"]
    assert search.await_count == 2
    assert responses.review.await_count == 2


def test_relevance_followup_leaves_reserved_corroboration_slot(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call())] * 3, draft=answer())
    responses.review.side_effect = [reviewed(("S1", "direct"), query="Nvidia earnings")]
    result = run()
    assert search.await_count == 2
    assert len(result.searches) == 2


def test_review_does_not_repeat_previous_query(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.review.side_effect = [reviewed(("S1", "direct"), query=" NVIDIA ")]
    run()
    search.assert_awaited_once()


def test_invalid_source_review_rejected_before_synthesis(monkeypatch):
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.review.side_effect = [reviewed(("S99", "direct"))] * 2
    with pytest.raises(research.ResearchError, match="every known source") as error:
        run()
    assert error.value.code == "source_review_invalid"
    responses.parse.assert_not_awaited()


def test_source_review_can_correct_a_missing_assessment(monkeypatch):
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.review.side_effect = [reviewed(), reviewed(("S1", "direct"))]
    result = run()
    assert result.findings
    assert responses.review.await_count == 2
    correction = json.loads(responses.review.await_args_list[1].kwargs["input"][-1]["content"])
    assert correction["required_source_ids"] == ["S1"]


@pytest.mark.parametrize("phase", ["review", "parse"])
@pytest.mark.parametrize("recover", [True, False])
def test_source_and_synthesis_sdk_json_errors_have_one_repair(monkeypatch, phase, recover):
    schema = SourceReview if phase == "review" else ResearchDraft
    with pytest.raises(ValidationError) as caught:
        schema.model_validate_json("PRIVATE_INVALID_JSON")
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    target = getattr(responses, phase)
    valid = reviewed(("S1", "direct")) if phase == "review" else SimpleNamespace(status="completed", output_parsed=answer())
    target.side_effect = [caught.value, valid if recover else caught.value]
    if recover:
        assert run().findings
    else:
        with pytest.raises(research.ResearchError) as failed:
            run()
        assert failed.value.code == ("source_review_invalid" if phase == "review" else "synthesis_invalid")
    assert target.await_count == 2
    assert "PRIVATE_INVALID_JSON" not in json.dumps(target.await_args_list[1].kwargs["input"])
    search.assert_awaited_once()


@pytest.mark.parametrize("followup_articles", [[], [ARTICLE]])
def test_relevance_review_not_repeated_without_new_evidence(monkeypatch, followup_articles):
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.review.side_effect = [reviewed(("S1", "direct"), query="Nvidia earnings")]
    search.side_effect = [[ARTICLE], followup_articles]
    assert run().findings
    assert search.await_count == 2
    responses.review.assert_awaited_once()


@pytest.mark.parametrize("followup_articles", [[], [ARTICLE]])
def test_corroboration_review_not_repeated_without_new_evidence(monkeypatch, followup_articles):
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.corroborate.side_effect = [corroborated(ARTICLE, query="Nvidia chip confirmation")]
    search.side_effect = [[ARTICLE], followup_articles]
    assert run().findings[0].corroboration.status == "single_source_reporting"
    assert search.await_count == 2
    responses.corroborate.assert_awaited_once()


def test_incomplete_source_review_rejected(monkeypatch):
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.review.side_effect = [SimpleNamespace(status="incomplete", output_parsed=None)]
    with pytest.raises(research.ResearchError) as error:
        run()
    assert error.value.code == "source_review_incomplete"
    responses.parse.assert_not_awaited()


def test_excluded_source_cannot_be_cited_even_during_repair(monkeypatch):
    background = {**ARTICLE, "url": "https://example.com/background", "description": "PRIVATE_BACKGROUND"}
    responses, _ = setup_agent(monkeypatch, [response(call()), response()],
                               articles=[background, ARTICLE], draft=answer(["S1"]))
    responses.review.side_effect = [reviewed(("S1", "background"), ("S2", "direct"))]
    with pytest.raises(research.ResearchError, match="unknown sources"):
        run()
    assert responses.parse.await_count == 2
    repair_input = json.dumps(responses.parse.call_args.kwargs["input"])
    assert "PRIVATE_BACKGROUND" not in repair_input


def test_long_copied_description_gets_corrected(monkeypatch):
    description = ("Apple's M-series chips have transformed Mac performance, but Intel CPUs still have "
                   "advantages. Here's how the two compare in speed, graphics, and flexibility.")
    article = {**ARTICLE, "description": description}
    invalid = answer()
    invalid.findings[0].text = description
    invalid.findings[0].evidence[0].excerpt = description
    corrected = answer()
    corrected.findings[0].text = "The report compares Apple and Intel chip performance."
    corrected.findings[0].evidence[0].excerpt = description
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], articles=[article])
    responses.parse.side_effect = [SimpleNamespace(status="completed", output_parsed=invalid),
                                   SimpleNamespace(status="completed", output_parsed=corrected)]
    result = run()
    assert result.findings[0].text == corrected.findings[0].text
    assert responses.parse.await_count == 2


def test_invalid_quote_is_repaired_once_without_more_searches(monkeypatch):
    invalid = answer()
    invalid.findings[0].evidence[0].field = "title"
    invalid.findings[0].evidence[0].excerpt = "Nvidia announces a chip with invented details."
    responses, search = setup_agent(monkeypatch, [response(call()), response()])
    responses.parse.side_effect = [
        SimpleNamespace(status="completed", output_parsed=invalid),
        SimpleNamespace(status="completed", output_parsed=answer()),
    ]
    result = run()
    assert result.findings[0].evidence[0].excerpt == ARTICLE["description"]
    assert responses.parse.await_count == 2
    search.assert_awaited_once()
    repair = json.loads(responses.parse.await_args_list[1].kwargs["input"][-1]["content"])
    assert "S1.title" in repair["validation_error"]
    assert repair["available_sources"][0]["title"] == ARTICLE["title"]
    assert "tools" not in responses.parse.await_args_list[1].kwargs


def test_repair_that_still_invents_quotes_is_rejected(monkeypatch):
    invalid = answer()
    invalid.findings[0].evidence[0].excerpt = "Not in any retrieved source."
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=invalid)
    with pytest.raises(research.ResearchError, match="does not match"):
        run()
    assert responses.parse.await_count == 2
    search.assert_awaited_once()


def test_valid_answer_needs_no_repair(monkeypatch):
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    run()
    responses.parse.assert_awaited_once()


def test_exact_quote_with_wrong_field_is_resolved_without_model_retry(monkeypatch):
    draft = answer()
    draft.findings[0].evidence[0].field = "title"
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=draft)
    result = run()
    assert result.findings[0].evidence[0].field == "description"
    responses.parse.assert_awaited_once()


def test_repair_may_abstain_if_evidence_is_insufficient(monkeypatch):
    invalid = answer()
    invalid.findings[0].evidence[0].excerpt = "Invented evidence."
    responses, _ = setup_agent(monkeypatch, [response(call()), response()])
    responses.parse.side_effect = [
        SimpleNamespace(status="completed", output_parsed=invalid),
        SimpleNamespace(status="completed", output_parsed=ResearchDraft(
            findings=[], limitations="Insufficient evidence for the proposed finding.",
        )),
    ]
    result = run()
    assert result.findings == []
    assert result.sources == []


@pytest.mark.parametrize("tool", [call(name="fetch_url"), call(arguments="not json"),
                                  call(query=" "), call(arguments='{"query":"Nvidia","extra":1}')])
def test_invalid_tool_never_executes(monkeypatch, tool):
    _, search = setup_agent(monkeypatch, [response(tool)])
    with pytest.raises(research.ResearchError):
        run()
    search.assert_not_awaited()


def test_multiple_calls_rejected(monkeypatch):
    _, search = setup_agent(monkeypatch, [response(call(), call())])
    with pytest.raises(research.ResearchError):
        run()
    search.assert_not_awaited()


def test_malformed_source_and_unsafe_url_excluded(monkeypatch):
    setup_agent(monkeypatch, [response(call()), response()],
                articles=[{}, {**ARTICLE, "url": "javascript:alert(1)"}])
    assert run().sources == []


def test_missing_final_answer_rejected(monkeypatch):
    setup_agent(monkeypatch, [response(call()), response()])
    with pytest.raises(research.ResearchError, match="structured answer"):
        run()


def test_incomplete_response_rejected(monkeypatch):
    setup_agent(monkeypatch, [response(status="incomplete")])
    with pytest.raises(research.ResearchError, match="Incomplete"):
        run()


def test_total_timeout(monkeypatch):
    async def slow(_):
        await asyncio.sleep(1)
    monkeypatch.setattr(research, "_research", slow)
    monkeypatch.setattr(research, "RESEARCH_TIMEOUT", 0.01)
    with pytest.raises(asyncio.TimeoutError):
        run()


@pytest.mark.parametrize("question", ["", "  ", "x" * 1001])
def test_question_validation(question):
    assert TestClient(app).post("/api/research", json={"question": question}).status_code == 422


@pytest.mark.parametrize("error, status", [
    (research.ResearchError("unknown citation"), 502),
    (OpenAIError("private provider details"), 502),
    (asyncio.TimeoutError(), 504),
    (APITimeoutError(request=httpx.Request("POST", "https://example.test")), 504),
    (RateLimitError("private provider error", response=httpx.Response(429,
                   request=httpx.Request("POST", "https://example.test")), body=None), 503),
    (httpx.RequestError("connection failed"), 503),
    (httpx.HTTPStatusError("failure", request=httpx.Request("GET", "https://example.com"),
                           response=httpx.Response(429)), 502),
])
def test_endpoint_errors(monkeypatch, error, status):
    monkeypatch.setattr("app.main.research_question", AsyncMock(side_effect=error))
    result = TestClient(app).post("/api/research", json={"question": "Nvidia news?"})
    assert result.status_code == status
    assert "private" not in result.text


def test_research_diagnostics_do_not_log_provider_details_or_question(monkeypatch, caplog):
    monkeypatch.setattr("app.main.research_question", AsyncMock(side_effect=OpenAIError("PRIVATE_PROVIDER_ERROR")))
    result = TestClient(app).post("/api/research", json={"question": "PRIVATE_QUESTION"})
    assert result.status_code == 502
    assert "OpenAIError" in caplog.text
    assert "PRIVATE" not in caplog.text


def test_endpoint_full_agent(monkeypatch):
    setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    result = TestClient(app).post("/api/research", json={"question": "Nvidia news?"})
    assert result.status_code == 200
    assert result.json()["sources"][0]["id"] == "S1"
    finding = result.json()["findings"][0]
    assert finding["evidence_status"] == "announced"
    assert finding["evidence"][0]["excerpt"] == ARTICLE["description"]
    assert finding["corroboration"]["status"] == "single_source_reporting"


def corroborated(*articles, query=None, conflict=False):
    return SimpleNamespace(status="completed", output_parsed=CorroborationReview(assessments=[{
        "finding_index": 0,
        "assessed_claim": "Nvidia announced a new chip.",
        "revised_text": "Reports disagree about the chip announcement." if conflict else "Nvidia announced a new chip.",
        "evidence_status": "reported" if conflict else "announced", "explanation": "Fixture review.",
        "conflict_explanation": "The same chip announcement is reported and denied." if conflict else None,
        "sources": [{"source_id": f"S{i}", "stance": "challenges" if conflict and i > 1 else "supports",
                     "explanation": "Fixture relationship.", "evidence": [{"source_id": f"S{i}",
                     "field": "description", "excerpt": article["description"]}]}
                    for i, article in enumerate(articles, start=1)],
    }], follow_up_query=query, follow_up_reason="Corroborate an important chip claim." if query else None))


def test_targeted_corroboration_uses_reserved_third_search_and_adds_support(monkeypatch):
    extra = dict(ARTICLE, url="https://second.example.test/chip", source="Second Publisher",
                 description="Nvidia has announced a new processor.")
    responses, search = setup_agent(monkeypatch, [response(call()), response(call("Nvidia chip"))], draft=answer())
    search.side_effect = [[ARTICLE], [ARTICLE], [extra]]
    responses.corroborate.side_effect = [corroborated(ARTICLE, query="Nvidia processor announcement"),
                                        corroborated(ARTICLE, extra, query="Must not execute")]
    result = run()
    assert result.searches == ["Nvidia", "Nvidia chip", "Nvidia processor announcement"]
    assert search.await_count == 3
    assert result.findings[0].corroboration.status == "multiple_source_support"
    assert len(result.sources) == 2
    assert responses.parse.await_count == 1
    final_input = json.loads(responses.corroborate.await_args_list[1].kwargs["input"][0]["content"])
    assert final_input["searches_remaining"] == 0


def test_followup_can_reveal_conflict_and_preserve_both_sources(monkeypatch):
    denial = dict(ARTICLE, url="https://denial.example.test/chip", source="Other News",
                  description="Nvidia denied announcing that chip.")
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    search.side_effect = [[ARTICLE], [denial]]
    responses.corroborate.side_effect = [corroborated(ARTICLE, query="Nvidia chip denial"),
                                        corroborated(ARTICLE, denial, conflict=True)]
    finding = run().findings[0]
    assert finding.corroboration.status == "conflicting_reports"
    assert finding.corroboration.challenging_source_ids == ["S2"]
    assert {quote.source_id for quote in finding.evidence} == {"S1", "S2"}


@pytest.mark.parametrize("reason", ["repeated", "budget_exhausted"])
def test_corroboration_does_not_repeat_queries_or_exceed_budget(monkeypatch, reason):
    if reason == "budget_exhausted":
        monkeypatch.setattr(research, "MAX_SEARCHES", 1)
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.corroborate.side_effect = [corroborated(ARTICLE, query=" nViDiA " if reason == "repeated" else "Nvidia chip")]
    result = run()
    assert search.await_count == 1
    assert result.findings[0].corroboration.status == "single_source_reporting"


def test_failed_optional_search_keeps_valid_single_source_answer(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    search.side_effect = [[ARTICLE], httpx.RequestError("PRIVATE_PROVIDER_DETAILS")]
    responses.corroborate.side_effect = [corroborated(ARTICLE, query="Nvidia chip confirmation")]
    result = run()
    assert result.findings[0].corroboration.status == "single_source_reporting"
    assert "corroboration search failed" in result.limitations
    assert "PRIVATE" not in result.model_dump_json()


def test_corroboration_repair_is_bounded_and_does_not_search(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    invalid = corroborated(ARTICLE)
    invalid.output_parsed.assessments[0].sources[0].evidence[0].excerpt = "Invented quote."
    responses.corroborate.side_effect = [invalid, corroborated(ARTICLE)]
    assert run().findings[0].corroboration.status == "single_source_reporting"
    assert responses.corroborate.await_count == 2
    search.assert_awaited_once()


def test_invalid_corroboration_is_never_displayed_after_one_repair(monkeypatch):
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    invalid = corroborated(ARTICLE)
    invalid.output_parsed.assessments[0].sources[0].source_id = "S99"
    responses.corroborate.side_effect = [invalid, invalid]
    result = TestClient(app).post("/api/research", json={"question": "Nvidia news?"})
    assert result.status_code == 200
    data = result.json()
    assert data["findings"][0]["corroboration"] is None
    assert data["findings"][0]["source_ids"] == ["S1"]
    assert data["findings"][0]["text"] == answer().findings[0].text
    assert "S99" not in result.text
    assert "Source support was not assessed" in data["limitations"]
    assert responses.corroborate.await_count == 2


def test_context_only_candidate_does_not_lose_other_valid_findings(monkeypatch):
    weak = dict(ARTICLE, title="An unsupported headline", description="An unrelated disclosure.",
                url="https://weak.example.test/report")
    draft = answer()
    other = answer(["S2"]).findings[0]
    other.text = "A headline mentions a possible partnership."
    other.evidence[0].excerpt = weak["description"]
    draft.findings.append(other)
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], articles=[ARTICLE, weak], draft=draft)
    checked = corroborated(ARTICLE).output_parsed
    context = checked.assessments[0].model_copy(deep=True)
    context.finding_index = 1
    context.assessed_claim = other.text
    context.revised_text = other.text
    context.sources[0].source_id = "S2"
    context.sources[0].stance = "context"
    context.sources[0].evidence[0].source_id = "S2"
    context.sources[0].evidence[0].excerpt = weak["description"]
    checked.assessments.append(context)
    responses.corroborate.side_effect = [SimpleNamespace(status="completed", output_parsed=checked)]
    result = run()
    assert len(result.findings) == 1
    assert result.findings[0].source_ids == ["S1"]
    assert "omitted" in result.limitations
    responses.corroborate.assert_awaited_once()


@pytest.mark.parametrize("recover", [True, False])
def test_sdk_parse_failure_uses_same_bounded_repair_without_private_inputs(monkeypatch, recover):
    with pytest.raises(ValidationError) as caught:
        CorroborationReview.model_validate({"assessments": [{"private": "PRIVATE_PROVIDER_INPUT"}]})
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.corroborate.side_effect = [caught.value, corroborated(ARTICLE) if recover else caught.value]
    if recover:
        assert run().findings[0].corroboration.status == "single_source_reporting"
    else:
        result = run()
        assert result.findings[0].corroboration is None
        assert "Source support was not assessed" in result.limitations
    assert responses.corroborate.await_count == 2
    assert "PRIVATE_PROVIDER_INPUT" not in json.dumps(responses.corroborate.await_args_list[1].kwargs["input"])
    search.assert_awaited_once()


def test_corroboration_never_gets_previously_excluded_background(monkeypatch):
    background = dict(ARTICLE, url="https://background.example.test/", description="PRIVATE_BACKGROUND")
    extra = dict(ARTICLE, url="https://new.example.test/", source="Other News")
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    search.side_effect = [[ARTICLE, background], [background, extra]]
    responses.review.side_effect = [reviewed(("S1", "direct"), ("S2", "background"))]
    # S3 is retrieved but not required to be cited if it adds nothing.
    responses.corroborate.side_effect = [corroborated(ARTICLE, query="Nvidia chip confirmation"), corroborated(ARTICLE)]
    run()
    assert "PRIVATE_BACKGROUND" not in json.dumps(responses.corroborate.await_args_list[1].kwargs["input"])


def test_existing_page_contains_both_workflows():
    page = TestClient(app).get("/")
    assert page.status_code == 200
    assert 'id="research-form"' in page.text
    assert 'id="search-form"' in page.text


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_fixed_evaluation_cases_through_agent(monkeypatch, case):
    # Replay curated answers to test orchestration/validation, not live model reasoning.
    reference = ResearchDraft(findings=case["reference_findings"],
                              limitations=case["reference_limitations"])
    responses, _ = setup_agent(monkeypatch, [response(call()), response()],
                              articles=case["articles"], draft=reference)
    expected = reference.findings
    if case.get("reference_corroboration"):
        from app.services.corroboration import apply_corroboration
        from evaluations.run import fixture_sources
        checked = CorroborationReview.model_validate(case["reference_corroboration"])
        responses.corroborate.side_effect = [SimpleNamespace(status="completed", output_parsed=checked)]
        expected = apply_corroboration(reference, checked, {s.id: s for s in fixture_sources(case)})
    result = asyncio.run(research.research_question(case["question"]))
    assert (not result.findings) == case["expect_abstention"]
    assert [finding.model_dump(exclude={"corroboration"}) for finding in result.findings] == [
        finding.model_dump(exclude={"corroboration"}) for finding in expected
    ]


def test_real_sdk_protocol_with_simulated_http(monkeypatch):
    """Exercise actual SDK serialization and parsing without network or spend."""
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            output = [{"id": "fc_1", "type": "function_call", "name": "search_news",
                       "call_id": "call_1", "arguments": '{"query":"Nvidia"}',
                       "status": "completed"}]
        elif len(requests) == 2:
            output = [{"id": "msg_1", "type": "message", "role": "assistant",
                       "status": "completed", "content": [{"type": "output_text",
                       "text": "Evidence is sufficient.", "annotations": []}]}]
        elif body.get("text", {}).get("format", {}).get("name") == "SourceReview":
            review = SourceReview(assessments=[SourceAssessment(
                source_id="S1", relevance="direct", reason="Explicit chip announcement.",
            )])
            output = [{"id": "msg_review", "type": "message", "role": "assistant",
                       "status": "completed", "content": [{"type": "output_text",
                       "text": review.model_dump_json(), "annotations": []}]}]
        elif body.get("text", {}).get("format", {}).get("name") == "CompactCorroborationReview":
            review = CorroborationReview(assessments=[{
                "finding_index": 0, "revised_text": "Nvidia announced a new chip.",
                "assessed_claim": "Nvidia announced a new chip.",
                "evidence_status": "announced", "explanation": "One supporting report.",
                "sources": [{"source_id": "S1", "stance": "supports", "explanation": "Chip announcement.",
                             "evidence": [answer().findings[0].evidence[0]]}],
            }])
            review = compact_fixture(review, json.loads(body["input"][0]["content"]))
            output = [{"id": "msg_corrob", "type": "message", "role": "assistant",
                       "status": "completed", "content": [{"type": "output_text",
                       "text": review.model_dump_json(), "annotations": []}]}]
        else:
            output = [{"id": "msg_2", "type": "message", "role": "assistant",
                       "status": "completed", "content": [{"type": "output_text",
                       "text": answer().model_dump_json(), "annotations": []}]}]
        return httpx.Response(200, json={
            "id": f"resp_{len(requests)}", "object": "response", "created_at": 1,
            "model": "gpt-4.1-nano", "status": "completed", "output": output,
            "error": None, "incomplete_details": None, "parallel_tool_calls": False,
            "tool_choice": "auto", "tools": [],
        })

    async def exercise():
        async with AsyncOpenAI(api_key="offline-test-key",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as sdk:
            monkeypatch.setattr(research, "client", sdk)
            monkeypatch.setattr(research, "search_articles", AsyncMock(return_value=[ARTICLE]))
            return await research.research_question("Nvidia news?")

    result = asyncio.run(exercise())
    assert result.findings[0].source_ids == ["S1"]
    assert requests[0]["tools"][0]["strict"] is True
    assert requests[0]["tool_choice"] == "required"
    assert requests[1]["input"][-1]["call_id"] == "call_1"
    assert requests[2]["text"]["format"]["type"] == "json_schema"
    assert "tools" not in requests[2]
    assert requests[2]["text"]["format"]["name"] == "SourceReview"
    assert requests[3]["text"]["format"]["name"] == "ResearchDraft"
    assert "tools" not in requests[3]
    assert requests[4]["text"]["format"]["name"] == "CompactCorroborationReview"
    assert "tools" not in requests[4]
    assert result.findings[0].corroboration.status == "single_source_reporting"


def test_invalid_quote_drops_only_affected_finding_after_repair(monkeypatch):
    draft = answer()
    bad = draft.findings[0].model_copy(deep=True)
    bad.text = "An unsupported separate development."
    bad.evidence[0].excerpt = "Invented quotation."
    draft.findings.append(bad)
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=draft)
    result = run()
    assert len(result.findings) == 1
    assert result.findings[0].text == answer().findings[0].text
    assert result.findings[0].evidence[0].excerpt == ARTICLE["description"]
    assert "quotations could not be validated" in result.limitations
    assert "Invented quotation" not in result.model_dump_json()
    assert responses.parse.await_count == 2
    assert len(json.loads(responses.corroborate.call_args.kwargs["input"][0]["content"])["candidate_findings"]) == 1
    search.assert_awaited_once()
    assert len(draft.findings) == 2  # Do not mutate the provider artifact.


def test_partial_quote_recovery_does_not_bypass_answer_quality(monkeypatch):
    draft = answer()
    draft.findings[0].text = "What happened to Nvidia?"
    bad = draft.findings[0].model_copy(deep=True)
    bad.evidence[0].excerpt = "Invented quotation."
    draft.findings.append(bad)
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=draft)
    with pytest.raises(research.ResearchError) as failed:
        run()
    assert failed.value.code == "synthesis_quality_invalid"
    responses.corroborate.assert_not_awaited()


@pytest.mark.parametrize("error", [
    APITimeoutError(request=httpx.Request("POST", "https://example.test")),
    OpenAIError("PRIVATE_PROVIDER_MESSAGE"),
])
def test_optional_review_failure_returns_explicitly_unassessed_draft(monkeypatch, caplog, error):
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.corroborate.side_effect = error
    result = run()
    assert result.findings[0].corroboration is None
    assert result.findings[0].evidence[0].excerpt == ARTICLE["description"]
    assert "Source support was not assessed" in result.limitations
    assert "PRIVATE" not in result.model_dump_json() + caplog.text
    responses.corroborate.assert_awaited_once()
    assert "optional stage unavailable: corroboration" in caplog.text


def test_failed_followup_review_retains_prior_assessment(monkeypatch):
    extra = dict(ARTICLE, title="Another chip report", url="https://second.test/news", source="Second")
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    search.side_effect = [[ARTICLE], [extra]]
    responses.corroborate.side_effect = [
        corroborated(ARTICLE, query="Nvidia confirmation"),
        APITimeoutError(request=httpx.Request("POST", "https://example.test")),
    ]
    result = run()
    assert result.findings[0].corroboration.status == "single_source_reporting"
    assert result.findings[0].source_ids == ["S1"]
    assert "additional source-support review could not be completed" in result.limitations
    assert "Source support was not assessed" not in result.limitations
    assert responses.corroborate.await_count == 2


def test_optional_deadline_cancels_review_without_losing_draft(monkeypatch):
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    cancelled = []
    async def slow(**kwargs):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    responses.corroborate.side_effect = slow
    monkeypatch.setattr(research, "RESEARCH_TIMEOUT", 1.05)
    result = run()
    assert result.findings[0].corroboration is None
    assert cancelled == [True]
    assert "Source support was not assessed" in result.limitations


def test_optional_followup_search_deadline_retains_review(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    responses.corroborate.side_effect = [corroborated(ARTICLE, query="Nvidia confirmation")]
    cancelled = []
    async def news(query):
        if query == "Nvidia":
            return [ARTICLE]
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    search.side_effect = news
    monkeypatch.setattr(research, "RESEARCH_TIMEOUT", 1.05)
    result = run()
    assert result.findings[0].corroboration.status == "single_source_reporting"
    assert cancelled == [True]
    assert "corroboration search failed" in result.limitations


@pytest.mark.parametrize("stage", ["search_planning", "source_review", "synthesis"])
def test_core_timeout_has_stage_and_never_returns_unchecked_answer(monkeypatch, caplog, stage):
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    target = {"search_planning": responses.create, "source_review": responses.review,
              "synthesis": responses.parse}[stage]
    target.side_effect = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    result = TestClient(app).post("/api/research", json={"question": "PRIVATE_QUESTION"})
    assert result.status_code == 504
    assert stage + " model_request_timeout" in caplog.text
    assert "PRIVATE_QUESTION" not in caplog.text
    responses.corroborate.assert_not_awaited()


def test_corrob_requests_are_bounded_without_discarding_source_pool(monkeypatch):
    draft = answer()
    for i in range(1, 8):
        item = draft.findings[0].model_copy(deep=True)
        item.text = f'Attributed development number {i}.'
        draft.findings.append(item)
    responses, _ = setup_agent(monkeypatch, [response(call()), response()], draft=draft)
    result = run()
    assert len(result.findings) == 5
    assert 'limited to five' in result.limitations
    request = responses.corroborate.call_args.kwargs
    data = json.loads(request['input'][0]['content'])
    assert len(data['candidate_findings']) == 5
    assert request['max_output_tokens'] == 4500
    assert all('evidence' not in f for f in data['candidate_findings'])
    assert data['sources'][0]['evidence'][1]['text'] == ARTICLE['description']
