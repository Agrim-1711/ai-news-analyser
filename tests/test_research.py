import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI, OpenAIError

from app.main import app
from app.schemas import ResearchDraft, ResearchFinding
from app.services import research

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


def setup_agent(monkeypatch, turns, articles=None, draft=None):
    responses = SimpleNamespace(
        create=AsyncMock(side_effect=turns),
        parse=AsyncMock(return_value=SimpleNamespace(status="completed", output_parsed=draft)),
    )
    fake = SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace(responses=responses))
    monkeypatch.setattr(research, "client", fake)
    search = AsyncMock(return_value=[ARTICLE] if articles is None else articles)
    monkeypatch.setattr(research, "search_articles", search)
    return responses, search


def answer(ids=None):
    return ResearchDraft(findings=[ResearchFinding(
        text="Nvidia announced a new chip.", source_ids=ids or ["S1"],
    )], limitations="Coverage is limited.")


def run():
    return asyncio.run(research.research_question("What happened at Nvidia?"))


def test_tool_output_roundtrip_and_citations(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    result = run()
    search.assert_awaited_once_with("Nvidia")
    assert str(result.sources[0].url) == ARTICLE["url"]
    assert result.findings[0].source_ids == ["S1"]
    inputs = responses.parse.call_args.kwargs["input"]
    output = next(item for item in inputs if isinstance(item, dict) and item.get("type") == "function_call_output")
    assert output["call_id"] == "call_1"
    assert json.loads(output["output"])["articles"][0]["id"] == "S1"
    assert "descriptions" in result.limitations


def test_refines_search_and_deduplicates_sources(monkeypatch):
    _, search = setup_agent(monkeypatch, [response(call()), response(call("Nvidia chip")), response()], draft=answer())
    result = run()
    assert [c.args[0] for c in search.await_args_list] == ["Nvidia", "Nvidia chip"]
    assert len(result.sources) == 1


def test_search_budget(monkeypatch):
    responses, search = setup_agent(monkeypatch, [response(call())] * 3, draft=answer())
    result = run()
    assert search.await_count == 3
    assert responses.create.await_count == 3
    assert "search limit" in result.limitations


def test_no_evidence_abstains(monkeypatch):
    responses, search = setup_agent(
        monkeypatch,
        [response(call())] * 3,
        articles=[],
    )

    assert run().findings == []
    assert search.await_count == 3
    assert all(
        c.kwargs["tool_choice"] == "required"
        for c in responses.create.await_args_list
    )
    responses.parse.assert_not_awaited()


def test_empty_search_recovers_with_broader_query(monkeypatch):
    responses, search = setup_agent(
        monkeypatch,
        [
            response(call("recent developments affecting Nvidia")),
            response(call("Nvidia")),
            response(),
        ],
        draft=answer(),
    )
    search.side_effect = [[], [ARTICLE]]

    result = run()

    assert result.findings[0].source_ids == ["S1"]
    assert result.searches == [
        "recent developments affecting Nvidia",
        "Nvidia",
    ]
    assert responses.create.await_args_list[1].kwargs["tool_choice"] == "required"
    assert responses.create.await_args_list[2].kwargs["tool_choice"] == "auto"

    messages = responses.parse.call_args.kwargs["input"]
    tool_outputs = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("type") == "function_call_output"
    ]

    assert (
        "broader query"
        in json.loads(tool_outputs[0]["output"])["guidance"]
    )


def test_fabricated_citation_rejected(monkeypatch):
    setup_agent(monkeypatch, [response(call()), response()], draft=answer(["S99"]))
    with pytest.raises(research.ResearchError, match="unknown sources"):
        run()


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
    (httpx.RequestError("connection failed"), 503),
    (httpx.HTTPStatusError("failure", request=httpx.Request("GET", "https://example.com"),
                           response=httpx.Response(429)), 502),
])
def test_endpoint_errors(monkeypatch, error, status):
    monkeypatch.setattr("app.main.research_question", AsyncMock(side_effect=error))
    result = TestClient(app).post("/api/research", json={"question": "Nvidia news?"})
    assert result.status_code == status
    assert "private" not in result.text


def test_endpoint_full_agent(monkeypatch):
    setup_agent(monkeypatch, [response(call()), response()], draft=answer())
    result = TestClient(app).post("/api/research", json={"question": "Nvidia news?"})
    assert result.status_code == 200
    assert result.json()["sources"][0]["id"] == "S1"


def test_existing_page_contains_both_workflows():
    page = TestClient(app).get("/")
    assert page.status_code == 200
    assert 'id="research-form"' in page.text
    assert 'id="search-form"' in page.text


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