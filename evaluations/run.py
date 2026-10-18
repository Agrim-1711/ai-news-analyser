"""python -m evaluations.run [--live] [--output path.json]

Default mode validates reference fixtures only, with no model calls.
--live evaluates the actual agent with fixed news sources and paid OpenAI calls.
Semantic claim support, date coverage and status accuracy require human review.
"""
import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from app.schemas import CorroborationReview, ResearchDraft, ResearchResponse, ResearchSource
from app.services.corroboration import apply_corroboration
from app.services.evidence import validate_evidence
from evaluations.cases import CASES


def fixture_sources(case):
    return [ResearchSource(id=f"S{i}", **article)
            for i, article in enumerate(case["articles"], start=1)]


async def evaluate(live=False, case_ids=None):
    # Keep imports lazy so reference checks need neither credentials nor a DB.
    if live:
        from app.services.research import ResearchError, research_question
    rows = []
    for case in CASES:
        if case_ids is not None and case["id"] not in case_ids:
            continue
        try:
            if live:
                with patch("app.services.research.search_articles", AsyncMock(return_value=case["articles"])):
                    result = await research_question(case["question"])
                sources = result.sources
            else:
                result = ResearchDraft(findings=case["reference_findings"],
                                       limitations=case["reference_limitations"])
                sources = fixture_sources(case)
                if case.get("reference_corroboration"):
                    findings = apply_corroboration(result, CorroborationReview.model_validate(
                        case["reference_corroboration"]), {source.id: source for source in sources})
                    result = ResearchResponse(findings=findings, limitations=result.limitations,
                                              sources=sources, searches=[])
            validate_evidence(result, sources)
            cited_ids = {sid for finding in result.findings for sid in finding.source_ids}
            cited_urls = {str(s.url) for s in sources if s.id in cited_ids}
            wording = " ".join(f.text for f in result.findings).casefold().replace("’", "'")
            rows.append({
                "id": case["id"], "question": case["question"],
                "quote_provenance_pass": True,
                "abstention_matches_reference": (not result.findings) == case["expect_abstention"],
                "required_sources_cited": (
                    set(case.get("required_source_urls", [])).issubset(cited_urls)
                    and all(set(group).intersection(cited_urls) for group in case.get("required_source_groups", []))
                ),
                "forbidden_sources_excluded": not set(case.get("forbidden_source_urls", [])).intersection(cited_urls),
                "no_duplicate_story_citations": all(len(set(group).intersection(cited_urls)) <= 1
                                                    for group in case.get("duplicate_source_groups", [])),
                "no_teaser_language": not any(phrase in wording for phrase in ["here's how", "here is how", "find out"]),
                "support_labels_match_reference": (
                    [getattr(getattr(f, "corroboration", None), "status", None) for f in result.findings]
                    == case["expected_support_statuses"] if "expected_support_statuses" in case else None
                ),
                "shared_origin_matches_reference": (
                    [getattr(getattr(f, "corroboration", None), "possible_shared_origin", None) for f in result.findings]
                    == case["expected_shared_origin"] if "expected_shared_origin" in case else None
                ),
                "result": result.model_dump(mode="json"),
                "sources": [s.model_dump(mode="json") for s in sources],
                "human_review": {"rubric": case["review"], "claim_support": None,
                                 "date_coverage": None, "status_accuracy": None, "question_relevance": None,
                                 "relationship_accuracy": None, "source_independence": None,
                                 "notes": "Not yet reviewed"},
            })
        except Exception as exc:
            # Provider exceptions can contain request details; never export them.
            row = {"id": case["id"], "error_type": type(exc).__name__}
            if isinstance(exc, ValidationError):
                row["schema_errors"] = [{"location": list(error["loc"]), "type": error["type"]}
                                        for error in exc.errors(include_input=False, include_context=False)]
            if live and isinstance(exc, ResearchError):
                row["error_code"] = exc.code
            rows.append(row)
    return {"mode": "live_model_fixed_news" if live else "reference_fixture_validation",
            "model_quality_evaluated": live,
            "semantic_review_complete": False, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make paid OpenAI calls; news remains fixed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", choices=[case["id"] for case in CASES],
                        help="Evaluate only this case; repeat to select several")
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.live, args.case))
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
        print(f"Saved {len(report['cases'])} cases to {args.output}")
    else:
        print(payload)
    failed = any(row.get("error_type") or row.get("support_labels_match_reference") is False
                 or row.get("shared_origin_matches_reference") is False or not all(row[key] for key in (
        "abstention_matches_reference", "required_sources_cited", "forbidden_sources_excluded",
        "no_duplicate_story_citations", "no_teaser_language"))
                 for row in report["cases"])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
