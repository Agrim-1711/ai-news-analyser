"""Curated relationship checks test mechanics, not a live model's semantic judgment."""
import pytest

from app.schemas import CorroborationReview, ResearchDraft, ResearchSource
from app.services.corroboration import apply_corroboration
from app.services.evidence import EvidenceValidationError, validate_evidence
from app.services.quality import AnswerQualityError


def source(sid, description="Acme cancelled the Orion deal.", publisher=None):
    return ResearchSource(id=sid, title="Orion deal update", description=description,
                          url=f"https://{sid.lower()}.example.test/report",
                          source=publisher or f"Publisher {sid}")


def candidate(sources):
    first = next(iter(sources.values()))
    return ResearchDraft(findings=[{
        "text": "Acme reportedly cancelled the Orion deal.", "source_ids": [first.id],
        "evidence_status": "reported", "evidence": [{
            "source_id": first.id, "field": "description", "excerpt": first.description,
        }],
    }], limitations="Fixture sources only.")


def relation(article, stance="supports", **changes):
    return dict(source_id=article.id, stance=stance, explanation="Curated fixture relationship.",
                evidence=[dict(source_id=article.id, field="description", excerpt=article.description)],
                **changes)


def review(*relations, **changes):
    data = dict(finding_index=0, revised_text="Acme reportedly cancelled the Orion deal.",
                assessed_claim="Acme cancelled the Orion deal.",
                evidence_status="reported", sources=list(relations), explanation="Curated source review.")
    data.update(changes)
    return CorroborationReview(assessments=[data])


def origin_clue(other, origin, *articles):
    return dict(other_source_id=other, origin_name=origin, evidence=[
        dict(source_id=article.id, field="description", excerpt=article.description) for article in articles
    ])


def test_single_report_is_retained_without_forcing_a_second_source():
    sources = {"S1": source("S1")}
    finding = apply_corroboration(candidate(sources), review(relation(sources["S1"])), sources)[0]
    assert finding.corroboration.status == "single_source_reporting"
    assert finding.corroboration.supporting_groups == [["S1"]]


def test_two_distinct_reports_support_one_claim_without_certifying_independence():
    sources = {"S1": source("S1"), "S2": source("S2", "The Orion agreement was cancelled by Acme.")}
    finding = apply_corroboration(candidate(sources), review(*(relation(s) for s in sources.values())), sources)[0]
    assert finding.corroboration.status == "multiple_source_support"
    assert finding.corroboration.supporting_groups == [["S1"], ["S2"]]
    assert set(finding.source_ids) == {"S1", "S2"}
    validate_evidence(ResearchDraft(findings=[finding], limitations="Fixture."), sources.values())


def test_near_identical_syndication_is_one_group_even_across_publishers():
    text = "Acme cancelled the Orion deal after its board reviewed the agreement on Tuesday morning."
    sources = {"S1": source("S1", text), "S2": source("S2", text.replace("Tuesday", "Wednesday"))}
    finding = apply_corroboration(candidate(sources), review(*(relation(s) for s in sources.values())), sources)[0]
    assert finding.corroboration.status == "single_source_reporting"
    assert finding.corroboration.possible_shared_origin is True
    assert finding.corroboration.supporting_groups == [["S1", "S2"]]


def test_explicit_shared_attribution_is_grouped_transitively():
    sources = {sid: source(sid, description) for sid, description in [
        ("S1", "Acme said it cancelled Orion."),
        ("S2", "According to a Wire News report, Acme cancelled the deal."),
        ("S3", "The cancellation appears in a Wire News report."),
    ]}
    sources["S1"].source = "Wire News"
    assessed = review(relation(sources["S1"]), relation(sources["S2"], shared_origin_with=["S1"],
                      origin_evidence=[origin_clue("S1", "Wire News", sources["S2"])]),
                      relation(sources["S3"], shared_origin_with=["S2"],
                      origin_evidence=[origin_clue("S2", "Wire News", sources["S2"], sources["S3"])]))
    finding = apply_corroboration(candidate(sources), assessed, sources)[0]
    assert finding.corroboration.status == "single_source_reporting"
    assert finding.corroboration.supporting_groups == [["S1", "S2", "S3"]]


def test_same_publisher_does_not_inflate_source_groups():
    sources = {"S1": source("S1", publisher="Newsroom"),
               "S2": source("S2", "The Orion agreement ended.", publisher=" NEWSROOM ")}
    finding = apply_corroboration(candidate(sources), review(*(relation(s) for s in sources.values())), sources)[0]
    assert finding.corroboration.status == "single_source_reporting"
    assert len(finding.corroboration.supporting_groups) == 1


def test_conflict_preserves_both_accounts_and_does_not_choose_a_winner():
    sources = {"S1": source("S1"), "S2": source("S2", "Acme says the Orion deal remains in force.")}
    assessed = review(relation(sources["S1"]), relation(sources["S2"], "challenges"),
                      revised_text="Reports disagree on whether the Orion deal was cancelled.",
                      conflict_explanation="The same agreement is described as cancelled and still in force.")
    finding = apply_corroboration(candidate(sources), assessed, sources)[0]
    assert finding.corroboration.status == "conflicting_reports"
    assert finding.corroboration.challenging_source_ids == ["S2"]
    assert "disagree" in finding.text
    assert {item.source_id for item in finding.evidence} == {"S1", "S2"}


def test_different_dates_or_regions_are_not_automatically_called_conflicting():
    sources = {"S1": source("S1", "Orion launched in the UK on Monday."),
               "S2": source("S2", "Orion launched in the US on Friday.")}
    # The curated assessment recognises different scopes. Semantic recognition is
    # the model's job and must also be checked in live evaluation.
    assessed = review(relation(sources["S1"]), relation(sources["S2"], "context"),
                      assessed_claim="Orion launched in the UK on Monday.",
                      revised_text="Orion launched in the UK on Monday.")
    finding = apply_corroboration(candidate(sources), assessed, sources)[0]
    assert finding.corroboration.status == "single_source_reporting"
    assert finding.corroboration.challenging_source_ids == []
    assert finding.corroboration.context_source_ids == ["S2"]
    assert finding.corroboration.supporting_groups == [["S1"]]


def test_unsupported_candidate_can_be_removed():
    sources = {"S1": source("S1")}
    assert apply_corroboration(candidate(sources), review(), sources) == []


@pytest.mark.parametrize("change", ["unknown", "duplicate", "missing_original", "wrong_quote_id",
                                   "invented_quote", "self_origin", "unknown_origin",
                                   "missing_conflict", "no_support"])
def test_invalid_review_is_rejected(change):
    sources = {"S1": source("S1"), "S2": source("S2")}
    checked = review(relation(sources["S1"]))
    assessment = checked.assessments[0]
    item = assessment.sources[0]
    if change == "unknown":
        item.source_id = "S99"
    elif change == "duplicate":
        assessment.sources.append(item.model_copy(deep=True))
    elif change == "missing_original":
        assessment.sources = [type(item).model_validate(relation(sources["S2"]))]
    elif change == "wrong_quote_id":
        item.evidence[0].source_id = "S2"
    elif change == "invented_quote":
        item.evidence[0].excerpt = "PRIVATE_INVENTED_QUOTE"
    elif change == "self_origin":
        item.shared_origin_with = ["S1"]
    elif change == "unknown_origin":
        item.shared_origin_with = ["S99"]
    elif change == "missing_conflict":
        assessment.sources.append(type(item).model_validate(relation(sources["S2"], "challenges")))
    elif change == "no_support":
        item.stance = "challenges"
        assessment.conflict_explanation = "Disagreement."
    with pytest.raises((AnswerQualityError, EvidenceValidationError)):
        apply_corroboration(candidate(sources), checked, sources)


def test_context_only_candidate_is_omitted_without_repair():
    sources = {"S1": source("S1")}
    assert apply_corroboration(candidate(sources), review(relation(sources["S1"], "context")), sources) == []


def test_agreement_on_nvidia_statement_does_not_establish_shared_origin():
    sources = {"S1": source("S1", "Nvidia says the RTX 5090 isn't discontinued despite severe shortages, disappearing retailer listings, and prices reaching $8,500 on Amazon."),
               "S2": source("S2", "NVIDIA denies the discontinuation of the RTX 5090 despite reports, emphasizing its importance amid severe supply constraints as prices exceed $5000.")}
    assessed = review(relation(sources["S1"], shared_origin_with=["S2"],
                      origin_reason="Both cover Nvidia's denial, but we cannot establish whether one draws on the other."),
                      relation(sources["S2"]))
    original = assessed.model_dump()
    finding = apply_corroboration(candidate(sources), assessed, sources)[0]
    assert finding.corroboration.status == "multiple_source_support"
    assert finding.corroboration.supporting_groups == [["S1"], ["S2"]]
    assert finding.corroboration.possible_shared_origin is False
    assert finding.corroboration.source_assessments[0].shared_origin_with == []
    assert finding.corroboration.source_assessments[0].origin_reason is None
    assert assessed.model_dump() == original


@pytest.mark.parametrize("description", ["Acme said the Orion deal ended.",
                                         "According to Acme, the Orion deal ended.",
                                         "Acme reports the Orion deal ended."])
def test_matching_company_statements_are_not_attribution_to_a_shared_report(description):
    sources = {"S1": source("S1", description), "S2": source("S2", description)}
    checked = review(relation(sources["S1"], shared_origin_with=["S2"],
                     origin_reason="Both quote Acme.", origin_evidence=[origin_clue("S2", "Acme", *sources.values())]),
                     relation(sources["S2"]))
    finding = apply_corroboration(candidate(sources), checked, sources)[0]
    assert finding.corroboration.status == "multiple_source_support"
    assert finding.corroboration.possible_shared_origin is False


def test_two_explicit_attributions_to_the_same_press_release_can_group():
    sources = {"S1": source("S1", "According to Acme's press release, the Orion deal was cancelled."),
               "S2": source("S2", "Acme's press release states that the agreement ended.")}
    checked = review(relation(sources["S1"], shared_origin_with=["S2"],
                     origin_evidence=[origin_clue("S2", "Acme", *sources.values())]), relation(sources["S2"]))
    finding = apply_corroboration(candidate(sources), checked, sources)[0]
    assert finding.corroboration.status == "single_source_reporting"
    assert finding.corroboration.possible_shared_origin is True
    assert "Acme" in finding.corroboration.grouping_notes[0]
    assert finding.corroboration.source_assessments[0].origin_reason.startswith("Retrieved attribution")


def test_wire_byline_and_primary_publisher_metadata_support_a_link():
    sources = {"S1": source("S1", publisher="Reuters"),
               "S2": source("S2", "Oct 18 (Reuters) - Acme cancelled the Orion deal.")}
    checked = review(relation(sources["S1"]), relation(sources["S2"], shared_origin_with=["S1"],
                     origin_evidence=[origin_clue("S1", "Reuters", sources["S2"])]))
    finding = apply_corroboration(candidate(sources), checked, sources)[0]
    assert finding.corroboration.possible_shared_origin is True
    assert finding.corroboration.supporting_groups == [["S1", "S2"]]


def test_origin_attribution_can_come_from_a_title():
    sources = {"S1": source("S1", publisher="Wire News"), "S2": source("S2", "The Orion agreement ended.")}
    sources["S2"].title = "A Wire News report: Acme cancels Orion"
    clue = dict(other_source_id="S1", origin_name="Wire News", evidence=[{
        "source_id": "S2", "field": "title", "excerpt": sources["S2"].title,
    }])
    checked = review(relation(sources["S1"]), relation(sources["S2"], shared_origin_with=["S1"], origin_evidence=[clue]))
    finding = apply_corroboration(candidate(sources), checked, sources)[0]
    assert finding.corroboration.possible_shared_origin is True
    assert finding.corroboration.source_assessments[1].origin_evidence[0].evidence[0].field == "title"


@pytest.mark.parametrize("failure", ["invented_quote", "only_one_side", "wrong_source", "bare_name"])
def test_unsubstantiated_origin_evidence_is_ignored_without_losing_claim(failure):
    sources = {"S1": source("S1", "A Wire News report says the deal ended."),
               "S2": source("S2", "According to a Wire News report, Acme cancelled Orion.")}
    clue = origin_clue("S2", "Wire News", *sources.values())
    if failure == "invented_quote":
        clue["evidence"][0]["excerpt"] = "A Wire News report contains fabricated words."
    elif failure == "only_one_side":
        clue["evidence"] = clue["evidence"][:1]
    elif failure == "wrong_source":
        clue["evidence"][0]["source_id"] = "S99"
    elif failure == "bare_name":
        clue["evidence"][0]["excerpt"] = "Wire News"
    checked = review(relation(sources["S1"], shared_origin_with=["S2"],
                     origin_reason="Unsubstantiated model guess.", origin_evidence=[clue]), relation(sources["S2"]))
    finding = apply_corroboration(candidate(sources), checked, sources)[0]
    assert finding.corroboration.status == "multiple_source_support"
    assert finding.corroboration.possible_shared_origin is False
    assert finding.corroboration.source_assessments[0].origin_evidence == []
    assert finding.corroboration.source_assessments[0].origin_reason is None


def test_same_publisher_grouping_does_not_prove_reuse_of_one_report():
    sources = {"S1": source("S1", publisher="Newsroom"),
               "S2": source("S2", "The Orion agreement ended.", publisher="Newsroom")}
    checked = review(relation(sources["S1"], shared_origin_with=["S2"], origin_reason="Same publisher."),
                     relation(sources["S2"]))
    finding = apply_corroboration(candidate(sources), checked, sources)[0]
    assert finding.corroboration.status == "single_source_reporting"
    assert finding.corroboration.possible_shared_origin is False
    assert "same publisher" in finding.corroboration.grouping_notes[0]


@pytest.mark.parametrize("description", [
    "This article does not rely on a Wire News report. Acme cancelled Orion.",
    "The Wire News report was not used for this article. Acme cancelled Orion.",
])
def test_negated_attribution_is_not_a_shared_origin_link(description):
    sources = {"S1": source("S1", publisher="Wire News"), "S2": source("S2", description)}
    checked = review(relation(sources["S1"]), relation(sources["S2"], shared_origin_with=["S1"],
                     origin_evidence=[origin_clue("S1", "Wire News", sources["S2"])]))
    finding = apply_corroboration(candidate(sources), checked, sources)[0]
    assert finding.corroboration.possible_shared_origin is False
    assert finding.corroboration.status == "multiple_source_support"


def test_negation_of_the_news_claim_does_not_negate_its_attribution():
    sources = {"S1": source("S1", publisher="Wire News"),
               "S2": source("S2", "The chip is not discontinued, according to a Wire News report.")}
    checked = review(relation(sources["S1"]), relation(sources["S2"], shared_origin_with=["S1"],
                     origin_evidence=[origin_clue("S1", "Wire News", sources["S2"])]))
    finding = apply_corroboration(candidate(sources), checked, sources)[0]
    assert finding.corroboration.possible_shared_origin is True


@pytest.mark.parametrize("indexes", [[], [1], [0, 0]])
def test_every_finding_must_be_assessed_once(indexes):
    sources = {"S1": source("S1")}
    checked = review(relation(sources["S1"]))
    checked.assessments = [checked.assessments[0].model_copy(update={"finding_index": i}) for i in indexes]
    with pytest.raises(AnswerQualityError, match="every finding"):
        apply_corroboration(candidate(sources), checked, sources)
