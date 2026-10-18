import pytest
from pydantic import ValidationError

from app.schemas import ResearchDraft, ResearchEvidence, ResearchFinding, ResearchSource
from app.services.evidence import (
    EvidenceValidationError, align_evidence_fields, expand_description_excerpts, validate_evidence,
)


def source(id="S1", **changes):
    return ResearchSource(**{
        "id": id, "title": "Acme announces a chip", "description": "Launch is expected in 2027.",
        "url": f"https://example.com/{id}", **changes,
    })


def draft(excerpt="Acme announces a chip", field="title", source_ids=None,
          evidence_id="S1", status="announced"):
    return ResearchDraft(findings=[ResearchFinding(
        text="Acme announced a chip.", source_ids=source_ids or ["S1"],
        evidence_status=status,
        evidence=[ResearchEvidence(source_id=evidence_id, field=field, excerpt=excerpt)],
    )], limitations="Only metadata is available.")


def test_title_quote_is_accepted():
    assert validate_evidence(draft(), [source()]) == {"S1"}


def test_description_quote_is_accepted():
    assert validate_evidence(draft("expected in 2027.", "description"), [source()]) == {"S1"}


def test_layout_whitespace_is_normalized():
    assert validate_evidence(draft("Acme\n announces  a chip"), [source()]) == {"S1"}


@pytest.mark.parametrize("excerpt,field", [
    ("Acme announces a rocket", "title"),     # invented quote
    ("Acme ... a chip", "title"),            # stitched fragments
    ("acme announces a chip", "title"),      # changed source text
    ("Acme announces a chip", "description"), # right source, wrong field
    ("chip Launch is expected", "title"),    # cannot span fields
])
def test_mismatched_quotes_are_rejected(excerpt, field):
    with pytest.raises(EvidenceValidationError, match="does not match"):
        validate_evidence(draft(excerpt, field), [source()])


def test_missing_description_cannot_supply_a_quote():
    with pytest.raises(EvidenceValidationError):
        validate_evidence(draft("Launch", "description"), [source(description=None)])


def test_quote_must_match_its_own_source():
    with pytest.raises(EvidenceValidationError):
        validate_evidence(draft(), [source(title="Different report"), source("S2")])


@pytest.mark.parametrize("ids,evidence_id", [(["S1", "S2"], "S1"), (["S1"], "S2")])
def test_every_citation_needs_its_own_evidence(ids, evidence_id):
    with pytest.raises(EvidenceValidationError, match="every cited source"):
        validate_evidence(draft(source_ids=ids, evidence_id=evidence_id), [source(), source("S2")])


def test_unknown_citation_is_rejected():
    with pytest.raises(EvidenceValidationError, match="unknown sources"):
        validate_evidence(draft(source_ids=["S99"], evidence_id="S99"), [source()])


@pytest.mark.parametrize("status", ["reported", "announced", "speculative"])
def test_allowed_statuses(status):
    assert draft(status=status).findings[0].evidence_status == status


def test_verified_is_not_an_allowed_status():
    with pytest.raises(ValidationError):
        draft(status="verified")


@pytest.mark.parametrize("excerpt", ["", " ", "\n\t"])
def test_blank_evidence_is_invalid(excerpt):
    with pytest.raises(ValidationError):
        draft(excerpt)


def test_findings_cannot_omit_evidence():
    data = draft().model_dump()
    data["findings"][0]["evidence"] = []
    with pytest.raises(ValidationError):
        ResearchDraft.model_validate(data)


def test_abstention_needs_no_sources():
    empty = ResearchDraft(findings=[], limitations="No relevant evidence.")
    assert validate_evidence(empty, []) == set()


def test_multi_source_evidence():
    result = draft(source_ids=["S1", "S2"])
    result.findings[0].evidence.append(ResearchEvidence(
        source_id="S2", field="description", excerpt="Launch is expected in 2027.",
    ))
    assert validate_evidence(result, [source(), source("S2")]) == {"S1", "S2"}


def test_matching_quote_does_not_prove_claim_support():
    # This must stay explicit: provenance validation is not an entailment grader.
    result = draft()
    result.findings[0].text = "Acme has already shipped ten million chips."
    assert validate_evidence(result, [source()]) == {"S1"}


def test_quote_field_resolved_only_when_exactly_present_in_same_source():
    original = draft("Launch is expected in 2027.", "title")
    aligned = align_evidence_fields(original, [source()])
    assert aligned.findings[0].evidence[0].field == "description"
    assert original.findings[0].evidence[0].field == "title"
    assert aligned.findings[0].evidence[0].excerpt == original.findings[0].evidence[0].excerpt
    validate_evidence(aligned, [source()])


def test_quote_alignment_does_not_invent_or_borrow_from_other_source():
    original = draft("PRIVATE_INVENTED_QUOTE", "title")
    aligned = align_evidence_fields(original, [source(), source("S2", description="PRIVATE_INVENTED_QUOTE")])
    assert aligned.findings[0].evidence[0].field == "title"
    with pytest.raises(EvidenceValidationError):
        validate_evidence(aligned, [source()])


def test_fragment_expands_to_its_source_sentence_with_actor_and_action():
    description = "Earlier reporting gave background. Microsoft said it blocked phishing as it disrupted YAM Store. More details follow."
    article = source(description=description)
    original = draft("as it disrupted YAM Store.", "description")
    result = expand_description_excerpts(original, [article])
    assert result.findings[0].evidence[0].excerpt == "Microsoft said it blocked phishing as it disrupted YAM Store."
    assert original.findings[0].evidence[0].excerpt == "as it disrupted YAM Store."
    validate_evidence(result, [article])


def test_description_context_preserves_whitespace_normalized_provenance():
    article = source(description="Microsoft said\nit disrupted YAM Store.")
    result = expand_description_excerpts(draft("disrupted YAM Store.", "description"), [article])
    assert result.findings[0].evidence[0].excerpt == "Microsoft said it disrupted YAM Store."
    validate_evidence(result, [article])


@pytest.mark.parametrize("excerpt,field,description", [
    ("Acme announces a chip", "title", "Launch is expected in 2027."),
    ("Fabricated quote", "description", "An unrelated description."),
    ("Store. More", "description", "Microsoft disrupted YAM Store. More details follow."),
    ("short fragment", "description", "A " + "long " * 110 + "short fragment ends here."),
])
def test_context_expansion_preserves_titles_invalid_quotes_and_long_or_spanning_quotes(excerpt, field, description):
    result = expand_description_excerpts(draft(excerpt, field), [source(description=description)])
    assert result.findings[0].evidence[0].excerpt == excerpt


def test_context_never_borrows_from_another_source():
    original = draft("Microsoft disrupted YAM Store.", "description")
    articles = [source(), source("S2", description="Microsoft disrupted YAM Store.")]
    result = expand_description_excerpts(original, articles)
    with pytest.raises(EvidenceValidationError):
        validate_evidence(result, articles)
