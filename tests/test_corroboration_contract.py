"""Check the new model boundary without spending API credit."""
import pytest
from pydantic import ValidationError

from app.schemas import ResearchDraft, ResearchSource
from app.services.corroboration import apply_corroboration
from app.services.corroboration_contract import (
    CompactCorroborationReview, evidence_catalog, expand_review,
)
from app.services.evidence import validate_evidence
from app.services.quality import AnswerQualityError


def sources():
    return {sid: ResearchSource(id=sid, title='Acme deal update', description=text,
                source=publisher, url=url) for sid, text, publisher, url in [
        ('S1', 'Acme cancelled its Orion deal.', 'Wire News', 'https://wire.test/orion'),
        ('S2', 'According to Wire News, Acme cancelled its Orion deal.', 'Second', 'https://second.test/orion'),
    ]}


def compact(**updates):
    finding = dict(finding_index=0, assessed_claim='Acme cancelled its Orion deal.',
        revised_text='Reports say Acme cancelled its Orion deal.', evidence_status='reported',
        sources=[dict(source_id='S1', stance='supports', explanation='Reports the cancellation.',
                      evidence_ids=['S1:D'])], explanation='One attributed report.', origins=[])
    finding.update(updates)
    return CompactCorroborationReview(assessments=[finding])


def test_server_restores_exact_quotes_and_reuses_support_checks():
    available = sources()
    catalog, _ = evidence_catalog(available)
    review = expand_review(compact(), catalog)
    assert review.assessments[0].sources[0].evidence[0].excerpt == available['S1'].description
    draft = ResearchDraft(findings=[dict(text='Acme cancelled its Orion deal.', source_ids=['S1'],
        evidence_status='reported', evidence=[catalog['S1:D']])], limitations='Fixture.')
    result = apply_corroboration(draft, review, available)
    validate_evidence(ResearchDraft(findings=result, limitations='Fixture.'), available.values())
    assert result[0].corroboration.status == 'single_source_reporting'
    assert result[0].evidence[0].excerpt == available['S1'].description


@pytest.mark.parametrize('refs', [['S99:D'], ['S2:D'], ['S1:D', 'S1:D']])
def test_unknown_cross_source_and_duplicate_ids_are_rejected(refs):
    catalog, _ = evidence_catalog(sources())
    value = compact()
    value.assessments[0].sources[0].evidence_ids = refs
    with pytest.raises(AnswerQualityError):
        expand_review(value, catalog)


def test_compact_contract_forbids_generated_quote_text():
    value = compact().model_dump()
    value['assessments'][0]['sources'][0]['excerpt'] = 'A fabricated quotation.'
    with pytest.raises(ValidationError):
        CompactCorroborationReview.model_validate(value)


def test_origin_ids_preserve_explicit_attribution_and_backend_grouping():
    available = sources()
    catalog, _ = evidence_catalog(available)
    value = compact()
    item = value.assessments[0]
    second = item.sources[0].model_copy(update={'source_id': 'S2', 'evidence_ids': ['S2:D']})
    value = compact(sources=[item.sources[0], second], origins=[dict(
        source_id='S2', other_source_id='S1', origin_name='Wire News', evidence_ids=['S2:D'])])
    review = expand_review(value, catalog)
    draft = ResearchDraft(findings=[dict(text='Acme cancelled its Orion deal.', source_ids=['S1'],
        evidence_status='reported', evidence=[catalog['S1:D']])], limitations='Fixture.')
    result = apply_corroboration(draft, review, available)
    assert result[0].corroboration.supporting_groups == [['S1', 'S2']]
    assert result[0].corroboration.possible_shared_origin is True
    assert result[0].corroboration.source_assessments[1].origin_evidence[0].evidence[0].excerpt == available['S2'].description


def test_origin_reference_cannot_borrow_evidence_from_unrelated_source():
    catalog, _ = evidence_catalog(sources())
    value = compact(origins=[dict(source_id='S1', other_source_id='S99',
                                  origin_name='Wire News', evidence_ids=['S2:D'])])
    with pytest.raises(AnswerQualityError):
        expand_review(value, catalog)


def test_complete_unaltered_source_text_including_escape_sequences_is_restored():
    available = sources()
    available['S1'].description = r'Acme\u2019s announcement said \"Orion\" was cancelled.'
    catalog, _ = evidence_catalog(available)
    review = expand_review(compact(), catalog)
    assert review.assessments[0].sources[0].evidence[0].excerpt == available['S1'].description


def test_old_request_catalog_does_not_resolve_new_source_ids():
    catalog, _ = evidence_catalog({'S1': sources()['S1']})
    value = compact()
    value.assessments[0].sources[0].source_id = 'S2'
    value.assessments[0].sources[0].evidence_ids = ['S2:D']
    with pytest.raises(AnswerQualityError):
        expand_review(value, catalog)


def test_only_validated_draft_passages_can_enter_catalog():
    available = sources()
    draft = ResearchDraft(findings=[dict(text='A claim.', source_ids=['S1'], evidence_status='reported',
        evidence=[dict(source_id='S1', field='description', excerpt='Fabricated quote.')])], limitations='Fixture.')
    with pytest.raises(AnswerQualityError):
        evidence_catalog(available, draft)
