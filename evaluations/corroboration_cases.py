"""Synthetic scenarios. Reference relationships are curated, not model measurements."""


def article(sid, description):
    return dict(title="Acme Orion update", description=description,
                url=f"https://publisher-{sid}.example.test/orion", source=f"Fixture Publisher {sid}",
                published_at="2026-10-18T10:00:00Z")


def case(case_id, articles, revised, stances, expected, rubric, conflict=None, shared=False):
    initial = dict(text="One report says Acme cancelled the Orion deal.", source_ids=["S1"],
                   evidence_status="reported", evidence=[dict(source_id="S1", field="description",
                                                              excerpt=articles[0]["description"])])
    relations = [dict(source_id=f"S{i}", stance=stance, explanation="Curated fixture relationship.",
                      evidence=[dict(source_id=f"S{i}", field="description", excerpt=articles[i-1]["description"])])
                 for i, stance in enumerate(stances, start=1)]
    return dict(id=case_id, question="Was the Acme Orion deal cancelled?", articles=articles,
                reference_findings=[initial], reference_limitations="Full reports were not retrieved.",
                expect_abstention=False, review=rubric,
                expected_support_statuses=[expected], expected_shared_origin=[shared],
                reference_corroboration=dict(assessments=[dict(
                    finding_index=0, assessed_claim="Acme cancelled the Orion deal.",
                    revised_text=revised, evidence_status="reported", sources=relations,
                    explanation="Curated fixture assessment; independence is not established.",
                    conflict_explanation=conflict,
                )]))


COPIED_TEXT = "Acme cancelled the Orion deal after the board reviewed the agreement at its scheduled meeting on Monday."
CORROBORATION_CASES = [
    case("corroboration_multiple", [article(1, "Acme cancelled the Orion deal."),
                                   article(2, "The Orion agreement has been terminated by Acme.")],
         "Two reports say Acme cancelled the Orion agreement.", ["supports", "supports"],
         "multiple_source_support", "Use both supporting reports; never claim they are independently verified."),
    case("corroboration_shared_origin", [article(1, COPIED_TEXT), article(2, COPIED_TEXT)],
         "Reports say Acme cancelled the Orion deal after a board review.", ["supports", "supports"],
         "single_source_reporting", "Identical syndication must not inflate source groups; flag possible shared origin.", shared=True),
    case("corroboration_conflict", [article(1, "Acme cancelled the Orion deal on 18 October 2026."),
                                   article(2, "Acme says the Orion deal remains in force on 18 October 2026.")],
         "Reports disagree on whether the Orion deal was cancelled on 18 October 2026.", ["supports", "challenges"],
         "conflicting_reports", "Preserve cancellation and denial for the same deal/date; choose no winner.",
         conflict="The agreement is described as cancelled and still in force on the same date."),
    case("corroboration_different_scopes", [article(1, "Acme cancelled its UK Orion agreement on Monday."),
                                           article(2, "Acme retained its separate US Orion agreement on Friday.")],
         "One report says Acme cancelled its UK Orion agreement on Monday.", ["supports", "context"],
         "single_source_reporting", "Separate UK and US agreements at different times are not contradictory."),
    case("corroboration_single", [article(1, "Acme cancelled the Orion deal.")],
         "One report says Acme cancelled the Orion deal.", ["supports"],
         "single_source_reporting", "Keep the useful single-source claim; do not invent corroboration or reject it solely for lacking another source."),
]
CORROBORATION_CASE_IDS = [item["id"] for item in CORROBORATION_CASES]
CORROBORATION_CASES[3]["reference_corroboration"]["assessments"][0]["assessed_claim"] = "Acme cancelled its UK Orion agreement on Monday."
