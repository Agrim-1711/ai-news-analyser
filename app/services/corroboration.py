"""Validate assessed relationships and group obvious copies; never certify truth."""
import re
from difflib import SequenceMatcher

from app.schemas import (
    ClaimSupport, CorroborationReview, ResearchDraft, ResearchEvidence,
    ResearchResultFinding, SharedOriginEvidence,
)
from app.services.evidence import align_evidence_fields, expand_description_excerpts, validate_evidence
from app.services.quality import AnswerQualityError, validate_synthesis


def _normalise(text):
    return " ".join(re.findall(r"\w+", (text or "").casefold()))


def _near_copy(left, right):
    a, b = _normalise(left.description), _normalise(right.description)
    # Short facts often match naturally; do not infer syndication from those.
    return bool(len(a.split()) >= 12 and len(b.split()) >= 12
                and SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.92)


def _attribution_quote(quote, origin_name, sources, publisher_origin):
    source = sources.get(quote.source_id)
    if source is None:
        return False
    excerpt = " ".join(quote.excerpt.split())
    available = " ".join((getattr(source, quote.field) or "").split())
    if not excerpt or excerpt not in available:
        return False
    plain_name = origin_name.strip()
    # Publisher metadata often includes "The" while a report's attribution omits it.
    name = (r"(?:the\s+)?" + re.escape(plain_name[4:])
            if plain_name.casefold().startswith("the ") else re.escape(plain_name))
    # A company 'saying' the same thing is agreement, not proof that outlets
    # reused a report. Require a report/release attribution or a wire byline.
    patterns = [
        rf"\b{name}(?:['’]s)?\s+(?:report|reporting|press release|announcement)\b",
        rf"\b(?:report by|report from|reporting by|press release from|announcement from)\s+{name}\b",
        rf"\({name}\)\s*[-–—]",
    ]
    if publisher_origin:
        patterns.append(rf"\b(?:according to|reported by|via|reprinted from|republished from|syndicated by)\s+{name}\b")
        patterns.append(rf"\b{name}\s+(?:(?:newspaper|news agency)\s+)?(?:reported|reports)\b")
    for pattern in patterns:
        for match in re.finditer(pattern, excerpt, flags=re.IGNORECASE):
            before = excerpt[max(0, match.start() - 70):match.start()]
            after = excerpt[match.end():match.end() + 70]
            negated = re.search(r"\b(?:not|no|never|without|denies|denied)\b[^.!?;:,]{0,55}$", before, re.IGNORECASE)
            denied_use = re.match(r"\s+(?:was|is|were|are)\s+(?:not|never)\s+(?:used|cited|referenced|reprinted|republished)", after, re.IGNORECASE)
            if not negated and not denied_use:
                return True
    return False


def _validated_origins(assessments, sources):
    """Ignore unsupported origin guesses without discarding valid claim evidence."""
    relations = [item.model_copy(deep=True) for item in assessments]
    # Explicit attribution to another assessed publisher does not depend on the
    # model volunteering an origin link. Keep the actual retrieved passage as proof.
    for item in relations:
        for other in relations:
            if item.source_id == other.source_id:
                continue
            publisher = sources[other.source_id].source
            if not publisher or len(_normalise(publisher)) < 3:
                continue
            source = sources[item.source_id]
            for field in ("title", "description"):
                text = getattr(source, field)
                if not text or not text.strip():
                    continue
                quote = ResearchEvidence(source_id=item.source_id, field=field, excerpt=text)
                if _attribution_quote(quote, publisher, sources, publisher_origin=True):
                    if other.source_id not in item.shared_origin_with:
                        item.shared_origin_with.append(other.source_id)
                    item.origin_evidence.append(SharedOriginEvidence(
                        other_source_id=other.source_id, origin_name=publisher, evidence=[quote],
                    ))
                    break
    notes = {}
    for item in relations:
        accepted, clues = [], []
        for other in item.shared_origin_with:
            pair = tuple(sorted((item.source_id, other)))
            if _near_copy(sources[item.source_id], sources[other]):
                accepted.append(other)
                notes[pair] = "Descriptions are near-identical; a shared origin is possible."
                continue
            for clue in item.origin_evidence:
                origin = clue.origin_name.strip()
                if clue.other_source_id != other or _normalise(origin) in {
                    "report", "statement", "announcement", "press release", "company", "sources", "news",
                }:
                    continue
                ids = {item.source_id, other}
                publisher_origin = any(_normalise(sources[sid].source) == _normalise(origin) for sid in ids)
                if any(quote.source_id not in ids or not _attribution_quote(quote, origin, sources, publisher_origin)
                       for quote in clue.evidence):
                    continue
                # Both sides need retrieved attribution. A primary publisher's
                # own name can establish its side; a model explanation cannot.
                attributed = {quote.source_id for quote in clue.evidence}
                attributed.update(sid for sid in ids
                                  if _normalise(sources[sid].source) == _normalise(origin))
                if attributed == ids:
                    accepted.append(other)
                    clues.append(clue)
                    notes[pair] = f"Retrieved attribution links both reports to {origin}."
                    break
        item.shared_origin_with = accepted
        item.origin_evidence = clues
        item.origin_reason = " ".join(dict.fromkeys(notes[tuple(sorted((item.source_id, other)))]
                                                    for other in accepted)) or None
    return relations, notes


def _groups(assessments, sources, origin_notes):
    ids = [item.source_id for item in assessments]
    parent = {sid: sid for sid in ids}
    possible_shared_origin = False
    notes = {}

    def root(sid):
        while parent[sid] != sid:
            sid = parent[sid]
        return sid

    def join(left, right):
        parent[root(right)] = root(left)

    for index, sid in enumerate(ids):
        left = sources[sid]
        item = assessments[index]
        for other in item.shared_origin_with:
            join(sid, other)
            possible_shared_origin = True
            pair = tuple(sorted((sid, other)))
            notes[pair] = origin_notes[pair]
        for other in ids[:index]:
            right = sources[other]
            same_publisher = (left.source and right.source
                              and left.source.strip().casefold() == right.source.strip().casefold())
            same_host = left.url.host == right.url.host
            copied = _near_copy(left, right)
            if same_publisher or same_host or copied:
                join(sid, other)
                pair = tuple(sorted((sid, other)))
                if copied:
                    notes.setdefault(pair, "Descriptions are near-identical; a shared origin is possible.")
                elif same_publisher:
                    notes.setdefault(pair, "Grouped under the same publisher; this does not establish reuse of a specific report.")
                else:
                    notes.setdefault(pair, "Grouped under the same website; this does not establish reuse of a specific report.")
            possible_shared_origin = possible_shared_origin or copied
    grouped = {}
    for sid in ids:
        grouped.setdefault(root(sid), []).append(sid)
    return list(grouped.values()), possible_shared_origin, [
        f"{left} / {right}: {reason}" for (left, right), reason in notes.items()
    ]


def apply_corroboration(draft: ResearchDraft, review: CorroborationReview, sources):
    """Reject invalid relationships; derive display labels without trusting model counts."""
    indexes = [item.finding_index for item in review.assessments]
    if len(indexes) != len(set(indexes)) or set(indexes) != set(range(len(draft.findings))):
        raise AnswerQualityError("Corroboration must assess every finding exactly once")
    findings = []
    for assessment in sorted(review.assessments, key=lambda item: item.finding_index):
        original = draft.findings[assessment.finding_index]
        ids = [item.source_id for item in assessment.sources]
        if len(ids) != len(set(ids)) or not set(ids).issubset(sources):
            raise AnswerQualityError("Corroboration cites duplicate or unknown sources")
        # Explicit abstention is allowed, never a mandatory second source.
        if not ids:
            continue
        if not set(original.source_ids).issubset(ids):
            raise AnswerQualityError("Corroboration must account for the original cited sources")
        supporting = [item.source_id for item in assessment.sources if item.stance == "supports"]
        challenging = [item.source_id for item in assessment.sources if item.stance == "challenges"]
        context = [item.source_id for item in assessment.sources if item.stance == "context"]
        if not supporting and not challenging:
            # A context-only assessment is an honest lack of support, not a
            # malformed review requiring another model call for every finding.
            continue
        if not supporting:
            raise AnswerQualityError("A retained finding needs supporting evidence")
        if challenging and not (assessment.conflict_explanation or "").strip():
            raise AnswerQualityError("Conflicting reports need an explanation of the incompatibility")
        for item in assessment.sources:
            if {quote.source_id for quote in item.evidence} != {item.source_id}:
                raise AnswerQualityError("Relationship evidence must belong to its assessed source")
            if (len(item.shared_origin_with) != len(set(item.shared_origin_with))
                    or not set(item.shared_origin_with).issubset(set(ids) - {item.source_id})):
                raise AnswerQualityError("Shared-origin links must reference other assessed sources")
        result = ResearchResultFinding(
            text=assessment.revised_text, source_ids=ids, evidence_status=assessment.evidence_status,
            evidence=[quote for item in assessment.sources for quote in item.evidence],
        )
        quoted = ResearchDraft(findings=[result], limitations="Relationship validation.")
        quoted = align_evidence_fields(quoted, sources.values())
        validate_evidence(quoted, sources.values())
        quoted = expand_description_excerpts(quoted, sources.values())
        validate_evidence(quoted, sources.values())
        result.evidence = quoted.findings[0].evidence
        # Keep relationship quotes consistent with the final displayed excerpts.
        relations, origin_notes = _validated_origins(assessment.sources, sources)
        for item in relations:
            item.evidence = [quote for quote in result.evidence if quote.source_id == item.source_id]
        groups, shared, grouping_notes = _groups(relations, sources, origin_notes)
        support_groups = [[sid for sid in group if sid in supporting] for group in groups]
        support_groups = [group for group in support_groups if group]
        status = ("conflicting_reports" if challenging else
                  "multiple_source_support" if len(support_groups) >= 2 else "single_source_reporting")
        result.corroboration = ClaimSupport(
            assessed_claim=assessment.assessed_claim,
            status=status, supporting_source_ids=supporting, challenging_source_ids=challenging,
            context_source_ids=context,
            supporting_groups=support_groups, possible_shared_origin=shared,
            grouping_notes=grouping_notes,
            explanation=assessment.explanation,
            conflict_explanation=assessment.conflict_explanation if challenging else None,
            source_assessments=relations,
        )
        findings.append(result)
    validate_synthesis(ResearchDraft(findings=findings, limitations="Final validation."), sources)
    return findings
