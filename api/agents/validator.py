"""
Validator Agent: Verifies synthesized outputs and reduces hallucinations.
Checks factual grounding, confidence thresholds, and response completeness.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from api.agents.analyzer import AnalyzedQuery, QueryIntent
from api.agents.synthesizer import SynthesizedResponse

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.3
MAX_UNGROUNDED_RATIO = 0.5


@dataclass
class ValidationReport:
    """Report from the validation agent."""
    is_valid: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrected_response: Optional[SynthesizedResponse] = None
    validation_steps: List[str] = field(default_factory=list)


class ValidatorAgent:
    """
    Agent 5: Validates synthesized responses.
    Checks:
    - Entity grounding (entities exist in retrieved data)
    - Confidence thresholds
    - Response completeness
    - Factual consistency between structured data and NL insight
    """

    def validate(
        self,
        analyzed: AnalyzedQuery,
        response: SynthesizedResponse,
    ) -> ValidationReport:
        """Run all validation checks."""
        report = ValidationReport(is_valid=True, confidence=response.confidence)
        issues = []
        warnings = []
        steps = []

        # ── Check 1: Data completeness ──
        if not response.structured_data:
            if analyzed.intent not in (QueryIntent.EXPLAIN_CONNECTION, QueryIntent.USER_PROFILE):
                issues.append("No structured data returned from retrieval")
        steps.append("data_completeness_check")

        # ── Check 2: Entity grounding ──
        ungrounded = self._check_entity_grounding(response)
        if ungrounded:
            warnings.append(f"NL insight may reference entities not in retrieved data")
        steps.append("entity_grounding_check")

        # ── Check 3: Confidence threshold ──
        avg_confidence = self._compute_confidence(response)
        if avg_confidence < MIN_CONFIDENCE:
            warnings.append(f"Low average confidence: {avg_confidence:.2f}")
            response.confidence = avg_confidence
        steps.append("confidence_threshold_check")

        # ── Check 4: Response length sanity ──
        if len(response.natural_language_insight) < 10:
            warnings.append("Natural language insight is very short")
        if len(response.natural_language_insight) > 2000:
            warnings.append("Natural language insight is very long, may be hallucinating")
            # Truncate to prevent runaway outputs
            response.natural_language_insight = response.natural_language_insight[:1500] + "..."
        steps.append("response_length_check")

        # ── Check 5: Source validation ──
        if not response.sources:
            warnings.append("No retrieval sources identified — response may be unsupported")
        steps.append("source_validation_check")

        # ── Check 6: GNN score consistency ──
        gnn_issues = self._validate_gnn_scores(response)
        if gnn_issues:
            warnings.extend(gnn_issues)
        steps.append("gnn_score_consistency_check")

        # ── Check 7: Deduplication ──
        response = self._deduplicate_entities(response)
        steps.append("deduplication_check")

        # Determine overall validity
        report.is_valid = len(issues) == 0
        report.issues = issues
        report.warnings = warnings
        report.validation_steps = steps
        report.confidence = avg_confidence
        report.corrected_response = response

        if not report.is_valid:
            logger.warning(f"Validation failed: {issues}")
        elif warnings:
            logger.debug(f"Validation warnings: {warnings}")

        return report

    def _check_entity_grounding(self, response: SynthesizedResponse) -> bool:
        """Check if NL insight references entities that don't exist in structured data."""
        if not response.natural_language_insight or not response.structured_data:
            return False

        known_names = set()
        for entity in response.structured_data:
            if "name" in entity:
                known_names.add(entity["name"].lower())
            if "title" in entity:
                known_names.add(entity["title"][:20].lower())

        # Simple heuristic: check if insight mentions clearly fabricated names
        # (In production, this would be more sophisticated NLI-based checking)
        insight_words = set(response.natural_language_insight.lower().split())

        # For now, just flag if no known names appear in the insight when data exists
        if known_names and not any(name.split()[0] in insight_words for name in known_names if name):
            return True  # Potentially ungrounded

        return False

    def _compute_confidence(self, response: SynthesizedResponse) -> float:
        """Compute average confidence from GNN predictions and fusion scores."""
        scores = []

        for entity in response.structured_data:
            if "gnn_score" in entity:
                scores.append(float(entity["gnn_score"]))
            elif "fusion_score" in entity:
                scores.append(min(1.0, float(entity["fusion_score"]) * 100))
            elif "similarity_score" in entity:
                scores.append(float(entity["similarity_score"]))

        for pred in response.gnn_predictions:
            if "confidence" in pred:
                scores.append(float(pred["confidence"]))
            elif "probability" in pred:
                scores.append(float(pred["probability"]))

        return round(sum(scores) / len(scores), 4) if scores else response.confidence

    def _validate_gnn_scores(self, response: SynthesizedResponse) -> List[str]:
        """Check for anomalous GNN scores."""
        issues = []
        for pred in response.gnn_predictions:
            score = pred.get("probability", pred.get("confidence", 0.5))
            try:
                score = float(score)
                if score < 0 or score > 1:
                    issues.append(f"Out-of-range GNN score: {score}")
            except (TypeError, ValueError):
                issues.append(f"Non-numeric GNN score: {score}")
        return issues

    def _deduplicate_entities(self, response: SynthesizedResponse) -> SynthesizedResponse:
        """Remove duplicate entities from structured data."""
        seen_ids = set()
        deduped = []
        for entity in response.structured_data:
            eid = entity.get("id") or entity.get("user_id") or entity.get("post_id", "")
            if eid not in seen_ids:
                seen_ids.add(eid)
                deduped.append(entity)
        response.structured_data = deduped
        return response

    def format_final_response(
        self,
        analyzed: AnalyzedQuery,
        response: SynthesizedResponse,
        report: ValidationReport,
    ) -> Dict[str, Any]:
        """Format the validated response into the final API output."""
        final = response if report.corrected_response is None else report.corrected_response
        return {
            "intent": final.intent,
            "results": final.structured_data,
            "gnn_predictions": final.gnn_predictions,
            "insight": final.natural_language_insight,
            "graph_context": final.graph_context_summary,
            "retrieval_mode": str(final.retrieval_mode),
            "sources": final.sources,
            "validation": {
                "is_valid": report.is_valid,
                "confidence": report.confidence,
                "warnings": report.warnings,
                "issues": report.issues,
            },
            "query": analyzed.raw_query,
        }
