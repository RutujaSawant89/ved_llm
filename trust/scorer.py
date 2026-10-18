from typing import Dict, Any, List
import logging
from rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class TrustScorer:
    """Calculates TrustLM Trust & Reliability Score based on real retrieval & verification metrics."""

    def calculate_score(
        self,
        verification_results: Dict[str, Any],
        chunks: List[RetrievedChunk],
        answer: str,
    ) -> Dict[str, Any]:
        """Calculate trust score (0-100%) and multi-dimensional evaluation metrics."""
        # Insufficient evidence fallback
        if not chunks or "not contain sufficient information" in answer.lower():
            return {
                "score": "LOW",
                "score_value": 20.0,
                "evidence_coverage": "LOW",
                "relevance_score": 15.0,
                "factual_support": 20.0,
                "consistency_score": 25.0,
                "reason": "Insufficient evidence found in uploaded documents.",
                "chart_data": [
                    {"name": "Trust Score", "value": 20.0, "max_value": 100.0},
                    {"name": "Relevance", "value": 15.0, "max_value": 100.0},
                    {"name": "Coverage", "value": 10.0, "max_value": 100.0},
                    {"name": "Factual Support", "value": 20.0, "max_value": 100.0},
                    {"name": "Consistency", "value": 25.0, "max_value": 100.0},
                ],
            }

        # 1. Retrieval Relevance Score (from real cosine similarities)
        top_scores = [c.similarity_score for c in chunks[:5] if hasattr(c, 'similarity_score')]
        if not top_scores and chunks:
            top_scores = [max(0.0, 1.0 - getattr(c, 'distance', 0.5)) for c in chunks[:5]]
        avg_sim = sum(top_scores) / len(top_scores) if top_scores else 0.5
        relevance_score = round(min(100.0, max(0.0, avg_sim * 100)), 1)

        # 2. Evidence Coverage Score
        # How well the top chunks cover the context window / evidence depth
        chunk_count = len(chunks)
        coverage_factor = min(1.0, 0.4 + (chunk_count * 0.15))
        coverage_score = round(min(100.0, relevance_score * 0.7 + coverage_factor * 30), 1)

        # 3. Factual Grounding from Claim Verification
        supported = verification_results.get("supported", 0)
        partially = verification_results.get("partially_supported", 0)
        unsupported = verification_results.get("unsupported", 0)
        total_claims = supported + partially + unsupported

        if total_claims > 0:
            factual_support = round(((supported * 1.0 + partially * 0.5) / total_claims) * 100, 1)
            if unsupported > 0:
                factual_support = max(0.0, factual_support - (unsupported * 15))
        else:
            factual_support = round(relevance_score * 0.9, 1)

        # 4. Consistency Score
        # Consistency across retrieved chunks (low variance in similarity)
        if len(top_scores) > 1:
            variance = sum((s - avg_sim) ** 2 for s in top_scores) / len(top_scores)
            consistency_score = round(max(50.0, min(100.0, (1.0 - variance * 2) * 100)), 1)
        else:
            consistency_score = 85.0

        # 5. Composite Trust Score
        # 40% Factual Grounding + 35% Relevance + 15% Coverage + 10% Consistency
        composite_trust = (
            factual_support * 0.40
            + relevance_score * 0.35
            + coverage_score * 0.15
            + consistency_score * 0.10
        )
        trust_score_pct = int(round(min(100.0, max(0.0, composite_trust))))

        coverage_label = "HIGH" if coverage_score >= 70 else "MEDIUM" if coverage_score >= 40 else "LOW"

        chart_data = [
            {"name": "Trust Score", "value": float(trust_score_pct), "max_value": 100.0},
            {"name": "Relevance", "value": relevance_score, "max_value": 100.0},
            {"name": "Coverage", "value": coverage_score, "max_value": 100.0},
            {"name": "Factual Support", "value": factual_support, "max_value": 100.0},
            {"name": "Consistency", "value": consistency_score, "max_value": 100.0},
        ]

        reason = (
            f"High trust ({trust_score_pct}%): {supported} claim(s) verified directly against "
            f"{chunk_count} chunk(s) with {relevance_score}% average relevance."
        ) if trust_score_pct >= 70 else (
            f"Moderate trust ({trust_score_pct}%): partial grounding across {chunk_count} retrieved chunk(s)."
        )

        return {
            "score": f"{trust_score_pct}%",
            "score_value": float(trust_score_pct),
            "evidence_coverage": f"{int(coverage_score)}% ({coverage_label})",
            "coverage_score": coverage_score,
            "relevance_score": relevance_score,
            "factual_support": factual_support,
            "consistency_score": consistency_score,
            "reason": reason,
            "chart_data": chart_data,
        }
