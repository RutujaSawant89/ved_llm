from typing import Dict, Any, List
import logging
from rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

class TrustScorer:
    """Calculates the TrustLM Trust Score based on verification results and evidence."""
    
    def calculate_score(self, verification_results: Dict[str, Any], chunks: List[RetrievedChunk], answer: str) -> Dict[str, Any]:
        """Calculate trust score (0-100%)."""
        
        # If insufficient evidence fallback was triggered
        if "The provided notes do not contain sufficient information" in answer:
            return {
                "score": "LOW",
                "evidence_coverage": "LOW",
                "reason": "Insufficient evidence"
            }
            
        claims = verification_results.get("claims", [])
        if not claims:
            # If no claims extracted but it wasn't the fallback, it might be a generic answer
            return {
                "score": "UNKNOWN",
                "evidence_coverage": "UNKNOWN",
                "reason": "No factual claims detected"
            }
            
        supported = verification_results.get("supported", 0)
        partially_supported = verification_results.get("partially_supported", 0)
        unsupported = verification_results.get("unsupported", 0)
        
        total_claims = supported + partially_supported + unsupported
        
        # Base score from claims
        claim_score = 0.0
        if total_claims > 0:
            claim_score = ((supported * 1.0) + (partially_supported * 0.5)) / total_claims
            
        # Evidence coverage (heuristic based on top distances)
        # In a real system, this might be calculated by how many chunks are highly relevant
        top_dists = [c.distance for c in chunks[:3]]
        avg_dist = sum(top_dists) / len(top_dists) if top_dists else 1.0
        
        # Lower distance is better (0.0 is perfect match)
        # Convert avg_dist to a 0-1 coverage score roughly (dist usually 0.2-0.8 for MiniLM)
        coverage_score = max(0.0, 1.0 - avg_dist)
        
        # Weighting: Claims are more important than pure semantic coverage
        final_score_pct = int((claim_score * 0.7 + coverage_score * 0.3) * 100)
        
        # If there are unsupported claims, penalize heavily
        if unsupported > 0:
            final_score_pct = max(0, final_score_pct - (unsupported * 20))
            
        coverage_label = "HIGH" if coverage_score >= 0.7 else "MEDIUM" if coverage_score >= 0.4 else "LOW"
        
        return {
            "score": f"{final_score_pct}%",
            "evidence_coverage": f"{int(coverage_score * 100)}% ({coverage_label})",
            "reason": "Based on claim verification and semantic relevance"
        }
