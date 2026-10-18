"""Local claim verification engine for TrustLM RAG.

Extracts factual claims from the generated answer and verifies them against
the retrieved document context using semantic and n-gram overlap.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


class Verifier:
    """Verifies generated answer claims against retrieved document context."""

    def __init__(self, support_threshold: float = 0.4, partial_threshold: float = 0.2):
        self.support_threshold = support_threshold
        self.partial_threshold = partial_threshold

    def _extract_claims(self, text: str) -> List[str]:
        """Split answer into distinct factual statements."""
        # Split into sentences or bullet points
        raw_lines = re.split(r'(?<=[.!?])\s+|\n+|- |\* ', text)
        claims = []
        for line in raw_lines:
            cleaned = line.strip()
            # Ignore short phrases or meta introductions
            if len(cleaned) < 15:
                continue
            if cleaned.lower().startswith(("based on", "according to", "here is", "in summary", "the provided")):
                # Check if it has a substantive second clause
                parts = re.split(r'[,:]', cleaned, maxsplit=1)
                if len(parts) > 1 and len(parts[1].strip()) >= 15:
                    cleaned = parts[1].strip()
                else:
                    continue
            claims.append(cleaned)
        return claims[:8]  # cap at 8 claims for speed

    def _tokenize(self, text: str) -> set[str]:
        """Simple lowercase alphanumeric token set, excluding common stopwords."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on",
            "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
            "will", "just", "don", "should", "now", "it", "its", "that", "this", "these"
        }
        tokens = set(re.findall(r'\b[a-zA-Z0-9_]{3,}\b', text.lower()))
        return tokens - stopwords

    def verify(self, answer: str, context: str) -> Dict[str, Any]:
        """Verify each claim in the answer against the retrieved context string."""
        if not answer or not context or "not contain sufficient information" in answer.lower():
            return {
                "claims": [],
                "supported": 0,
                "partially_supported": 0,
                "unsupported": 0,
                "grounding_rate": 0.0,
                "enabled": True,
                "reason": "Insufficient evidence to verify claims.",
            }

        context_tokens = self._tokenize(context)
        context_lower = context.lower()
        extracted = self._extract_claims(answer)

        if not extracted:
            # If answer is a single short sentence, treat whole answer as claim
            cleaned = answer.strip()
            if len(cleaned) >= 10:
                extracted = [cleaned]

        claims_result = []
        supported = 0
        partially_supported = 0
        unsupported = 0

        for claim in extracted:
            claim_tokens = self._tokenize(claim)
            if not claim_tokens:
                continue

            overlap = claim_tokens.intersection(context_tokens)
            overlap_ratio = len(overlap) / max(1, len(claim_tokens))

            # Also check if exact sub-phrases of 4+ words appear in context
            words = claim.split()
            phrase_matched = False
            if len(words) >= 4:
                for i in range(len(words) - 3):
                    ngram = " ".join(words[i:i+4]).lower()
                    if ngram in context_lower:
                        phrase_matched = True
                        break

            if phrase_matched or overlap_ratio >= self.support_threshold:
                status = "supported"
                supported += 1
                confidence = min(1.0, 0.7 + overlap_ratio * 0.3)
            elif overlap_ratio >= self.partial_threshold:
                status = "partially_supported"
                partially_supported += 1
                confidence = 0.5 + overlap_ratio * 0.2
            else:
                status = "unsupported"
                unsupported += 1
                confidence = max(0.1, overlap_ratio)

            claims_result.append({
                "claim": claim,
                "status": status,
                "confidence": round(confidence, 2),
            })

        total = max(1, supported + partially_supported + unsupported)
        grounding_rate = round(((supported * 1.0) + (partially_supported * 0.5)) / total, 2)

        reason = (
            f"{supported}/{len(claims_result)} statements fully supported by uploaded document evidence."
            if claims_result else "No factual claims requiring verification."
        )

        return {
            "claims": claims_result,
            "supported": supported,
            "partially_supported": partially_supported,
            "unsupported": unsupported,
            "grounding_rate": grounding_rate,
            "enabled": True,
            "reason": reason,
        }
