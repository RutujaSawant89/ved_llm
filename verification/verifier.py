from typing import Any, Dict


class Verifier:
    """Explicitly report that local claim verification is not implemented."""

    def verify(self, answer: str, context: str) -> Dict[str, Any]:
        """Return a clear disabled status rather than calling an external service."""
        _ = answer, context
        return {
            "claims": [],
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "enabled": False,
            "reason": "Local claim verification is not implemented.",
        }
