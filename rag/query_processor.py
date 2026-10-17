from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Lightweight query understanding layer before retrieval."""
    
    def process(self, query: str) -> Dict[str, Any]:
        """Identify intent and important concepts from the user query."""
        query_lower = query.lower()
        
        # Determine intent heuristically
        intent = "factual"
        if "vs" in query_lower or "compare" in query_lower or "difference" in query_lower:
            intent = "comparison"
        elif "explain" in query_lower and ("how" in query_lower or "process" in query_lower or "flow" in query_lower):
            intent = "process"
        elif "explain" in query_lower:
            intent = "explanation"
        elif "example" in query_lower:
            intent = "example"
        elif "advantage" in query_lower or "benefit" in query_lower:
            intent = "advantages/disadvantages"
        elif "disadvantage" in query_lower or "drawback" in query_lower:
            intent = "advantages/disadvantages"
        elif "application" in query_lower or "use case" in query_lower:
            intent = "application"
        elif "what is" in query_lower or "define" in query_lower:
            intent = "definition"
        elif "and" in query_lower or "," in query_lower:
            # Simple heuristic for multi-concept synthesis
            intent = "multi-concept synthesis"
            
        # Basic keyword extraction (remove stop words in a real system)
        stop_words = {"what", "is", "the", "how", "does", "explain", "compare", "difference", "between", "and", "or", "in", "of", "a", "an", "to"}
        words = [w.strip("?,.") for w in query_lower.split()]
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        
        logger.info(f"[QUERY PROCESSOR] Intent: {intent}, Concepts: {concepts}")
        
        return {
            "original_query": query,
            "intent": intent,
            "concepts": concepts
        }
