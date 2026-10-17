from typing import List, Dict, Any
import logging
from rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

class Reranker:
    """Reranks retrieved chunks based on semantic relevance, topic match, and query intent."""
    
    def rerank(self, chunks: List[RetrievedChunk], query_info: Dict[str, Any], top_k: int = 5) -> List[RetrievedChunk]:
        """Rerank chunks and return the best ones."""
        if not chunks:
            return []
            
        intent = query_info.get("intent", "factual")
        concepts = query_info.get("concepts", [])
        
        scored_chunks = []
        for chunk in chunks:
            score = chunk.similarity_score
            
            # Intent matching boost
            content_type = chunk.extra_metadata.get("content_type", "")
            if intent == "comparison" and content_type == "comparison":
                score += 0.2
            elif intent == "process" and content_type == "process":
                score += 0.2
            elif intent == "definition" and content_type == "definition":
                score += 0.2
            elif intent == "example" and content_type == "example":
                score += 0.2
                
            # Topic matching boost
            topic = chunk.extra_metadata.get("topic", "").lower()
            if any(concept in topic for concept in concepts):
                score += 0.15
                
            scored_chunks.append((score, chunk))
            
        # Sort by boosted score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"[RERANKER] Reranked {len(chunks)} chunks, keeping top {top_k}")
        
        # Return top K
        return [c[1] for c in scored_chunks[:top_k]]
