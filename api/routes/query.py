"""Query route — RAG pipeline query with empty KB & low relevance safeguards."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceInfo,
    TrustInfo,
    VerificationInfo,
)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def run_query(body: QueryRequest, request: Request):
    pipeline = request.app.state.pipeline

    # 1. Guard against empty knowledge base
    if pipeline.vector_store.count() == 0:
        return QueryResponse(
            answer="No documents have been uploaded to the knowledge base yet. Please upload your documents in the Documents tab first so TrustLM can retrieve and evaluate answers against them.",
            sources=[],
            trust=TrustInfo(
                score="0%",
                score_value=0.0,
                evidence_coverage="0% (NONE)",
                reason="Knowledge base is empty. Please upload documents in the Documents tab.",
                relevance_score=0.0,
                factual_support=0.0,
                consistency_score=0.0,
                chart_data=[
                    {"name": "Trust Score", "value": 0.0, "max_value": 100.0},
                    {"name": "Relevance", "value": 0.0, "max_value": 100.0},
                    {"name": "Coverage", "value": 0.0, "max_value": 100.0},
                    {"name": "Factual Support", "value": 0.0, "max_value": 100.0},
                    {"name": "Consistency", "value": 0.0, "max_value": 100.0},
                ],
            ),
            verification=VerificationInfo(
                enabled=True,
                reason="No documents available to verify claims.",
                supported=0,
                partially_supported=0,
                unsupported=0,
                grounding_rate=0.0,
                claims=[],
            ),
            generator=pipeline.generator.name,
            generation_available=True,
        )

    # 2. Run query through RAG pipeline
    result = pipeline.query(body.question, top_k=body.top_k)

    sources = []
    for chunk in result.retrieved_chunks:
        extra = chunk.extra_metadata or {}
        sources.append(
            SourceInfo(
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                relevance=round(chunk.similarity_score * 100, 1),
                text=chunk.text[:300],
                module=extra.get("module", ""),
                topic=extra.get("topic", ""),
            )
        )

    # 3. Guard against irrelevant queries with low retrieval relevance
    max_sim = max([c.similarity_score for c in result.retrieved_chunks], default=0.0)
    if max_sim < 0.25:
        rel_pct = round(max_sim * 100, 1)
        return QueryResponse(
            answer=f"The uploaded document does not contain sufficient information to answer '{body.question}'. The highest retrieval relevance score was only {rel_pct}%.",
            sources=sources,
            trust=TrustInfo(
                score="LOW",
                score_value=15.0,
                evidence_coverage="LOW",
                reason=f"Retrieved passages have low relevance ({rel_pct}%) to this question.",
                relevance_score=rel_pct,
                factual_support=10.0,
                consistency_score=20.0,
                chart_data=[
                    {"name": "Trust Score", "value": 15.0, "max_value": 100.0},
                    {"name": "Relevance", "value": rel_pct, "max_value": 100.0},
                    {"name": "Coverage", "value": 10.0, "max_value": 100.0},
                    {"name": "Factual Support", "value": 10.0, "max_value": 100.0},
                    {"name": "Consistency", "value": 20.0, "max_value": 100.0},
                ],
            ),
            verification=VerificationInfo(
                enabled=True,
                reason="Insufficient evidence to verify claims.",
                supported=0,
                partially_supported=0,
                unsupported=0,
                grounding_rate=0.0,
                claims=[],
            ),
            generator=result.generator_name,
            generation_available=True,
        )

    verification_data = result.metadata.get("verification", {})
    trust_data = result.metadata.get("trust", {})

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        trust=TrustInfo(
            score=trust_data.get("score", "UNKNOWN"),
            score_value=float(trust_data.get("score_value", 0.0)),
            evidence_coverage=trust_data.get("evidence_coverage", "UNKNOWN"),
            reason=trust_data.get("reason", ""),
            relevance_score=float(trust_data.get("relevance_score", 0.0)),
            factual_support=float(trust_data.get("factual_support", 0.0)),
            consistency_score=float(trust_data.get("consistency_score", 0.0)),
            chart_data=trust_data.get("chart_data", []),
        ),
        verification=VerificationInfo(
            enabled=verification_data.get("enabled", True),
            reason=verification_data.get("reason", ""),
            supported=verification_data.get("supported", 0),
            partially_supported=verification_data.get("partially_supported", 0),
            unsupported=verification_data.get("unsupported", 0),
            grounding_rate=float(verification_data.get("grounding_rate", 1.0)),
            claims=verification_data.get("claims", []),
        ),
        generator=result.generator_name,
        generation_available=True,
    )
