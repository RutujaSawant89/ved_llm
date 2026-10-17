export default function CheckpointLine() {
  return (
    <div className="checkpoint-line">
      <span style={{ color: 'var(--green)' }}>
        RAG Engine: Active · ChromaDB Semantic Retrieval & Trust Verification Ready
      </span>
      <span className="sep" style={{ color: 'var(--text-muted)' }}>·</span>
      <span style={{ color: 'var(--text-muted)' }}>MiniLM-L6-v2 Embeddings</span>
    </div>
  )
}
