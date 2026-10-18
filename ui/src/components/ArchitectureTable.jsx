import { formatNumber } from '../utils/format'

export default function ArchitectureTable({ arch }) {
  const data = arch || {
    vocab_size: 50257,
    context_length: 256,
    embedding_dim: 256,
    num_heads: 8,
    num_layers: 6,
    token_embeddings: 12865792,
    positional_embeddings: 65536,
    transformer_blocks_total: 1579008,
    transformer_block_per_layer: 263168,
    final_layernorm: 256,
    lm_head_unique: 0,
    total_parameters: 14435073,
  }

  const totalParams = data.total_parameters || 14435073
  const totalM = (totalParams / 1e6).toFixed(1)

  return (
    <div>
      <div className="section-heading">Architecture</div>
      <div className="section-subheading">
        {data.num_layers || 6}-layer decoder-only Transformer · {totalM}M parameters
      </div>

      <table className="arch-table">
        <tbody>
          <tr>
            <td>Token Embedding</td>
            <td>{formatNumber(data.vocab_size)} × {data.embedding_dim}</td>
            <td>{formatNumber(data.token_embeddings)} params</td>
          </tr>
          <tr>
            <td>Positional Embedding</td>
            <td>{data.context_length} × {data.embedding_dim}</td>
            <td>{formatNumber(data.positional_embeddings)} params</td>
          </tr>
          <tr>
            <td>Transformer Block ×{data.num_layers}</td>
            <td>MHA({data.num_heads}h) + FFN({data.embedding_dim * 4})</td>
            <td>{formatNumber(data.transformer_block_per_layer)} params/block</td>
          </tr>
          <tr>
            <td>Layer Norm</td>
            <td>{data.embedding_dim}</td>
            <td>{formatNumber(data.final_layernorm)} params</td>
          </tr>
          <tr>
            <td>LM Head</td>
            <td>{data.tie_weights !== false ? '(tied)' : `${data.vocab_size} × ${data.embedding_dim}`}</td>
            <td>{formatNumber(data.lm_head_unique || 0)} params</td>
          </tr>
          <tr className="arch-total">
            <td>Total</td>
            <td></td>
            <td>{formatNumber(totalParams)} params</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
