"""Context-prefill queries: query vectors of the context tokens from one prefill pass."""

import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer

from evaluation.utils import detect_user_tags
from .config import ContextPrefillConfig


class ContextPrefillQueryGenerator:
    """Extract post-RoPE query vectors of the article tokens in formatted_context."""

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        config: ContextPrefillConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

    def _extract_article_token_indices(self, formatted_context: str) -> range:
        """Token range of the article inside the first user turn of formatted_context."""
        user_start_tag, user_end_tag = detect_user_tags(formatted_context)

        user_start_pos = formatted_context.find(user_start_tag)
        if user_start_pos == -1:
            raise ValueError(f"Could not find '{user_start_tag}' tag in formatted context")

        article_text_start = formatted_context.find('\n', user_start_pos + len(user_start_tag))
        if article_text_start == -1:
            raise ValueError("Could not find newline after user start tag")
        article_text_start += 1

        article_text_end = formatted_context.find(user_end_tag, article_text_start)
        if article_text_end == -1:
            raise ValueError(f"Could not find '{user_end_tag}' tag after article content")

        prefix = formatted_context[:article_text_start]
        prefix_tokens = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        article_start_idx = prefix_tokens.input_ids.shape[1]

        context_up_to_article_end = formatted_context[:article_text_end]
        tokens_up_to_end = self.tokenizer(context_up_to_article_end, return_tensors="pt", add_special_tokens=False)
        article_end_idx = tokens_up_to_end.input_ids.shape[1]

        return range(article_start_idx, article_end_idx)

    def generate_queries(
        self,
        n_queries_per_attention_head: int,
        formatted_context: str,
        past_key_values: Optional[Tuple] = None,
        indices: Optional[range] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Return (queries, stats), queries of shape (num_layers, num_heads, n, head_dim).

        Queries start at `indices.start` if given, else at the article start,
        and are randomly subsampled to n_queries_per_attention_head.
        past_key_values is unused.
        """
        head_dim = getattr(self.model.config, 'head_dim',
                           self.model.config.hidden_size // self.model.config.num_attention_heads)

        if indices is not None:
            article_indices = indices
        else:
            article_indices = self._extract_article_token_indices(formatted_context)

        # The chat template already added <bos>.
        context_inputs = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False).to(self.device)

        query_vectors = self._extract_query_vectors_from_prefill(
            input_ids=context_inputs.input_ids,
            head_dim=head_dim,
            start_token_idx=article_indices.start,
        )

        stats = {
            'mode': 'context_prefill',
            'article_token_range': f'{article_indices.start}-{article_indices.stop}',
        }

        if query_vectors is None:
            num_layers = self.model.config.num_hidden_layers
            num_attention_heads = self.model.config.num_attention_heads
            empty_queries = torch.zeros((num_layers, num_attention_heads, 0, head_dim),
                                        device=self.device, dtype=self.dtype)
            stats['n_context_prefill_tokens_extracted'] = 0
            return empty_queries, stats

        last_n = int(getattr(self.config, 'last_n', 0) or 0)
        if last_n > 0 and query_vectors.shape[2] > last_n:
            query_vectors = query_vectors[:, :, -last_n:, :]
            stats['last_n'] = last_n
        n_extracted = query_vectors.shape[2]

        if n_extracted > n_queries_per_attention_head:
            indices = torch.randperm(n_extracted, device=query_vectors.device)[:n_queries_per_attention_head]
            indices = indices.sort()[0]
            final_queries = query_vectors[:, :, indices, :]
            stats['n_context_prefill_tokens_extracted'] = n_extracted
            stats['n_context_prefill_tokens_subsampled'] = n_queries_per_attention_head
            stats['subsample_indices'] = indices.detach().cpu().tolist()
        else:
            final_queries = query_vectors
            stats['n_context_prefill_tokens_extracted'] = n_extracted

        return final_queries, stats

    def _extract_query_vectors_from_prefill(
        self,
        input_ids: torch.Tensor,
        head_dim: int,
        start_token_idx: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Capture queries of all layers with pre-attention hooks in one forward pass.

        Returns (num_layers, num_heads, n_tokens, head_dim) for tokens from
        start_token_idx on, or None on failure.
        """
        try:
            num_layers = self.model.config.num_hidden_layers
            num_attention_heads = self.model.config.num_attention_heads

            all_layer_queries = []

            def make_hook_all_tokens(layer_idx):
                def hook_fn(module, args, kwargs):
                    hidden_states = None
                    if len(args) > 0:
                        hidden_states = args[0]
                    elif 'hidden_states' in kwargs:
                        hidden_states = kwargs['hidden_states']

                    if hidden_states is not None:
                        position_embeddings = kwargs.get('position_embeddings')

                        q = module.q_proj(hidden_states)  # (batch, seq_len, num_heads * head_dim)

                        batch_size, seq_len, _ = q.shape
                        q = q.view(batch_size, seq_len, num_attention_heads, head_dim)

                        from models.qwen3.modeling_qwen3 import Qwen3Attention
                        from models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention
                        from models.gemma3.modeling_gemma3 import Gemma3Attention
                        if isinstance(module, (Qwen3Attention, Qwen3MoeAttention, Gemma3Attention)):
                            q = module.q_norm(q)

                        q = q.transpose(1, 2)

                        if position_embeddings is not None:
                            from models.qwen3.modeling_qwen3 import rotate_half
                            cos, sin = position_embeddings
                            cos = cos.unsqueeze(1)
                            sin = sin.unsqueeze(1)
                            q = (q * cos) + (rotate_half(q) * sin)

                        q = q.transpose(1, 2)

                        all_layer_queries.append(q[0])  # (seq_len, num_attention_heads, head_dim)

                return hook_fn

            hooks = []
            try:
                for layer_idx in range(num_layers):
                    target_layer = self.model.model.layers[layer_idx].self_attn
                    handle = target_layer.register_forward_pre_hook(
                        make_hook_all_tokens(layer_idx),
                        with_kwargs=True
                    )
                    hooks.append(handle)

                with torch.no_grad():
                    self.model(
                        input_ids=input_ids,
                        use_cache=False,
                        return_dict=True,
                    )

                if all_layer_queries:
                    queries = torch.stack(all_layer_queries, dim=0)
                    queries = queries.permute(0, 2, 1, 3)

                    if start_token_idx is not None:
                        queries = queries[:, :, start_token_idx:, :]

                    return queries
                else:
                    return None

            finally:
                for handle in hooks:
                    handle.remove()

        except Exception as e:
            print(f"Warning: Failed to extract query vectors in context prefill mode: {e}")
            import traceback
            traceback.print_exc()
            return None
