"""
Chunked compaction for long contexts.

The full context <prefix><chunk_1>...<chunk_N><suffix> is prefilled once. Each chunk is
compacted separately, with queries generated against the KV slice <prefix><chunk_i><suffix>.
Slices keep their original RoPE positions, so no re-rotation is needed. Sliding-window
layers always keep the original cache from the full prefill.
"""
import torch
import time
from typing import Tuple, Dict, Optional, Any, List, Union, Callable

from .base import FullCacheCompactionAlgorithm
from .per_layer_head_on_policy import PerLayerHeadOnPolicyCompaction
from ..query_generation import QueryConfig
from ..chunking import ChunkingStrategy


class ChunkedCompaction(FullCacheCompactionAlgorithm):
    """Compacts each chunk of the article with an inner method and concatenates the results after the prefix."""

    def __init__(
        self,
        inner_compaction_method: Union[FullCacheCompactionAlgorithm, Callable[[], FullCacheCompactionAlgorithm]],
        chunking_strategy: ChunkingStrategy,
        chunk_system_prompt_template: str = "You are a helpful assistant. Below is a portion of {article_name}.",
        config_name: Optional[str] = None,
        raw_chunks: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        inner_compaction_method : FullCacheCompactionAlgorithm or callable
            Method applied to each chunk, or a factory that builds a fresh one per chunk.
        chunk_system_prompt_template : str
            System prompt for each chunk's query generation, with an {article_name}
            placeholder. Ignored for Gemma, which has no system prompt.
        raw_chunks : list of int, optional
            Chunk indices (0-based, in chunking order) that are not compacted: their
            original K/V are concatenated unchanged with zero bias, so the served cache
            mixes compacted and uncompacted blocks under one softmax.
        """
        self.inner_compaction_method = inner_compaction_method
        self.chunking_strategy = chunking_strategy
        self.chunk_system_prompt_template = chunk_system_prompt_template
        self.config_name = config_name
        self.raw_chunks = set(int(i) for i in raw_chunks) if raw_chunks else set()

        self._is_factory = callable(inner_compaction_method) and not isinstance(inner_compaction_method, FullCacheCompactionAlgorithm)

    def _get_inner_method(self, chunk_idx: Optional[int] = None) -> FullCacheCompactionAlgorithm:
        """Return the inner method (a fresh instance when built from a factory)."""
        if self._is_factory:
            method = self.inner_compaction_method()
            if chunk_idx is not None and hasattr(method, 'config_name') and self.config_name:
                method.config_name = f"{self.config_name}_chunk{chunk_idx}"
        else:
            method = self.inner_compaction_method

        # Text-returning methods (summarize) must return a cache inside chunking.
        if hasattr(method, 'return_cache') and not method.returns_cache():
            method.return_cache = True

        return method

    def name(self) -> str:
        if self.config_name:
            return self.config_name
        inner_name = self._get_inner_method().name()
        return f"chunked_{self.chunking_strategy.name}_{inner_name}"

    def compact_kv_cache(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        verbose_logging: bool = False,
        vllm_model: Optional[Any] = None,
        article_text: Optional[str] = None,
        article_name: Optional[str] = None,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact the article chunk by chunk.

        Parameters
        ----------
        past_key_values : tuple of (K, V)
            Cache of the full-context prefill.
        target_size : float
            Compaction ratio in (0, 1), applied to every chunk.
        indices : range, optional
            Article span in the context.
        article_text : str, optional
            Article text; extracted from formatted_context if missing.
        article_name : str, optional
            Name used in the chunk system prompt.
        """
        return self._compact_kv_cache_kv_based(
            past_key_values=past_key_values,
            target_size=target_size,
            indices=indices,
            query_config=query_config,
            model=model,
            tokenizer=tokenizer,
            formatted_context=formatted_context,
            verbose_logging=verbose_logging,
            vllm_model=vllm_model,
            article_text=article_text,
            article_name=article_name,
            sliding_layer_indices=sliding_layer_indices,
        )

    def _compact_kv_cache_kv_based(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        verbose_logging: bool = False,
        vllm_model: Optional[Any] = None,
        article_text: Optional[str] = None,
        article_name: Optional[str] = None,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Slice the full-prefill cache into chunks and compact each one.

        Per chunk, two caches are built: the raw chunk KV that gets compacted, and a
        CompactedPrefixCache of prefix + chunk + suffix (with the original sliding-window
        layers and original_seq_len for RoPE) used only for query generation.
        """
        if article_text is None:
            article_text = self._extract_article_text(formatted_context, tokenizer)

        if article_name is None:
            article_name = "the document"

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if past_key_values is None:
            raise ValueError(
                "Chunked compaction requires a pre-extracted KV cache."
            )

        num_layers = len(past_key_values)
        sliding_layer_indices = sliding_layer_indices or set()

        # Sliding-window layers may hold fewer tokens, so read a full-attention layer.
        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape

        if indices is not None:
            prefix_end = indices.start
            article_len = len(indices)
        else:
            prefix_end = 0
            article_len = seq_len

        prefix_cache = self._extract_cache_slice(past_key_values, 0, prefix_end)
        article_end = prefix_end + article_len

        sliding_window = None
        if sliding_layer_indices:
            config = getattr(model, 'config', None)
            if config is not None:
                sliding_window = getattr(config, 'sliding_window', None)

        # An absolute target is converted to a per-chunk ratio of the article length.
        if target_size >= 1:
            if article_len > 0:
                non_article_len = seq_len - article_len
                target_size = (target_size - non_article_len) / article_len
            else:
                target_size = 1.0

        target_size = min(1.0, target_size)

        print(f"\n{'='*60}")
        print(f"KV-Based Chunked Compaction")
        print(f"{'='*60}")
        print(f"Chunking strategy: {self.chunking_strategy.name}")
        print(f"Inner method: {self._get_inner_method().name()}")
        print(f"Per-chunk compaction ratio: {target_size:.2%}")
        print(f"Prefix tokens: {prefix_end}")
        print(f"Article tokens: {article_len}")
        print(f"Full sequence length: {seq_len}")

        start_time = time.time()
        chunks = self.chunking_strategy.chunk(article_text, tokenizer)
        chunk_time = time.time() - start_time
        print(f"Chunked article into {len(chunks)} chunks in {chunk_time:.2f}s")

        compacted_chunks = []
        chunk_stats = []
        total_original_tokens = 0
        total_compacted_tokens = 0
        total_effective_article_tokens = 0.0

        total_query_generation_time = 0.0
        total_inner_compaction_time = 0.0

        if self._is_gemma_model(model):
            chunk_system_prompt = None
        else:
            chunk_system_prompt = self.chunk_system_prompt_template.format(article_name=article_name)

        for chunk_idx, chunk in enumerate(chunks):
            print(f"\n--- Processing chunk {chunk_idx + 1}/{len(chunks)} ---")
            if chunk.metadata.get('note_id'):
                print(f"Note ID: {chunk.metadata['note_id']}")
            elif chunk.metadata.get('filename'):
                print(f"Filename: {chunk.metadata['filename']}")

            chunk_num_tokens = chunk.end_token_idx - chunk.start_token_idx
            chunk_start_global = prefix_end + chunk.start_token_idx
            chunk_end_global = chunk_start_global + chunk_num_tokens

            chunk_kv_raw = self._extract_cache_slice(past_key_values, chunk_start_global, chunk_end_global)

            if chunk_idx in self.raw_chunks:
                print(f"Chunk {chunk_idx} kept raw (uncompacted): {chunk_num_tokens} tokens, beta = 0")
                chunk_compacted = tuple(
                    (K, torch.zeros(K.shape[0], K.shape[1], K.shape[2], device=K.device, dtype=K.dtype), V)
                    for K, V in chunk_kv_raw
                )
                total_original_tokens += chunk_num_tokens
                total_compacted_tokens += chunk_num_tokens
                total_effective_article_tokens += chunk_num_tokens
                compacted_chunks.append(chunk_compacted)
                chunk_stats.append({
                    'raw_chunk': True,
                    'chunk_idx': chunk_idx,
                    'effective_article_tokens': chunk_num_tokens,
                    'chunk_original_tokens': chunk_num_tokens,
                    'chunk_compacted_tokens': chunk_num_tokens,
                    'chunk_timing': {'query_generation_time': 0.0, 'compaction_time': 0.0, 'total_time': 0.0},
                })
                continue

            # Query-generation cache: prefix + chunk + suffix, plus the original sliding layers.
            chunk_kv_for_queries_tuples = []
            for layer_idx in range(num_layers):
                if layer_idx in sliding_layer_indices:
                    keys = past_key_values[layer_idx][0]
                    values = past_key_values[layer_idx][1]
                else:
                    prefix_keys = prefix_cache[layer_idx][0]
                    chunk_keys = chunk_kv_raw[layer_idx][0]
                    # The suffix starts after the whole article, not after this chunk.
                    suffix_keys = past_key_values[layer_idx][0][:, :, article_end:, :]
                    keys = torch.cat([prefix_keys, chunk_keys, suffix_keys], dim=2)

                    prefix_values = prefix_cache[layer_idx][1]
                    chunk_values = chunk_kv_raw[layer_idx][1]
                    suffix_values = past_key_values[layer_idx][1][:, :, article_end:, :]
                    values = torch.cat([prefix_values, chunk_values, suffix_values], dim=2)

                beta = torch.zeros(keys.shape[0], keys.shape[1], keys.shape[2],
                                  device=device, dtype=dtype)
                chunk_kv_for_queries_tuples.append((keys, beta, values))

            # original_seq_len = seq_len places self-study tokens after the full context.
            from models.cache import CompactedPrefixCache
            chunk_kv_for_queries = CompactedPrefixCache(
                compacted_cache=tuple(chunk_kv_for_queries_tuples),
                original_seq_len=seq_len,
                sliding_layer_indices=sliding_layer_indices,
                sliding_window=sliding_window,
            )

            chunk_article_len = chunk_num_tokens
            total_original_tokens += chunk_article_len
            print(f"Chunk tokens: {chunk_article_len}")

            chunk_target = max(1, int(chunk_article_len * target_size))
            print(f"Chunk target: {chunk_target} tokens ({target_size:.2%} of {chunk_article_len})")

            chunk_compactor = self._get_inner_method(chunk_idx)

            chunk_start_time = time.time()

            # In chunk_kv_for_queries the chunk spans [prefix_end, prefix_end + chunk_article_len).
            compact_kwargs = dict(
                past_key_values=chunk_kv_raw,
                target_size=chunk_target,
                indices=range(0, chunk_article_len),
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=self._format_chunk_context(model, tokenizer, chunk.text, chunk_system_prompt),
                verbose_logging=verbose_logging,
                vllm_model=vllm_model,
                sliding_layer_indices=sliding_layer_indices,
                past_key_values_for_queries=chunk_kv_for_queries,
            )
            if isinstance(chunk_compactor, PerLayerHeadOnPolicyCompaction):
                compact_kwargs['query_cache_article_boundaries'] = (prefix_end, prefix_end + chunk_article_len)
            chunk_compacted, chunk_stat = chunk_compactor.compact_kv_cache(**compact_kwargs)

            chunk_total_time = time.time() - chunk_start_time

            chunk_query_gen_time = 0.0
            if 'query_generation' in chunk_stat and 'query_generation_time' in chunk_stat['query_generation']:
                chunk_query_gen_time = chunk_stat['query_generation']['query_generation_time']
            chunk_compaction_time = chunk_total_time - chunk_query_gen_time

            total_query_generation_time += chunk_query_gen_time
            total_inner_compaction_time += chunk_compaction_time

            # The inner cache holds only the chunk; average its length over global layers.
            compacted_article_len = 0
            num_global_layers = 0
            for layer_idx in range(num_layers):
                if layer_idx not in sliding_layer_indices:
                    compacted_article_len += chunk_compacted[layer_idx][0].shape[2]
                    num_global_layers += 1

            if num_global_layers > 0:
                compacted_article_len = compacted_article_len / num_global_layers

            total_compacted_tokens += compacted_article_len

            chunk_effective_article_tokens = chunk_stat.get('effective_article_tokens', compacted_article_len)
            total_effective_article_tokens += chunk_effective_article_tokens

            compacted_chunks.append(chunk_compacted)
            chunk_stat['chunk_idx'] = chunk_idx
            chunk_stat['chunk_original_tokens'] = chunk_article_len
            chunk_stat['chunk_compacted_tokens'] = compacted_article_len
            chunk_stat['chunk_timing'] = {
                'query_generation_time': chunk_query_gen_time,
                'compaction_time': chunk_compaction_time,
                'total_time': chunk_total_time,
            }
            chunk_stats.append(chunk_stat)

            del chunk_kv_for_queries
            del chunk_compacted

        print(f"\n--- Concatenating {len(compacted_chunks)} compacted chunks (no RoPE correction needed) ---")
        combined_cache = self._concatenate_chunks_kv_based(
            compacted_chunks, prefix_cache, device, dtype,
            sliding_layer_indices=sliding_layer_indices
        )

        num_layers = len(combined_cache)
        num_global_layers = num_layers - len(sliding_layer_indices)
        total_tensor_len = 0
        for layer_idx in range(num_layers):
            if layer_idx not in sliding_layer_indices:
                total_tensor_len += combined_cache[layer_idx][0].shape[2]

        if num_global_layers > 0:
            avg_tensor_compacted_len = total_tensor_len / num_global_layers
        else:
            avg_tensor_compacted_len = 0

        print(f"Final average cache size: {avg_tensor_compacted_len:.1f} tokens across {num_global_layers} global layers")
        print(f"Total article tokens: {total_original_tokens} -> {total_compacted_tokens}")
        print(f"Aggregated timing: query_gen={total_query_generation_time:.2f}s, "
              f"compaction={total_inner_compaction_time:.2f}s")

        effective_article_tokens = total_effective_article_tokens
        num_kept = seq_len - article_len
        effective_compacted_seq_len = effective_article_tokens + num_kept
        tensor_article_tokens = avg_tensor_compacted_len - num_kept

        all_stats = {
            'method': self.name(),
            'chunking_strategy': self.chunking_strategy.name,
            'inner_method': self._get_inner_method().name(),
            'num_chunks': len(chunks),
            'raw_chunks': sorted(self.raw_chunks),
            'tensor_compacted_seq_len': avg_tensor_compacted_len,
            'effective_article_tokens': effective_article_tokens,
            'tensor_article_tokens': tensor_article_tokens,
            'effective_compacted_seq_len': effective_compacted_seq_len,
            'total_original_article_tokens': total_original_tokens,
            'total_compacted_article_tokens': total_compacted_tokens,
            'target_ratio': target_size,
            'chunk_stats': chunk_stats,
            'per_layer_head_metrics': {},
            'query_generation': self._aggregate_query_generation_stats(chunk_stats, total_query_generation_time),
            'inner_compaction_time': total_inner_compaction_time,
        }

        return combined_cache, all_stats

    def _concatenate_chunks_kv_based(
        self,
        compacted_chunks: List[Tuple],
        prefix_cache: Tuple,
        device: torch.device,
        dtype: torch.dtype,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple:
        """Concatenate the prefix (beta = 0) and the compacted chunks, layer by layer."""
        if sliding_layer_indices is None:
            sliding_layer_indices = set()

        if not compacted_chunks:
            raise ValueError("No compacted chunks to concatenate")

        num_layers = len(compacted_chunks[0])
        combined = []

        for layer_idx in range(num_layers):
            K_prefix = prefix_cache[layer_idx][0]
            V_prefix = prefix_cache[layer_idx][1]
            beta_prefix = torch.zeros(
                K_prefix.shape[0], K_prefix.shape[1], K_prefix.shape[2],
                device=device, dtype=dtype
            )

            C1_parts = [chunk[layer_idx][0] for chunk in compacted_chunks]
            beta_parts = [chunk[layer_idx][1] for chunk in compacted_chunks]
            C2_parts = [chunk[layer_idx][2] for chunk in compacted_chunks]

            C1_combined = torch.cat([K_prefix] + C1_parts, dim=2)
            beta_combined = torch.cat([beta_prefix] + beta_parts, dim=2)
            C2_combined = torch.cat([V_prefix] + C2_parts, dim=2)

            combined.append((C1_combined, beta_combined, C2_combined))

        return tuple(combined)

    def _aggregate_query_generation_stats(self, chunk_stats: List[Dict], total_query_generation_time: float) -> Dict:
        """Sum the per-chunk query-generation stats into one dict shaped like QueryGenerator's."""
        chunks_with_qstats = [
            cs['query_generation'] for cs in chunk_stats
            if 'query_generation' in cs
        ]

        if not chunks_with_qstats:
            return {'query_generation_time': total_query_generation_time}

        total_queries_per_kv_head = sum(
            qs.get('final_n_queries_per_kv_head', 0) for qs in chunks_with_qstats
        )

        aggregated_methods = {}
        for qs in chunks_with_qstats:
            for method, method_info in qs.get('methods_used', {}).items():
                if method not in aggregated_methods:
                    aggregated_methods[method] = {
                        'n_queries_requested_per_kv_head': 0,
                        'n_queries_actual_per_kv_head': 0,
                        'fraction': 0.0,
                    }
                aggregated_methods[method]['n_queries_requested_per_kv_head'] += method_info.get('n_queries_requested_per_kv_head', 0)
                aggregated_methods[method]['n_queries_actual_per_kv_head'] += method_info.get('n_queries_actual_per_kv_head', 0)

        if total_queries_per_kv_head > 0:
            for method in aggregated_methods:
                aggregated_methods[method]['fraction'] = (
                    aggregated_methods[method]['n_queries_actual_per_kv_head'] / total_queries_per_kv_head
                )

        return {
            'query_generation_time': total_query_generation_time,
            'final_n_queries_per_kv_head': total_queries_per_kv_head,
            'methods_used': aggregated_methods,
            'num_chunks': len(chunks_with_qstats),
        }

    def _extract_article_text(self, formatted_context: str, tokenizer) -> str:
        """Return the text of the user turn of a chat-formatted context."""
        from evaluation.utils import detect_user_tags

        user_start_tag, user_end_tag = detect_user_tags(formatted_context)

        user_start_pos = formatted_context.find(user_start_tag)
        if user_start_pos == -1:
            raise ValueError(f"Could not find '{user_start_tag}' in formatted context")

        article_start = formatted_context.find('\n', user_start_pos + len(user_start_tag))
        if article_start == -1:
            raise ValueError("Could not find newline after user start tag")
        article_start += 1

        user_end_pos = formatted_context.find(user_end_tag, article_start)
        if user_end_pos == -1:
            article_end = len(formatted_context)
        else:
            article_end = user_end_pos

        return formatted_context[article_start:article_end].rstrip()

    def _is_gemma_model(self, model: Any) -> bool:
        model_name = getattr(model.config, 'name_or_path', '') or ''
        return "gemma" in model_name.lower()

    def _format_chunk_context(self, model, tokenizer, chunk_text: str, system_prompt: Optional[str]) -> str:
        """Wrap a chunk in the chat template (no system prompt for Gemma)."""
        if system_prompt is None or self._is_gemma_model(model):
            context_messages = [
                {"role": "user", "content": chunk_text}
            ]
        else:
            context_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text}
            ]
        context = tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        if context.startswith("<bos>"):
            context = context[len("<bos>"):]
        
        return context

    def _extract_cache_slice(
        self,
        cache: Tuple,
        start: int,
        end: int,
        is_compacted: bool = False
    ) -> Tuple:
        """Slice positions [start, end) of a (K, V) cache, or of a (C1, beta, C2) cache if is_compacted."""
        if start >= end:
            if is_compacted:
                return tuple([
                    (
                        layer[0][:, :, 0:0, :],
                        layer[1][:, :, 0:0],
                        layer[2][:, :, 0:0, :]
                    )
                    for layer in cache
                ])
            else:
                return tuple([
                    (layer[0][:, :, 0:0, :], layer[1][:, :, 0:0, :])
                    for layer in cache
                ])

        if is_compacted:
            return tuple([
                (
                    layer[0][:, :, start:end, :],
                    layer[1][:, :, start:end],
                    layer[2][:, :, start:end, :]
                )
                for layer in cache
            ])
        else:
            return tuple([
                (layer[0][:, :, start:end, :], layer[1][:, :, start:end, :])
                for layer in cache
            ])
