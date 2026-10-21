"""Summarization baseline: replace the context with a vLLM-generated summary of bounded length."""
import torch
from typing import Tuple, Dict, Optional, Any

from .base import FullCacheCompactionAlgorithm
from ..query_generation import QueryConfig


class SummarizeCompaction(FullCacheCompactionAlgorithm):
    """Replaces the article span of the context by its summary and returns the new text (or its cache)."""

    def __init__(
        self,
        prompt: str = "Summarize the following text:\n\n{article_text}\n\nSummary:",
        config_name: Optional[str] = None,
        return_cache: bool = False
    ):
        """
        Parameters
        ----------
        prompt : str
            Summarization prompt; must contain '{article_text}'.
        return_cache : bool
            Return a KV cache of the summarized context instead of its text
            (needed inside ChunkedCompaction).
        """
        self.prompt = prompt
        self.config_name = config_name
        self.return_cache = return_cache

        if '{article_text}' not in self.prompt:
            raise ValueError("prompt must contain '{article_text}' placeholder")

    def name(self) -> str:
        if self.config_name:
            return self.config_name
        return "summarize"

    def returns_cache(self) -> bool:
        return self.return_cache

    def requires_preextracted_cache(self) -> bool:
        return False

    def compact_kv_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        vllm_model: Optional[Any] = None,
        verbose_logging: bool = False,
        sliding_layer_indices: Optional[set] = None,
        past_key_values_for_queries: Optional[Any] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Summarize the article span `indices` of formatted_context.

        The summary may use at most target_size minus the tokens outside `indices`.
        Returns the summarized context text, or, with return_cache, its KV cache with
        beta = 0, together with stats. past_key_values and query_config are unused.
        """
        if vllm_model is None:
            raise ValueError(
                "vllm_model must be provided for summarization-based compaction. "
                "Please initialize vLLM and pass it to the compact_kv_cache method."
            )

        if indices is None:
            raise ValueError(
                "indices must be provided for summarization-based compaction. "
                "This should specify which portion of the context to summarize (typically the article)."
            )

        # past_key_values may be None, so tokenize the context.
        tokenized_full = tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
        full_token_ids = tokenized_full.input_ids[0]  # (seq_len,)
        seq_len = len(full_token_ids)

        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads

        print(f"\n{'='*60}")
        print("Summarization-based compaction")
        print(f"{'='*60}")

        article_token_ids = full_token_ids[indices.start:indices.stop]
        article_text = tokenizer.decode(article_token_ids, skip_special_tokens=True)

        print(f"Original article length: {len(article_token_ids)} tokens")
        print(f"Article text preview: {article_text[:200]}...")

        num_to_compact = len(indices)
        num_to_keep = seq_len - num_to_compact

        max_summary_tokens = target_size - num_to_keep

        if max_summary_tokens <= 0:
            raise ValueError(
                f"target_size ({target_size}) must be greater than the number of "
                f"positions to keep ({num_to_keep}). Got max_summary_tokens = {max_summary_tokens}"
            )

        print(f"Target size: {target_size} tokens (keeping {num_to_keep} unchanged, summarizing to {max_summary_tokens})")

        print(f"\nGenerating summary with vLLM (max {max_summary_tokens} tokens)...")

        # Free cached memory before vLLM wakes up.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"GPU Memory before vLLM wake: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
            print(f"GPU Memory available: {total - allocated:.2f}GB")

        vllm_model.wake_up()

        try:
            from vllm import SamplingParams

            summary_prompt_text = self.prompt.format(article_text=article_text)
            summary_messages = [{"role": "user", "content": summary_prompt_text}]
            summary_prompt = tokenizer.apply_chat_template(
                summary_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            # vLLM adds <bos> itself (Gemma).
            if summary_prompt.startswith("<bos>"):
                summary_prompt = summary_prompt[len("<bos>"):]

            sampling_params = SamplingParams(
                temperature=0.7,
                top_k=20,
                top_p=0.8,
                max_tokens=max_summary_tokens,
            ) # Default Qwen3 sampling params

            summary_outputs = vllm_model.generate([summary_prompt], sampling_params)
            summary_text = summary_outputs[0].outputs[0].text

            print(f"Generated summary: {summary_text[:200]}...")

            summary_tokens = tokenizer.encode(summary_text, add_special_tokens=False)
            print(f"Summary length: {len(summary_tokens)} tokens")

        finally:
            vllm_model.sleep()

        # Keep the chat-template text around the article.
        tokens_before_article = full_token_ids[:indices.start]
        tokens_after_article = full_token_ids[indices.stop:]

        text_before = tokenizer.decode(tokens_before_article, skip_special_tokens=False)
        text_after = tokenizer.decode(tokens_after_article, skip_special_tokens=False)

        summarized_context = text_before + summary_text + text_after

        summarized_token_ids = tokenizer.encode(summarized_context, add_special_tokens=False)
        new_seq_len = len(summarized_token_ids)

        print(f"\nReconstructed context with summary")
        print(f"Original context length: {len(full_token_ids)} tokens")
        print(f"Summarized context length: {new_seq_len} tokens")
        print(f"Compaction ratio: {seq_len / new_seq_len:.2f}x")

        stats = {
            'method': 'summarize',
            'original_seq_len': seq_len,
            'tensor_compacted_seq_len': new_seq_len,
            'effective_compacted_seq_len': new_seq_len,
            'compaction_ratio': seq_len / new_seq_len,
            'original_article_tokens': len(article_token_ids),
            'summary_tokens': len(summary_tokens),
            'article_compaction_ratio': len(article_token_ids) / len(summary_tokens),
            'effective_article_tokens': len(summary_tokens),
            'tensor_article_tokens': len(summary_tokens),
            'summary_text': summary_text,
            'per_layer_head_metrics': {},  # expected by the evaluation stats
            'summarization_params': {
                'prompt': self.prompt,
                'max_summary_tokens': max_summary_tokens,
                'target_size': target_size,
            },
        }

        if indices is not None:
            stats['is_partial_compaction'] = True
            stats['compaction_indices'] = {
                'start': indices.start,
                'end': indices.stop,
                'num_positions': len(indices),
            }
            stats['sub_target_size'] = len(summary_tokens)

        print(f"\n{'='*60}")
        print(f"Summarization complete!")
        print(f"  Original: {seq_len} tokens")
        print(f"  Summarized: {new_seq_len} tokens")
        print(f"  Compaction: {stats['compaction_ratio']:.2f}x")
        print(f"  Article: {len(article_token_ids)} → {len(summary_tokens)} tokens ({stats['article_compaction_ratio']:.2f}x)")
        print(f"{'='*60}\n")

        if not self.return_cache:
            return summarized_context, stats

        print("Extracting KV cache from summarized context...")
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        summarized_input_ids = tokenizer(
            summarized_context, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model(summarized_input_ids, use_cache=True)

        summarized_past_key_values = outputs.past_key_values

        # (K, V) -> (C1, beta, C2) with beta = 0, i.e. the cache is used as is.
        compacted_cache = []
        sliding_layer_indices = sliding_layer_indices or set()

        for layer_idx, (keys, values) in enumerate(summarized_past_key_values):
            if layer_idx in sliding_layer_indices:
                # Empty placeholders; ChunkedCompaction fills sliding-window layers.
                batch_size, num_heads_layer, _, head_dim = keys.shape
                placeholder_C1 = keys.new_zeros(batch_size, num_heads_layer, 0, head_dim)
                placeholder_beta = keys.new_zeros(batch_size, num_heads_layer, 0)
                placeholder_C2 = values.new_zeros(batch_size, num_heads_layer, 0, head_dim)
                compacted_cache.append((placeholder_C1, placeholder_beta, placeholder_C2))
            else:
                beta = torch.zeros(
                    keys.shape[0], keys.shape[1], keys.shape[2],
                    device=device, dtype=dtype
                )
                compacted_cache.append((keys, beta, values))

        print(f"Extracted KV cache with {new_seq_len} tokens")

        return tuple(compacted_cache), stats
