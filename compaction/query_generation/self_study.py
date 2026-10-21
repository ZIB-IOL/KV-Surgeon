# Adapted from Cartridges (Apache-2.0):
# https://github.com/HazyResearch/cartridges/blob/05f629f519662122cf465c849e421334cc6a4d97/cartridges/synthesizers/self_study.py#L37
"""Self-study queries: the model asks and answers questions about the context."""

import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer

from evaluation.utils import detect_user_tags
from .config import SelfStudyConfig
from models.cache import clone_dynamic_cache, clone_compacted_prefix_cache


def _get_sampling_params(enable_thinking: bool, max_tokens: int):
    """vLLM sampling parameters (Qwen3 defaults for thinking and non-thinking mode)."""
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("vLLM is required")

    if enable_thinking:
        return SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=max_tokens,
        )
    else:
        return SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=max_tokens,
        )


class SelfStudyQueryGenerator:
    """
    Self-study query generation.

    Model A writes conversation starters from the context, Model B answers them
    (both via vLLM), and the query vectors of each starter and answer are read
    out in a prefill pass with the HF model on top of the context cache.
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        config: SelfStudyConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
        vllm_model: Optional[Any] = None,
    ):
        """vllm_model is the shared vLLM engine; required unless no spec generates text."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.verbose = verbose
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        # vLLM is only needed if some spec generates text; config=None means
        # extraction only.
        if config is not None and vllm_model is None:
            needs_vllm = any(
                not spec.is_prefill() or not spec.is_direct()
                for spec in config.conversation_specs
            )
            assert not needs_vllm, (
                "vllm_model must be provided when conversation specs require generation. "
                "Initialize vLLM once and pass it to all SelfStudyQueryGenerator instances. "
                "If all specs use prefill_with_article=True, vLLM is not needed."
            )
        self.vllm_model = vllm_model
        if self.verbose and vllm_model is not None:
            print("Using provided vLLM model instance")

        # Models whose answers never contain </think>.
        model_name = self.model.config._name_or_path.lower()
        self._no_think_tags = (
            "llama" in model_name or
            "gemma" in model_name or
            "qwen3-4b-instruct" in model_name
        )

    def _extract_article_from_formatted_context(self, formatted_context: str) -> str:
        """Article text inside the first user turn of formatted_context."""
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

        article = formatted_context[article_text_start:article_text_end]
        return article

    def generate_queries(
        self,
        n_queries_per_attention_head: int,
        formatted_context: str,
        past_key_values: Optional[Tuple] = None,
        return_sequences: bool = False,
        indices: Optional[range] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[list]]:
        """
        Generate conversations with vLLM and extract their queries with the HF model.

        Parameters
        ----------
        past_key_values : optional
            Context cache that the extraction pass attends to.
        return_sequences : bool
            Also return per-conversation dicts (texts and suffix_token_ids) for
            on-policy re-extraction.
        indices : range, optional
            Token span used as the article for prefill specs.

        Returns
        -------
        queries : Tensor, shape (num_layers, num_heads, n, head_dim)
        stats : dict
        sequences : list or None
        """
        stats = {
            'n_conversations': 0,
            'n_self_study_tokens_extracted': 0,
            'conversation_starters_used': [],
        }
        n_queries = n_queries_per_attention_head
        specs = self.config.conversation_specs

        if self.vllm_model is not None:
            if self.verbose:
                print("Waking up vLLM for generation...")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if self.verbose:
                    print(f"GPU Memory before vLLM wake: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
                    print(f"GPU Memory available: {total - allocated:.2f}GB")

            self.vllm_model.wake_up()

        try:
            head_dim = getattr(self.model.config, 'head_dim',
                               self.model.config.hidden_size // self.model.config.num_attention_heads)

            if self.verbose:
                print(f"Generating conversations from {len(specs)} ConversationSpecs with vLLM...")

            # Phase 1: Model A writes conversation starters from seed prompts.
            specs_needing_a = [(i, spec) for i, spec in enumerate(specs) if not spec.is_direct()]

            spec_to_starters = {}  # {spec_index: [starter, ...]}

            if specs_needing_a:
                # Batch specs that share sampling settings.
                param_groups = {}
                for spec_idx, spec in specs_needing_a:
                    if spec.enable_thinking_a is None or spec.max_tokens_a is None:
                        raise ValueError(
                            f"ConversationSpec at index {spec_idx} must explicitly set "
                            f"enable_thinking_a and max_tokens_a"
                        )
                    enable_thinking = spec.enable_thinking_a
                    max_tokens = spec.max_tokens_a
                    group_key = (enable_thinking, max_tokens)
                    if group_key not in param_groups:
                        param_groups[group_key] = []
                    param_groups[group_key].append((spec_idx, spec))

                for (enable_thinking, max_tokens), group_specs in param_groups.items():
                    question_prompts = []
                    question_spec_indices = []

                    for spec_idx, spec in group_specs:
                        question_messages = [{"role": "user", "content": spec.seed_prompt}]
                        question_prompt = self.tokenizer.apply_chat_template(
                            question_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking
                        )
                        # Gemma: formatted_context already has <bos>.
                        if question_prompt.startswith("<bos>"):
                            question_prompt = question_prompt[len("<bos>"):]
                        full_prompt_a = formatted_context + question_prompt
                        question_prompts.append(full_prompt_a)
                        question_spec_indices.append(spec_idx)

                    sampling_params_a = _get_sampling_params(enable_thinking, max_tokens)
                    question_outputs = self.vllm_model.generate(question_prompts, sampling_params_a)

                    specs_to_retry = []  # [(spec_idx, spec, attempt), ...]

                    for spec_idx, output in zip(question_spec_indices, question_outputs):
                        generated_text = output.outputs[0].text
                        spec = specs[spec_idx]

                        if spec.uses_extraction():
                            starters = spec.extraction_fn(generated_text)
                            if not starters:
                                specs_to_retry.append((spec_idx, spec, 1))
                                extraction_fn_name = getattr(spec.extraction_fn, '__name__', str(spec.extraction_fn))
                                print(f"[SelfStudy] Warning: Extraction failed for spec {spec_idx} "
                                      f"(extraction_fn={extraction_fn_name}), attempt 1/3. "
                                      f"Generated text preview: {generated_text[:200]}...")
                        else:
                            starters = [generated_text]

                        spec_to_starters[spec_idx] = starters

                    # Retry failed extractions, 3 attempts in total.
                    while specs_to_retry:
                        retry_prompts = []
                        retry_info = []

                        for spec_idx, spec, attempt in specs_to_retry:
                            if attempt >= 3:
                                extraction_fn_name = getattr(spec.extraction_fn, '__name__', str(spec.extraction_fn))
                                print(f"[SelfStudy] Error: Extraction failed for spec {spec_idx} "
                                      f"(extraction_fn={extraction_fn_name}) after 3 attempts. Skipping.")
                                continue

                            question_messages = [{"role": "user", "content": spec.seed_prompt}]
                            question_prompt = self.tokenizer.apply_chat_template(
                                question_messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=enable_thinking
                            )
                            if question_prompt.startswith("<bos>"):
                                question_prompt = question_prompt[len("<bos>"):]
                            full_prompt_a = formatted_context + question_prompt
                            retry_prompts.append(full_prompt_a)
                            retry_info.append((spec_idx, spec, attempt))

                        if not retry_prompts:
                            break

                        retry_outputs = self.vllm_model.generate(retry_prompts, sampling_params_a)

                        specs_to_retry = []
                        for (spec_idx, spec, attempt), output in zip(retry_info, retry_outputs):
                            generated_text = output.outputs[0].text
                            starters = spec.extraction_fn(generated_text)

                            if starters:
                                spec_to_starters[spec_idx] = starters
                                if self.verbose:
                                    print(f"[SelfStudy] Retry succeeded for spec {spec_idx} on attempt {attempt + 1}")
                            else:
                                specs_to_retry.append((spec_idx, spec, attempt + 1))
                                extraction_fn_name = getattr(spec.extraction_fn, '__name__', str(spec.extraction_fn))
                                print(f"[SelfStudy] Warning: Extraction failed for spec {spec_idx} "
                                      f"(extraction_fn={extraction_fn_name}), attempt {attempt + 1}/3. "
                                      f"Generated text preview: {generated_text[:200]}...")

                if self.verbose:
                    total_starters = sum(len(starters) for starters in spec_to_starters.values())
                    print(f"Generated {total_starters} conversation starters from {len(specs_needing_a)} seed prompts with vLLM")

            for i, spec in enumerate(specs):
                if spec.is_direct():
                    spec_to_starters[i] = [spec.conversation_starter]

            # Phase 2: Model B answers each starter; prefill specs use the article.
            article = None
            has_prefill_specs = any(spec.is_prefill() for spec in specs)
            if has_prefill_specs:
                if indices is not None:
                    context_tokens = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False).input_ids[0]
                    article_tokens = context_tokens[indices.start:indices.stop]
                    article = self.tokenizer.decode(article_tokens, skip_special_tokens=False)
                else:
                    article = self._extract_article_from_formatted_context(formatted_context)
                if self.verbose:
                    print(f"Extracted article for prefill ({len(article)} chars)")

            all_starters = []
            starter_to_spec_idx = []
            for spec_idx in sorted(spec_to_starters.keys()):
                for starter in spec_to_starters[spec_idx]:
                    all_starters.append(starter)
                    starter_to_spec_idx.append(spec_idx)

            prefill_starters = []  # [(starter, spec_idx), ...]
            generation_starters = []

            for starter, spec_idx in zip(all_starters, starter_to_spec_idx):
                spec = specs[spec_idx]
                if spec.is_prefill():
                    prefill_starters.append((starter, spec_idx))
                else:
                    generation_starters.append((starter, spec_idx))

            starter_to_answer = {}  # {(starter, spec_idx): answer}

            if prefill_starters:
                for starter, spec_idx in prefill_starters:
                    starter_to_answer[(starter, spec_idx)] = article
                if self.verbose:
                    print(f"Prefilled {len(prefill_starters)} answers with article content (bypassing vLLM)")

            if generation_starters:
                param_groups_b = {}
                for starter, spec_idx in generation_starters:
                    spec = specs[spec_idx]
                    enable_thinking_b = spec.enable_thinking_b
                    max_tokens_b = spec.max_tokens_b
                    group_key = (enable_thinking_b, max_tokens_b)
                    if group_key not in param_groups_b:
                        param_groups_b[group_key] = []
                    param_groups_b[group_key].append((starter, spec_idx))

                for (enable_thinking_b, max_tokens_b), group_starters in param_groups_b.items():
                    answer_prompts = []
                    for starter, _ in group_starters:
                        answer_messages = [{"role": "user", "content": starter}]
                        answer_prompt = self.tokenizer.apply_chat_template(
                            answer_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking_b
                        )
                        if answer_prompt.startswith("<bos>"):
                            answer_prompt = answer_prompt[len("<bos>"):]
                        full_prompt_b = formatted_context + answer_prompt
                        answer_prompts.append(full_prompt_b)

                    sampling_params_b = _get_sampling_params(enable_thinking_b, max_tokens_b)
                    answer_outputs = self.vllm_model.generate(answer_prompts, sampling_params_b)

                    for (starter, spec_idx), output in zip(group_starters, answer_outputs):
                        starter_to_answer[(starter, spec_idx)] = output.outputs[0].text

                if self.verbose:
                    print(f"Generated {len(generation_starters)} answers with vLLM across {len(param_groups_b)} parameter groups")

            # Phase 3: extract query vectors of starter + answer with the HF model.
            all_query_vectors = []
            sequences = [] if return_sequences else None

            # The chat template already added <bos>.
            context_inputs = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False).to(self.device)
            n_context_tokens = context_inputs.input_ids.shape[1]

            total_conversations = len(all_starters)
            for idx, (starter, spec_idx) in enumerate(zip(all_starters, starter_to_spec_idx), 1):
                spec = specs[spec_idx]
                answer = starter_to_answer[(starter, spec_idx)]

                if spec.is_prefill():
                    enable_thinking_b = False
                else:
                    enable_thinking_b = spec.enable_thinking_b

                if self.verbose:
                    print(f"  [Conv {idx}/{total_conversations}] Starter: {starter}")
                    print(f"  [Conv {idx}/{total_conversations}] Answer: {answer[:100]}...")

                # Drop answers whose thinking was cut off before </think>.
                if enable_thinking_b and "</think>" not in answer and not self._no_think_tags:
                    if self.verbose:
                        print(f"  [Conv {idx}/{total_conversations}] Skipping: Model B thinking response missing </think> tag")
                    continue

                if self.verbose:
                    print(f"  [Conv {idx}/{total_conversations}] Extracting queries from answer...")

                answer_messages = [{"role": "user", "content": starter}]
                answer_prompt = self.tokenizer.apply_chat_template(
                    answer_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking_b
                )
                if answer_prompt.startswith("<bos>"):
                    answer_prompt = answer_prompt[len("<bos>"):]

                full_answer_text = formatted_context + answer_prompt + answer
                answer_inputs = self.tokenizer(full_answer_text, return_tensors="pt", add_special_tokens=False).to(self.device)

                # With a context cache, feed only the suffix so positions match inference.
                extraction_input_ids = answer_inputs.input_ids
                extraction_start_token_idx = n_context_tokens
                if past_key_values is not None:
                    if extraction_input_ids.shape[1] < n_context_tokens:
                        raise ValueError(
                            "Context tokens longer than provided input_ids; cannot align with past_key_values."
                        )
                    extraction_input_ids = extraction_input_ids[:, n_context_tokens:]
                    extraction_start_token_idx = 0

                query_vectors = self._extract_query_vectors_from_prefill(
                    input_ids=extraction_input_ids,
                    past_key_values=past_key_values,
                    head_dim=head_dim,
                    start_token_idx=extraction_start_token_idx,
                )

                if query_vectors is not None:
                    n_tokens = query_vectors.shape[2]
                    if self.verbose:
                        print(f"  [Conv {idx}/{total_conversations}] Extracted {n_tokens} tokens")
                    # Accumulate on CPU; only the subsample goes back to the GPU.
                    all_query_vectors.append(query_vectors.cpu())

                    if return_sequences:
                        suffix_token_ids = answer_inputs.input_ids[:, n_context_tokens:]  # (1, suffix_len)

                        sequences.append({
                            'full_text': full_answer_text,
                            'starter': starter,
                            'answer': answer,
                            'answer_prompt': answer_prompt,
                            'enable_thinking_b': enable_thinking_b,
                            'n_context_tokens': n_context_tokens,
                            'suffix_token_ids': suffix_token_ids,
                        })
                else:
                    if self.verbose:
                        print(f"  [Conv {idx}/{total_conversations}] Failed to extract query vectors")

                stats['conversation_starters_used'].append(starter)
                stats['n_conversations'] += 1

            if not all_query_vectors:
                raise RuntimeError(
                    "Self-study query generation failed: no query vectors extracted. "
                    "This may indicate OOM during extraction. "
                    "Try reducing batch size, max_tokens_a/max_tokens_b, or using fewer conversation specs."
                )

            concatenated_queries = torch.cat(all_query_vectors, dim=2)
            n_extracted = concatenated_queries.shape[2]

            if n_extracted > n_queries:
                indices = torch.randperm(n_extracted)[:n_queries]
                indices = indices.sort()[0]
                final_queries = concatenated_queries[:, :, indices, :].to(self.device)
                stats['n_self_study_tokens_extracted'] = n_extracted
                stats['n_self_study_tokens_subsampled'] = n_queries
                if return_sequences:
                    stats["subsample_indices"] = indices.tolist()
                if self.verbose:
                    print(f"  Subsampled {n_extracted} → {n_queries} tokens per attention head")
            else:
                final_queries = concatenated_queries.to(self.device)
                stats['n_self_study_tokens_extracted'] = n_extracted
                if self.verbose:
                    print(f"  Using all {n_extracted} extracted tokens per attention head (requested {n_queries})")

            return final_queries, stats, sequences
        finally:
            if self.vllm_model is not None:
                if self.verbose:
                    print("Putting vLLM to sleep...")
                self.vllm_model.sleep()

    def _extract_query_vectors_from_prefill(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple],
        head_dim: int,
        start_token_idx: Optional[int] = None,
        max_layer: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Capture post-RoPE queries of input_ids in one chunked prefill on a copy of past_key_values.

        Returns (layers, num_heads, n_tokens, head_dim) for tokens from
        start_token_idx on, for layers up to max_layer, or None on failure.
        """
        try:
            num_layers = self.model.config.num_hidden_layers
            num_attention_heads = self.model.config.num_attention_heads

            layers_to_process = num_layers if max_layer is None else (max_layer + 1)

            # One tensor per prefill chunk for each layer.
            all_layer_queries = {layer_idx: [] for layer_idx in range(layers_to_process)}

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

                        all_layer_queries[layer_idx].append(q[0])  # (seq_len, num_attention_heads, head_dim)

                return hook_fn

            hooks = []
            try:
                for layer_idx in range(layers_to_process):
                    target_layer = self.model.model.layers[layer_idx].self_attn
                    handle = target_layer.register_forward_pre_hook(
                        make_hook_all_tokens(layer_idx),
                        with_kwargs=True
                    )
                    hooks.append(handle)

                with torch.no_grad():
                    # The forward pass mutates the cache, so work on a copy.
                    past_kv_copy = None
                    if past_key_values is not None:
                        from models.cache import CompactedPrefixCache
                        from transformers.cache_utils import DynamicCache

                        if isinstance(past_key_values, CompactedPrefixCache):
                            past_kv_copy = clone_compacted_prefix_cache(past_key_values)
                        elif isinstance(past_key_values, DynamicCache):
                            past_kv_copy = clone_dynamic_cache(past_key_values)
                        else:
                            raise TypeError(
                                f"Unsupported cache type: {type(past_key_values)}. "
                                f"Expected CompactedPrefixCache or DynamicCache."
                            )

                    from models.generate import chunked_prefill
                    kv_seq_len = past_kv_copy.get_seq_length()
                    if kv_seq_len > 20000:
                        chunk_size = 256
                    elif kv_seq_len > 10000:
                        chunk_size = 1024
                    else:
                        chunk_size = 4096
                    chunked_prefill(
                        self.model,
                        input_ids=input_ids,
                        past_key_values=past_kv_copy,
                        max_layer=max_layer,
                        chunk_size=chunk_size,
                    )

                if all_layer_queries and any(len(v) > 0 for v in all_layer_queries.values()):
                    layer_queries = []
                    for layer_idx in range(layers_to_process):
                        layer_query_list = all_layer_queries[layer_idx]
                        if len(layer_query_list) == 1:
                            layer_queries.append(layer_query_list[0])
                        else:
                            layer_queries.append(torch.cat(layer_query_list, dim=0))

                    queries = torch.stack(layer_queries, dim=0)
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
            print(f"Warning: Failed to extract query vectors in prefill mode: {e}")
            import traceback
            traceback.print_exc()
            return None
