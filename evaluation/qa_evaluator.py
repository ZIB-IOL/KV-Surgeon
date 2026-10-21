"""QA evaluation of KV cache compaction methods: accuracy, perplexity, timing and memory."""
import torch
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from compaction.compaction_methods import FullCacheCompactionAlgorithm
from models.generate import (
    generate_with_compacted_cache_batch,
    generate_with_vllm_batch,
    get_generation_params,
)
from compaction.query_generation import QueryConfig
from .utils import (
    load_model_and_tokenizer,
    initialize_vllm,
    extract_full_kv_cache,
    format_question,
    parse_model_choice,
    compute_cache_memory_stats,
    print_query_generation_stats,
    offload_model_to_cpu,
    reload_model_to_gpu,
    compute_article_indices,
)
from .datasets import load_dataset, is_string_match_dataset


def strip_thinking(text: str) -> str:
    """Return the text after ``</think>``, or the whole text if there is none."""
    if '</think>' in text:
        return text.split('</think>')[-1].strip()
    return text


def lca_first_code_line(pred: str) -> str:
    """First non-empty line of a code completion, skipping markdown fences and thinking."""
    pred = strip_thinking(pred)
    for line in pred.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('```'):
            continue
        return stripped
    return ''


def lca_line_match(pred: str, refs: List[str]) -> float:
    """Exact match of the first generated code line against the gold line (both stripped)."""
    line = lca_first_code_line(pred)
    return 1.0 if any(line == r.strip() for r in refs) else 0.0


def score_string_match(pred: str, refs: List[str], task: str) -> float:
    """Score a string-match prediction with the scorer selected by the question's task."""
    if task.startswith('lca'):
        return lca_line_match(pred, refs)
    raise ValueError(f"No string-match scorer for task '{task}'")


class QAEvaluator:
    """Evaluate KV cache compaction methods on QA datasets."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = None,
        dtype: Optional[torch.dtype] = None,
        max_model_len: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        max_model_len : int, optional
            Maximum context length; None uses the model default.
        seed : int, optional
            vLLM engine seed. Global seeding is left to the caller.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.seed = seed
        self.model = None
        self.tokenizer = None
        self.vllm_model = None

    def _evaluate_questions_batched(
        self,
        questions: List[Dict],
        cache_for_generation: List,
        seq_len: int,
        max_new_tokens: int,
        batch_size: int,
        results_per_question: List[Dict],
        is_string_match_eval: bool = False,
    ):
        """Answer questions in batches from the compacted cache, appending to ``results_per_question``.

        ``seq_len`` is the original sequence length (for RoPE positions).
        """
        num_questions = len(questions)

        for batch_start in range(0, num_questions, batch_size):
            batch_end = min(batch_start + batch_size, num_questions)
            batch_questions = questions[batch_start:batch_end]
            actual_batch_size = len(batch_questions)

            if hasattr(self, '_finished_sequences_set'):
                self._finished_sequences_set.clear()
            if hasattr(self, '_prefill_attention_mask'):
                self._prefill_attention_mask[0] = None

            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(num_questions + batch_size - 1)//batch_size} "
                  f"(questions {batch_start+1}-{batch_end})")

            question_texts = [q['question'] for q in batch_questions]
            options_list = [q.get('options', None) for q in batch_questions]
            gold_labels = [q.get('gold_label', None) for q in batch_questions]

            for i, (q_text, opts, g_label) in enumerate(zip(question_texts, options_list, gold_labels)):
                q_idx = batch_start + i
                print(f"\nQ{q_idx+1}: {q_text[:200]}")
                if is_string_match_eval:
                    refs = batch_questions[i].get('answers', [])
                    print(f"  Expected answers: {refs}")
                elif opts:
                    for idx, opt in enumerate(opts):
                        print(f"  {chr(65+idx)}. {opt}")
                    print(f"  Gold answer: {chr(64+g_label)} ({g_label})")

            # String-match questions: thinking off, answer_prefix appended.
            if is_string_match_eval:
                formatted_questions = [
                    format_question(
                        self.tokenizer, q['question'], options=None,
                        model_name=self.model_name, enable_thinking=False,
                        answer_prefix=q.get('answer_prefix', ''),
                    )
                    for q in batch_questions
                ]
                batch_max_new_tokens = int(batch_questions[0].get('max_new_tokens', max_new_tokens))
            else:
                formatted_questions = [format_question(self.tokenizer, q_text, opts, self.model_name)
                                     for q_text, opts in zip(question_texts, options_list)]
                batch_max_new_tokens = max_new_tokens

            gen_start_time = time.time()
            device = next(self.model.parameters()).device
            compacted_cache_gpu = tuple([
                (c1.to(device), beta.to(device), c2.to(device))
                for c1, beta, c2 in cache_for_generation
            ])

            answers = generate_with_compacted_cache_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=formatted_questions,
                compacted_cache=compacted_cache_gpu,
                max_new_tokens=batch_max_new_tokens,
                original_seq_len=seq_len,
            )
            del compacted_cache_gpu

            gen_time = time.time() - gen_start_time

            total_tokens_in_batch = 0
            batch_token_counts = []
            for answer in answers:
                gen_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
                num_gen_tokens = len(gen_tokens)
                batch_token_counts.append(num_gen_tokens)
                total_tokens_in_batch += num_gen_tokens

            # Generation is parallel, so time per token is per batch.
            batch_time_per_token = gen_time / total_tokens_in_batch if total_tokens_in_batch > 0 else 0.0

            for i, (answer, q, q_text, opts, g_label) in enumerate(zip(
                answers, batch_questions, question_texts, options_list, gold_labels
            )):
                q_idx = batch_start + i
                print(f"A{q_idx+1}: {answer[:100]}...")

                num_gen_tokens = batch_token_counts[i]

                if is_string_match_eval:
                    refs = q.get('answers', [])
                    task = q.get('task', '')
                    score = score_string_match(answer, refs, task)
                    is_correct = score == 1.0
                    print(f"  Score: {score:.2f} | Task: {task} | Refs: {refs}")

                    question_result = {
                        'question_id': q.get('question_unique_id', f'q_{q_idx}'),
                        'question': q_text,
                        'task': task,
                        'answers': refs,
                        'model_answer_text': answer,
                        'score': score,
                        'is_correct': is_correct,
                        'generation_time': gen_time / actual_batch_size,
                        'num_generated_tokens': num_gen_tokens,
                        'time_per_token': batch_time_per_token,
                    }
                else:
                    model_choice = None
                    is_correct = None
                    if opts:
                        model_choice = parse_model_choice(answer, max_options=len(opts))
                        if model_choice is not None:
                            print(f"  Parsed choice: {chr(64+model_choice)} ({model_choice})")
                            if g_label is not None:
                                is_correct = (model_choice == g_label)
                                print(f"  Correct: {is_correct}")
                        else:
                            print(f"  Warning: Could not parse model choice from answer")

                    question_result = {
                        'question_id': q.get('question_unique_id', f'q_{q_idx}'),
                        'question': q_text,
                        'options': opts,
                        'gold_label': g_label,
                        'model_answer_text': answer,
                        'model_choice': model_choice,
                        'is_correct': is_correct,
                        'generation_time': gen_time / actual_batch_size,
                        'num_generated_tokens': num_gen_tokens,
                        'time_per_token': batch_time_per_token,
                    }
                results_per_question.append(question_result)

            print(f"  Batch generation: {gen_time:.2f}s, {total_tokens_in_batch} tokens, {batch_time_per_token:.3f}s/token")

    def evaluate_compaction_on_article(
        self,
        article_data: Dict,
        compaction_method: FullCacheCompactionAlgorithm,
        target_size: float,
        query_config: Optional[QueryConfig] = None,
        verbose_logging: bool = False,
        compute_perplexity: bool = False,
        article_idx: int = 0,
        max_new_tokens: int = 2048,
        n_questions_per_article: Optional[int] = None,
        batch_size: Optional[int] = None,
        is_string_match_eval: bool = False,
    ) -> Dict:
        """Compact one article's cache with ``compaction_method`` and answer its questions.

        Parameters
        ----------
        target_size : float
            Compacted article length, a fraction of the article if in (0, 1). The
            system prompt and chat template tokens are kept unchanged.
        compute_perplexity : bool
            Also compute the perplexity of the full-context answers under the compacted cache.
        n_questions_per_article : int, optional
            Use a fixed random subset of this many questions.
        is_string_match_eval : bool
            Score by string match (lca) instead of multiple-choice parsing.
        """
        from .utils import format_context

        # Text-based methods (original, summarize) return context text instead of a cache.
        use_text_based_generation = not compaction_method.returns_cache()

        print(f"\n{'='*60}")
        print(f"Evaluating on article: {article_data['title']}")
        print(f"Method: {compaction_method.name()}")
        print(f"{'='*60}")

        query_generation_time = 0.0
        effective_memory_stats = None
        context_for_generation = None  # For text-based methods

        if self.model is None:
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.model_name,
                self.device,
                self.dtype,
                self.max_model_len,
            )

        if use_text_based_generation:
            formatted_context = format_context(self.tokenizer, article_data['article'], model_name=self.model_name)
            inputs = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
            seq_len = inputs['input_ids'].shape[1]
            print(f"Formatted context: {seq_len} tokens")

            article_indices = compute_article_indices(
                self.tokenizer, formatted_context, article_data['article']
            )

            start_time = time.time()
            context_for_generation, compaction_stats = compaction_method.compact_kv_cache(
                past_key_values=None,
                target_size=seq_len,
                indices=article_indices,
                query_config=None,
                model=self.model,
                tokenizer=self.tokenizer,
                formatted_context=formatted_context,
                vllm_model=self.vllm_model,
            )
            compaction_time = time.time() - start_time
            extraction_time = 0.0

            context_tokens = self.tokenizer.encode(context_for_generation, add_special_tokens=False)
            effective_seq_len = len(context_tokens)
            print(f"Context for generation: {effective_seq_len} tokens")

            past_key_values = None
            compacted_cache = None  # Built later if needed for perplexity

        else:
            print(f"Extracting KV cache from formatted article...")
            start_time = time.time()
            seq_len, past_key_values, article_indices, formatted_context, _ = extract_full_kv_cache(
                self.model,
                self.tokenizer,
                article_data['article'],
                self.device,
                model_name=self.model_name,
            )
            extraction_time = time.time() - start_time

            print(f"Formatted context tokenized to {seq_len} tokens")
            print(f"Article portion: tokens {article_indices.start}-{article_indices.stop} ({len(article_indices)} tokens)")
            print(f"Tokens to keep unchanged: {seq_len - len(article_indices)} tokens")
            print(f"KV cache extraction took {extraction_time:.2f}s")

        article_len = len(article_indices)
        non_article_tokens = seq_len - article_len

        # Sliding-window layers (e.g. Gemma 3) are not compacted.
        sliding_layer_indices = set()
        sliding_window = None
        if self.model is not None:
            config = getattr(self.model, 'config', None)
            if config is not None:
                layer_types = getattr(config, 'layer_types', None)
                sliding_window = getattr(config, 'sliding_window', None)
                if layer_types is not None:
                    for layer_idx, layer_type in enumerate(layer_types):
                        if layer_type == "sliding_attention":
                            sliding_layer_indices.add(layer_idx)
                if sliding_layer_indices:
                    print(f"Detected {len(sliding_layer_indices)} sliding window layers")

        original_cache_size = None
        compacted_cache_size = None

        if not use_text_based_generation:
            if article_len > 0:
                if 0 < target_size < 1:
                    requested_article_size = max(1, int(article_len * target_size))
                    source_desc = f"{target_size:.1%} of {article_len}"
                else:
                    requested_article_size = max(0, int(target_size))
                    source_desc = f"requested {target_size}"
                article_target_size = min(article_len, requested_article_size)
                if article_target_size != requested_article_size:
                    print(f"Target article size: requested {requested_article_size} tokens but article has {article_len}; using {article_target_size}")
                else:
                    print(f"Target article size: {source_desc} = {article_target_size} tokens")
            else:
                article_target_size = 0
                print("Article portion is empty; target article size set to 0")

            actual_target_size = article_target_size + non_article_tokens
            print(f"Total target size (article {article_target_size} + non-article {non_article_tokens}) = {actual_target_size} tokens")
            print(f"Compacting KV cache to size {actual_target_size}...")
            print(f"Only compacting article portion: tokens {article_indices.start}-{article_indices.stop}")

            start_time = time.time()

            methods_str = ", ".join([f"{mc.method}({mc.fraction:.1%})" for mc in query_config.method_configs])
            print(f"Using QueryConfig: methods=[{methods_str}], "
                  f"max_query_vectors_per_kv_head={query_config.max_query_vectors_per_kv_head}")

            compacted_cache, compaction_stats = compaction_method.compact_kv_cache(
                past_key_values=past_key_values,
                target_size=actual_target_size,
                indices=article_indices,
                query_config=query_config,
                model=self.model,
                tokenizer=self.tokenizer,
                formatted_context=formatted_context,
                verbose_logging=verbose_logging,
                vllm_model=self.vllm_model,
                sliding_layer_indices=sliding_layer_indices,
            )
            total_compaction_time = time.time() - start_time

            # Query generation time is reported separately.
            if 'query_generation' in compaction_stats:
                qstats = compaction_stats['query_generation']
                if 'query_generation_time' in qstats:
                    query_generation_time = qstats['query_generation_time']
            compaction_time = total_compaction_time - query_generation_time

            # Replace the size-0 placeholders of sliding layers with the original cache.
            if sliding_layer_indices and sliding_window is not None:
                compacted_cache_list = list(compacted_cache)
                for layer_idx in sliding_layer_indices:
                    keys = past_key_values[layer_idx][0].clone()
                    values = past_key_values[layer_idx][1].clone()
                    beta = torch.zeros(keys.shape[0], keys.shape[1], keys.shape[2],
                                     device=keys.device, dtype=keys.dtype)
                    compacted_cache_list[layer_idx] = (keys, beta, values)
                compacted_cache = tuple(compacted_cache_list)
                print(f"Injected sliding layer data from original cache ({len(sliding_layer_indices)} layers)")

            memory_stats = compute_cache_memory_stats(past_key_values, compacted_cache)
            original_cache_size = memory_stats['original_cache_size']
            compacted_cache_size = memory_stats['compacted_cache_size']

            reported_tensor_len = compaction_stats.get('tensor_compacted_seq_len')
            # Average over global layers; nonuniform caches differ per layer.
            if reported_tensor_len is None:
                total_tensor_len = 0
                num_global_layers = 0
                for i in range(len(compacted_cache)):
                    if i not in sliding_layer_indices:
                        total_tensor_len += compacted_cache[i][0].shape[2]
                        num_global_layers += 1
                tensor_compacted_seq_len = total_tensor_len / num_global_layers if num_global_layers > 0 else actual_target_size
                compaction_stats['tensor_compacted_seq_len'] = tensor_compacted_seq_len
            else:
                tensor_compacted_seq_len = reported_tensor_len

            effective_len = compaction_stats.get('effective_compacted_seq_len')
            if effective_len is None:
                effective_len = tensor_compacted_seq_len
                compaction_stats['effective_compacted_seq_len'] = effective_len
            effective_compacted_seq_len = effective_len
            if effective_compacted_seq_len > 0:
                compaction_stats['compaction_ratio'] = seq_len / effective_compacted_seq_len

            effective_ratio = effective_compacted_seq_len / tensor_compacted_seq_len if tensor_compacted_seq_len > 0 else 1.0
            effective_ratio = min(1.0, effective_ratio) if effective_ratio > 0 else 0.0
            effective_compacted_cache_size = compacted_cache_size * effective_ratio
            effective_memory_reduction_pct = 100 * (1 - effective_compacted_cache_size / original_cache_size)
            effective_memory_stats = {
                'effective_compacted_cache_size': effective_compacted_cache_size,
                'effective_memory_reduction_pct': effective_memory_reduction_pct,
            }

            print(f"Total time: {total_compaction_time:.2f}s "
                  f"(query gen: {query_generation_time:.2f}s, compaction: {compaction_time:.2f}s)")

            effective_seq_len = effective_compacted_seq_len
        else:
            tensor_compacted_seq_len = effective_seq_len
            effective_compacted_seq_len = effective_seq_len

        print_query_generation_stats(compaction_stats)

        all_questions = article_data['questions']

        if n_questions_per_article is not None:
            import random
            rng = random.Random(67)
            questions = all_questions.copy()
            rng.shuffle(questions)
            questions = questions[:n_questions_per_article]
            print(f"Article has {len(all_questions)} questions (using random {len(questions)} questions)")
        else:
            questions = all_questions
            print(f"Article has {len(questions)} questions")

        context_for_ref_answers = context_for_generation if use_text_based_generation else formatted_context

        reference_answers = None
        if compute_perplexity:
            from .utils import get_or_generate_reference_answers, check_reference_answers_cached

            is_cached = check_reference_answers_cached(
                article_id=article_data['article_id'],
                questions=questions,
                model_name=self.model_name,
            )

            # Offload the HF model only if vLLM must generate and shares its GPU.
            needs_offload = (not is_cached and self.vllm_model is not None
                             and self.model is not None
                             and not getattr(self.vllm_model, "dedicated_gpu", False))
            if needs_offload:
                print("Offloading HuggingFace model to CPU for vLLM generation...")
                offload_model_to_cpu(self.model)

            reference_answers = get_or_generate_reference_answers(
                article_id=article_data['article_id'],
                model=self.model,
                tokenizer=self.tokenizer,
                formatted_context=context_for_ref_answers,
                questions=questions,
                model_name=self.model_name,
                max_new_tokens=max_new_tokens,
                device=self.device,
                vllm_model=self.vllm_model,
            )

            if needs_offload:
                print("Reloading HuggingFace model to GPU...")
                reload_model_to_gpu(self.model, self.device)

            total_tokens = sum(len(tids) for _, tids, _ in reference_answers)
            print(f"Reference answers: {len(reference_answers)} questions, {total_tokens} total tokens")

        # Decode runs with only the compacted cache resident.
        if past_key_values is not None:
            del past_key_values
            past_key_values = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_per_question = []
        num_test_questions = len(questions)

        # Measured decode-time memory (cache-based path only).
        measured_cache_bytes = None
        decode_peak_bytes = None

        perplexities_per_question = []
        avg_perplexity = None
        avg_log_perplexity = None

        if use_text_based_generation:
            print(f"\n{'='*60}")
            print(f"Generating answers using vLLM with {compaction_method.name()} context...")
            print(f"{'='*60}")

            if self.vllm_model is None:
                raise ValueError(
                    f"vLLM model required for text-based method '{compaction_method.name()}'. "
                    "Please initialize with use_vllm=True."
                )

            vllm_shares_gpu = not getattr(self.vllm_model, "dedicated_gpu", False)
            if self.model is not None and vllm_shares_gpu:
                offload_model_to_cpu(self.model)
                torch.cuda.empty_cache()

            self.vllm_model.wake_up()

            try:
                effective_batch_size = batch_size if batch_size is not None and batch_size > 0 else 1

                for batch_start in range(0, len(questions), effective_batch_size):
                    batch_end = min(batch_start + effective_batch_size, len(questions))
                    batch_questions = questions[batch_start:batch_end]

                    prompts = []
                    for q in batch_questions:
                        question_text = q['question']
                        if is_string_match_eval:
                            question_formatted = format_question(
                                self.tokenizer, question_text, options=None,
                                model_name=self.model_name, enable_thinking=False,
                                answer_prefix=q.get('answer_prefix', ''),
                            )
                        else:
                            options = q.get('options', None)
                            question_formatted = format_question(self.tokenizer, question_text, options, self.model_name)
                        full_prompt = context_for_generation + question_formatted
                        prompts.append(full_prompt)

                    if is_string_match_eval:
                        batch_max_new_tokens = int(batch_questions[0].get('max_new_tokens', max_new_tokens))
                    else:
                        batch_max_new_tokens = max_new_tokens
                    gen_start = time.time()
                    gen_params = get_generation_params(self.model)
                    answers = generate_with_vllm_batch(
                        vllm_model=self.vllm_model,
                        full_prompts=prompts,
                        max_new_tokens=batch_max_new_tokens,
                        temperature=gen_params['temperature'],
                        top_k=gen_params['top_k'],
                        top_p=gen_params['top_p'],
                    )
                    gen_time = time.time() - gen_start

                    for i, (q, answer) in enumerate(zip(batch_questions, answers)):
                        gen_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
                        num_gen_tokens = len(gen_tokens)
                        per_question_time = gen_time / len(batch_questions)
                        time_per_token = per_question_time / num_gen_tokens if num_gen_tokens > 0 else 0.0

                        if is_string_match_eval:
                            refs = q.get('answers', [])
                            task = q.get('task', '')
                            score = score_string_match(answer, refs, task)
                            is_correct = score == 1.0
                            print(f"Q{batch_start + i + 1}: {q['question'][:80]}...")
                            print(f"  A: {answer[:100]}... | Score: {score:.2f} | Task: {task}")
                            results_per_question.append({
                                'question_id': q.get('question_unique_id', f'q_{batch_start + i}'),
                                'question': q['question'],
                                'task': task,
                                'answers': refs,
                                'model_answer_text': answer,
                                'score': score,
                                'is_correct': is_correct,
                                'generation_time': per_question_time,
                                'num_generated_tokens': num_gen_tokens,
                                'time_per_token': time_per_token,
                            })
                        else:
                            correct_answer = q.get('gold_label', None)
                            model_choice = parse_model_choice(answer)
                            is_correct = model_choice == correct_answer if model_choice and correct_answer else False
                            print(f"Q{batch_start + i + 1}: {q['question'][:50]}...")
                            print(f"  A: {answer[:100]}... | Choice: {model_choice} | Correct: {correct_answer} | {'✓' if is_correct else '✗'}")
                            results_per_question.append({
                                'question_id': q.get('question_unique_id', f'q_{batch_start + i}'),
                                'question': q['question'],
                                'options': q.get('options', []),
                                'gold_label': correct_answer,
                                'model_answer_text': answer,
                                'model_choice': model_choice,
                                'is_correct': is_correct,
                                'generation_time': per_question_time,
                                'num_generated_tokens': num_gen_tokens,
                                'time_per_token': time_per_token,
                            })
            finally:
                self.vllm_model.sleep()

                if self.model is not None and vllm_shares_gpu:
                    reload_model_to_gpu(self.model, self.device)
        else:
            print(f"\n{'='*60}")
            print(f"Generating answers using COMPACTED cache...")

            if batch_size == 0:
                batch_size = None
            if batch_size is not None:
                print(f"Using batched generation with batch_size={batch_size}")
            print(f"{'='*60}")

            # Keep the cache on CPU (moved to GPU per batch), dense and contiguous,
            # and check it is uniform: one prefix length over global layers and no
            # -inf padded rows.
            cache_for_generation = []
            global_lens = set()
            padded_rows = 0
            measured_cache_bytes = 0
            for layer_idx, (C1, beta, C2) in enumerate(compacted_cache):
                C1 = C1.detach().contiguous().cpu()
                beta = beta.detach().contiguous().cpu()
                C2 = C2.detach().contiguous().cpu()
                if layer_idx not in sliding_layer_indices:
                    global_lens.add(C1.shape[-2])
                    padded_rows += int(torch.isinf(beta).sum().item())
                for t in (C1, beta, C2):
                    measured_cache_bytes += t.element_size() * t.numel()
                cache_for_generation.append((C1, beta, C2))
            if len(global_lens) > 1:
                print(f"WARNING: jagged compacted cache; global-layer prefix "
                      f"lengths {sorted(global_lens)}")
            if padded_rows > 0:
                print(f"WARNING: compacted cache holds {padded_rows} -inf "
                      f"padded rows (not dense)")
            print(f"Compacted cache for decode: {measured_cache_bytes / 1e6:.1f} MB "
                  f"dense, prefix lengths {sorted(global_lens)}")

            effective_batch_size = batch_size if batch_size is not None and batch_size > 0 else 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            self._evaluate_questions_batched(
                questions=questions,
                cache_for_generation=cache_for_generation,
                seq_len=seq_len,
                max_new_tokens=max_new_tokens,
                batch_size=effective_batch_size,
                results_per_question=results_per_question,
                is_string_match_eval=is_string_match_eval,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                decode_peak_bytes = int(torch.cuda.max_memory_allocated())
                print(f"Decode peak GPU memory: {decode_peak_bytes / 1e9:.2f} GB "
                      f"(model weights + compacted cache + activations)")

        # Text-based methods: prefill the generation context to score perplexity.
        if use_text_based_generation and compute_perplexity:
            print(f"\nExtracting KV cache from current context for perplexity computation...")

            inputs = self.tokenizer(context_for_generation, return_tensors="pt", add_special_tokens=False).to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    use_cache=True,
                    return_dict=True,
                )
            context_past_key_values = outputs.past_key_values

            # Store as (C1, beta=0, C2) for all layers.
            compacted_layers = []
            num_layers = len(context_past_key_values)
            for layer_idx in range(num_layers):
                keys = context_past_key_values[layer_idx][0]
                values = context_past_key_values[layer_idx][1]
                beta = torch.zeros(keys.shape[:-1], device=keys.device, dtype=keys.dtype)
                compacted_layers.append((keys, beta, values))
            compacted_cache = tuple(compacted_layers)

            print(f"Built compacted cache: {num_layers} layers, {effective_seq_len} tokens")

        if compute_perplexity:
            if reference_answers is not None and compacted_cache is not None:
                print(f"\n{'='*60}")
                print(f"Computing perplexity of reference answers under compacted cache...")
                print(f"{'='*60}")

                from .utils import compute_perplexity_on_compacted_cache

                for i, (question_id, gen_token_ids, gen_text) in enumerate(reference_answers):
                    q = questions[i]
                    question_text = q['question']
                    options = q.get('options', None)

                    question_formatted = format_question(self.tokenizer, question_text, options, self.model_name)

                    print(f"  Computing perplexity for question {i+1}/{len(reference_answers)}: {question_id}")

                    original_seq_len_for_ppl = effective_seq_len if use_text_based_generation else seq_len
                    perplexity, log_perplexity = compute_perplexity_on_compacted_cache(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        compacted_cache=compacted_cache,
                        generated_token_ids=gen_token_ids,
                        question_prompt=question_formatted,
                        device=self.device,
                        original_seq_len=original_seq_len_for_ppl,
                    )

                    perplexities_per_question.append({
                        'question_id': question_id,
                        'perplexity': perplexity,
                        'log_perplexity': log_perplexity,
                        'num_tokens': len(gen_token_ids)
                    })
                    print(f"    Perplexity: {perplexity:.2f}, Log perplexity: {log_perplexity:.4f} ({len(gen_token_ids)} tokens)")

                avg_perplexity = sum(p['perplexity'] for p in perplexities_per_question) / len(perplexities_per_question)
                avg_log_perplexity = sum(p['log_perplexity'] for p in perplexities_per_question) / len(perplexities_per_question)
                print(f"\n  Average perplexity: {avg_perplexity:.2f}")
                print(f"  Average log perplexity: {avg_log_perplexity:.4f}")
            else:
                print(f"Warning: No reference answers available, skipping perplexity computation")

        total_questions = len(results_per_question)
        total_gen_time = sum(r.get('generation_time', 0.0) for r in results_per_question)
        total_gen_tokens = sum(r.get('num_generated_tokens', 0) for r in results_per_question)
        avg_gen_time_per_question = total_gen_time / total_questions if total_questions > 0 else 0.0
        avg_time_per_token = total_gen_time / total_gen_tokens if total_gen_tokens > 0 else 0.0
        correct_answers = sum(1 for r in results_per_question if r.get('is_correct', False))

        if is_string_match_eval:
            # Accuracy is the mean per-question score, also reported per task.
            avg_score = sum(r.get('score', 0.0) for r in results_per_question) / total_questions if total_questions > 0 else 0.0

            from collections import defaultdict
            task_scores = defaultdict(list)
            for r in results_per_question:
                task_scores[r.get('task', 'unknown')].append(r.get('score', 0.0))
            per_task_score = {task: sum(scores) / len(scores) * 100 for task, scores in sorted(task_scores.items())}

            qa_results = {
                'num_questions': len(questions),
                'results_per_question': results_per_question,
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'parseable_answers': total_questions,  # every string-match answer is scored
                'accuracy': avg_score,
                'parse_rate': 1.0,
                'per_task_score': per_task_score,
                'is_string_match_eval': True,
                'total_generation_time': total_gen_time,
                'total_generated_tokens': total_gen_tokens,
                'avg_generation_time_per_question': avg_gen_time_per_question,
                'avg_time_per_token': avg_time_per_token,
            }

            print(f"\n{'='*60}")
            print(f"String-match Results Summary:")
            print(f"  Total questions: {total_questions}")
            print(f"  Average score: {avg_score * 100:.1f}%")
            print(f"  Per-task scores:")
            for task, score in per_task_score.items():
                count = len(task_scores[task])
                print(f"    {task}: {score:.1f}% ({count} questions)")
            print(f"  Total generation time: {total_gen_time:.2f}s")
            if avg_time_per_token > 0:
                print(f"  Avg time per token: {avg_time_per_token:.3f}s/token ({1.0/avg_time_per_token:.1f} tokens/s)")
            print(f"{'='*60}")
        else:
            parseable_answers = sum(1 for r in results_per_question if r.get('model_choice') is not None)

            qa_results = {
                'num_questions': len(questions),
                'results_per_question': results_per_question,
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'parseable_answers': parseable_answers,
                'accuracy': correct_answers / total_questions if total_questions > 0 else 0.0,
                'parse_rate': parseable_answers / total_questions if total_questions > 0 else 0.0,
                'total_generation_time': total_gen_time,
                'total_generated_tokens': total_gen_tokens,
                'avg_generation_time_per_question': avg_gen_time_per_question,
                'avg_time_per_token': avg_time_per_token,
            }

            if use_text_based_generation:
                qa_results['note'] = f'Generated answers with {compaction_method.name()} context for {num_test_questions} questions'
            else:
                qa_results['note'] = f'Generated answers with compacted cache for {num_test_questions} questions'

            print(f"\n{'='*60}")
            print(f"QA Results Summary:")
            print(f"  Total questions: {total_questions}")
            print(f"  Parseable answers: {parseable_answers} ({qa_results['parse_rate']:.1%})")
            print(f"  Correct answers: {correct_answers} ({qa_results['accuracy']:.1%})")
            print(f"  Total generation time: {total_gen_time:.2f}s")
            print(f"  Total generated tokens: {total_gen_tokens}")
            print(f"  Avg generation time/question: {avg_gen_time_per_question:.2f}s")
            if avg_time_per_token > 0:
                print(f"  Avg time per token: {avg_time_per_token:.3f}s/token ({1.0/avg_time_per_token:.1f} tokens/s)")
            print(f"{'='*60}")

        # Tensor size of the article portion, including padding.
        tensor_article_tokens = compaction_stats.get('tensor_article_tokens')
        if isinstance(tensor_article_tokens, list):
            tensor_article_tokens = (
                sum(float(x) for x in tensor_article_tokens) / len(tensor_article_tokens)
                if tensor_article_tokens else 0.0
            )
        if tensor_article_tokens is None:
            tensor_article_tokens = tensor_compacted_seq_len - non_article_tokens
        tensor_article_tokens = float(max(0.0, tensor_article_tokens))

        # Mean tokens per head, excluding -inf padding.
        effective_article_tokens = compaction_stats.get('effective_article_tokens')
        if isinstance(effective_article_tokens, list):
            effective_article_tokens = (
                sum(float(x) for x in effective_article_tokens) / len(effective_article_tokens)
                if effective_article_tokens else 0.0
            )
        if effective_article_tokens is None:
            effective_article_tokens = tensor_article_tokens
        effective_article_tokens = float(max(0.0, effective_article_tokens))

        safe_effective_article = max(effective_article_tokens, 1e-8)
        safe_tensor_article = max(tensor_article_tokens, 1e-8)

        article_compaction_ratio = (
            len(article_indices) / safe_effective_article if len(article_indices) > 0 else 0.0
        )
        article_tensor_compaction_ratio = (
            len(article_indices) / safe_tensor_article if len(article_indices) > 0 else 0.0
        )

        print(f"\nArticle compaction statistics:")
        print(f"  Original article tokens:  {len(article_indices)}")
        print(f"  Effective compacted article tokens: {effective_article_tokens:.2f}")
        if effective_article_tokens != tensor_article_tokens:
            print(f"  Tensor compacted article tokens:   {tensor_article_tokens:.0f}")
        print(f"  Article compaction ratio: {article_compaction_ratio:.2f}x")

        if original_cache_size is not None:
            memory_reduction = 100 * (1 - compacted_cache_size/original_cache_size)

            print(f"\nMemory statistics:")
            print(f"  Original cache:  {original_cache_size:,} elements ({seq_len} tokens)")
            print(f"  Compacted cache (tensor): {compacted_cache_size:,} elements ({tensor_compacted_seq_len} tokens)")
            print(f"  Memory reduction: {memory_reduction:.1f}%")
            if effective_memory_stats is not None:
                print(f"  Effective tokens: {effective_compacted_seq_len:.2f}")
                print(f"  Effective memory reduction: {effective_memory_stats['effective_memory_reduction_pct']:.1f}%")
            article_ratio_display = (
                article_len / safe_effective_article if article_len > 0 else float('nan')
            )
            print(f"  Article compaction ratio: {article_ratio_display:.2f}x")

        results = {
            'article_id': article_data['article_id'],
            'article_title': article_data['title'],
            'article_idx': article_idx,
            'method': compaction_method.name(),
            'target_size_param': target_size,
            'original_article_tokens': article_len,
            'effective_article_tokens': effective_article_tokens,
            'tensor_article_tokens': tensor_article_tokens,
            'non_article_tokens': non_article_tokens,
            'article_compaction_ratio': article_compaction_ratio,
            'article_tensor_compaction_ratio': article_tensor_compaction_ratio,
            'extraction_time': extraction_time,
            'compaction_time': compaction_time,
            'query_generation_time': query_generation_time,
            'compaction_stats': compaction_stats,
            'qa_results': qa_results,
        }

        if avg_perplexity is not None:
            results['avg_perplexity'] = avg_perplexity
            results['avg_log_perplexity'] = avg_log_perplexity
            results['perplexities_per_question'] = perplexities_per_question

        if measured_cache_bytes is not None:
            results['measured_compacted_cache_bytes'] = measured_cache_bytes
        if decode_peak_bytes is not None:
            results['decode_peak_memory_bytes'] = decode_peak_bytes
        if original_cache_size is not None:
            memory_reduction = 100 * (1 - compacted_cache_size/original_cache_size)
            results['original_cache_size'] = original_cache_size
            results['compacted_cache_size'] = compacted_cache_size
            results['memory_reduction_pct'] = memory_reduction
            if effective_memory_stats is not None:
                results['effective_memory_stats'] = effective_memory_stats

        return results

    def _compute_overall_stats(self, all_results: List[Dict]) -> Dict:
        """Aggregate per-article results into per-method statistics."""
        from collections import defaultdict

        results_by_method = defaultdict(list)
        for result in all_results:
            method = result['method']
            results_by_method[method].append(result)

        overall_stats = {}

        for method, method_results in results_by_method.items():
            total_questions = sum(r['qa_results'].get('total_questions', 0) for r in method_results)
            total_correct = sum(r['qa_results'].get('correct_answers', 0) for r in method_results)
            total_parseable = sum(r['qa_results'].get('parseable_answers', 0) for r in method_results)
            is_string_match = any(r['qa_results'].get('is_string_match_eval', False) for r in method_results)

            total_extraction_time = sum(r['extraction_time'] for r in method_results)
            total_compaction_time = sum(r['compaction_time'] for r in method_results)
            total_query_generation_time = sum(r.get('query_generation_time', 0.0) for r in method_results)

            total_generation_time = 0.0
            total_generated_tokens = 0
            for r in method_results:
                for q_result in r['qa_results']['results_per_question']:
                    if 'generation_time' in q_result:
                        total_generation_time += q_result['generation_time']
                    if 'num_generated_tokens' in q_result:
                        total_generated_tokens += q_result['num_generated_tokens']

            avg_time_per_token = total_generation_time / total_generated_tokens if total_generated_tokens > 0 else 0.0
            avg_tokens_per_second = total_generated_tokens / total_generation_time if total_generation_time > 0 else 0.0

            article_compaction_stats = None
            if method != 'original':
                article_original_tokens = []
                article_effective_tokens = []
                article_tensor_tokens = []
                article_compaction_ratios = []
                article_tensor_ratios = []
                non_article_tokens_list = []

                for r in method_results:
                    orig_tok = r.get('original_article_tokens')
                    if orig_tok is not None:
                        article_original_tokens.append(orig_tok)
                    eff_tok = r.get('effective_article_tokens')
                    if eff_tok is not None:
                        article_effective_tokens.append(eff_tok)
                    tensor_tok = r.get('tensor_article_tokens')
                    if tensor_tok is not None:
                        article_tensor_tokens.append(tensor_tok)
                    if 'article_compaction_ratio' in r:
                        article_compaction_ratios.append(r['article_compaction_ratio'])
                    if 'article_tensor_compaction_ratio' in r:
                        article_tensor_ratios.append(r['article_tensor_compaction_ratio'])
                    non_art = r.get('non_article_tokens')
                    if non_art is not None:
                        non_article_tokens_list.append(non_art)

                if article_original_tokens:
                    article_compaction_stats = {
                        'avg_original_article_tokens': sum(article_original_tokens) / len(article_original_tokens),
                        'avg_effective_article_tokens': sum(article_effective_tokens) / len(article_effective_tokens) if article_effective_tokens else 0.0,
                        'avg_article_compaction_ratio': sum(article_compaction_ratios) / len(article_compaction_ratios) if article_compaction_ratios else 0.0,
                    }
                    if article_tensor_tokens:
                        article_compaction_stats['avg_tensor_article_tokens'] = sum(article_tensor_tokens) / len(article_tensor_tokens)
                    if article_tensor_ratios:
                        article_compaction_stats['avg_tensor_article_compaction_ratio'] = sum(article_tensor_ratios) / len(article_tensor_ratios)
                    if non_article_tokens_list:
                        article_compaction_stats['avg_non_article_tokens'] = sum(non_article_tokens_list) / len(non_article_tokens_list)

            avg_target_size_param = sum(r.get('target_size_param', 0.0) for r in method_results) / len(method_results)

            memory_stats = None
            results_with_memory = [r for r in method_results if 'original_cache_size' in r]
            if results_with_memory:
                avg_original_cache_size = sum(r['original_cache_size'] for r in results_with_memory) / len(results_with_memory)
                avg_compacted_cache_size = sum(r['compacted_cache_size'] for r in results_with_memory) / len(results_with_memory)
                avg_memory_reduction_pct = sum(r['memory_reduction_pct'] for r in results_with_memory) / len(results_with_memory)
                memory_stats = {
                    'avg_original_cache_size': avg_original_cache_size,
                    'avg_compacted_cache_size': avg_compacted_cache_size,
                    'avg_memory_reduction_pct': avg_memory_reduction_pct,
                }
                results_with_effective = [r for r in results_with_memory if 'effective_memory_stats' in r]
                if results_with_effective:
                    avg_effective_size = sum(r['effective_memory_stats']['effective_compacted_cache_size'] for r in results_with_effective) / len(results_with_effective)
                    avg_effective_reduction = sum(r['effective_memory_stats']['effective_memory_reduction_pct'] for r in results_with_effective) / len(results_with_effective)
                    memory_stats['avg_effective_compacted_cache_size'] = avg_effective_size
                    memory_stats['avg_effective_memory_reduction_pct'] = avg_effective_reduction

            results_with_perplexity = [r for r in method_results if 'avg_perplexity' in r]
            avg_perplexity_across_articles = None
            avg_log_perplexity_across_articles = None
            if results_with_perplexity:
                avg_perplexity_across_articles = (
                    sum(r['avg_perplexity'] for r in results_with_perplexity) / len(results_with_perplexity)
                )
                avg_log_perplexity_across_articles = (
                    sum(r['avg_log_perplexity'] for r in results_with_perplexity) / len(results_with_perplexity)
                )

            if is_string_match:
                # Mean per-question score over all articles' questions.
                total_score_sum = sum(
                    sum(q.get('score', 0.0) for q in r['qa_results'].get('results_per_question', []))
                    for r in method_results
                )
                overall_accuracy = total_score_sum / total_questions if total_questions > 0 else 0.0
            else:
                overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
            overall_stats[method] = {
                'num_articles': len(method_results),
                'total_questions': total_questions,
                'total_correct': total_correct,
                'total_parseable': total_parseable,
                'overall_accuracy': overall_accuracy,
                'overall_parse_rate': total_parseable / total_questions if total_questions > 0 else 0.0,
                'avg_target_size_param': avg_target_size_param,
                'total_extraction_time': total_extraction_time,
                'total_compaction_time': total_compaction_time,
                'total_query_generation_time': total_query_generation_time,
                'total_generation_time': total_generation_time,
                'avg_extraction_time_per_article': total_extraction_time / len(method_results),
                'avg_compaction_time_per_article': total_compaction_time / len(method_results),
                'avg_query_generation_time_per_article': total_query_generation_time / len(method_results),
                'avg_generation_time_per_question': total_generation_time / total_questions if total_questions > 0 else 0.0,
                'avg_time_per_token': avg_time_per_token,
                'avg_tokens_per_second': avg_tokens_per_second,
                'total_generated_tokens': total_generated_tokens,
            }

            if memory_stats:
                overall_stats[method]['memory_stats'] = memory_stats

            if article_compaction_stats:
                overall_stats[method]['article_compaction_stats'] = article_compaction_stats

            if avg_perplexity_across_articles is not None:
                overall_stats[method]['overall_avg_perplexity'] = avg_perplexity_across_articles
                overall_stats[method]['overall_avg_log_perplexity'] = avg_log_perplexity_across_articles
                overall_stats[method]['num_articles_with_perplexity'] = len(results_with_perplexity)

            overall_stats[method]['per_article'] = []
            for r in method_results:
                overall_stats[method]['per_article'].append({
                    'article_idx': r['article_idx'],
                    'article_title': r['article_title'],
                    'num_questions': r['qa_results'].get('total_questions', r['qa_results']['num_questions']),
                    'correct': r['qa_results'].get('correct_answers', 0),
                    'accuracy': r['qa_results'].get('accuracy', 0.0),
                    'article_compaction_ratio': r.get('article_compaction_ratio', 0.0),
                })

        return overall_stats

    def _log_overall_results(self, overall_stats: Dict, methods: List[str]):
        """Print the per-method summary."""
        print(f"\n{'='*80}")
        print(f"OVERALL RESULTS SUMMARY")
        print(f"{'='*80}")

        for method in overall_stats.keys():
            stats = overall_stats[method]
            print(f"\nMethod: {method.upper()}")
            print(f"{'-'*80}")
            print(f"  Articles evaluated: {stats['num_articles']}")
            print(f"  Total questions: {stats['total_questions']}")
            print(f"  Overall accuracy: {stats['overall_accuracy']:.2%} ({stats['total_correct']}/{stats['total_questions']})")
            print(f"  Overall parse rate: {stats['overall_parse_rate']:.2%} ({stats['total_parseable']}/{stats['total_questions']})")

            if 'overall_avg_perplexity' in stats:
                print(f"\n  Perplexity:")
                print(f"    Overall avg perplexity: {stats['overall_avg_perplexity']:.2f}")
                print(f"    Overall avg log perplexity: {stats['overall_avg_log_perplexity']:.4f}")
                print(f"    Articles with perplexity: {stats['num_articles_with_perplexity']}")

            if 'article_compaction_stats' in stats:
                astats = stats['article_compaction_stats']
                print(f"\n  Article compaction statistics:")
                print(f"    Avg original article tokens:   {astats['avg_original_article_tokens']:.0f}")
                print(f"    Avg effective article tokens:  {astats['avg_effective_article_tokens']:.0f}")
                if 'avg_tensor_article_tokens' in astats and astats['avg_tensor_article_tokens'] != astats['avg_effective_article_tokens']:
                    print(f"    Avg tensor article tokens:     {astats['avg_tensor_article_tokens']:.0f}")
                print(f"    Avg article compaction ratio:  {astats['avg_article_compaction_ratio']:.2f}x")
                if 'avg_tensor_article_compaction_ratio' in astats:
                    print(f"    Avg tensor compaction ratio:   {astats['avg_tensor_article_compaction_ratio']:.2f}x")
                if 'avg_non_article_tokens' in astats:
                    print(f"    Avg non-article tokens:        {astats['avg_non_article_tokens']:.0f}")

            if 'memory_stats' in stats:
                mstats = stats['memory_stats']
                print(f"\n  Memory Statistics:")
                print(f"    Original cache:  {mstats['avg_original_cache_size']:,.0f} elements")
                print(f"    Compacted cache: {mstats['avg_compacted_cache_size']:,.0f} elements")
                print(f"    Target size param:  {stats['avg_target_size_param']:.3f}")
                print(f"    Memory reduction: {mstats['avg_memory_reduction_pct']:.1f}%")
                if 'avg_effective_compacted_cache_size' in mstats:
                    print(f"    Effective compacted cache: {mstats['avg_effective_compacted_cache_size']:,.0f} elements")
                if 'avg_effective_memory_reduction_pct' in mstats:
                    print(f"    Effective memory reduction: {mstats['avg_effective_memory_reduction_pct']:.1f}%")
            print(f"\n  Timing:")
            print(f"    Total extraction time: {stats['total_extraction_time']:.2f}s")
            print(f"    Total query generation time: {stats['total_query_generation_time']:.2f}s")
            print(f"    Total compaction time: {stats['total_compaction_time']:.2f}s")
            print(f"    Avg extraction time/article: {stats['avg_extraction_time_per_article']:.2f}s")
            print(f"    Avg query generation time/article: {stats['avg_query_generation_time_per_article']:.2f}s")
            print(f"    Avg compaction time/article: {stats['avg_compaction_time_per_article']:.2f}s")
            print(f"\n  Generation:")
            print(f"    Total generation time: {stats['total_generation_time']:.2f}s")
            print(f"    Total generated tokens: {stats['total_generated_tokens']}")
            print(f"    Avg generation time/question: {stats['avg_generation_time_per_question']:.2f}s")
            if stats['avg_time_per_token'] > 0:
                print(f"    Avg time per token: {stats['avg_time_per_token']:.3f}s/token")
                print(f"    Avg tokens per second: {stats['avg_tokens_per_second']:.1f} tokens/s")
            else:
                print(f"    Avg time per token: N/A")
                print(f"    Avg tokens per second: N/A")

            print(f"\n  Per-article breakdown:")
            print(f"    {'Idx':<6} {'Title':<40} {'Qs':<4} {'Correct':<8} {'Accuracy':<10} {'Ratio':<10}")
            print(f"    {'-'*81}")
            for article_stats in stats['per_article']:
                title_short = article_stats['article_title'][:37] + '...' if len(article_stats['article_title']) > 40 else article_stats['article_title']
                print(f"    {article_stats['article_idx']:<6} {title_short:<40} {article_stats['num_questions']:<4} "
                      f"{article_stats['correct']:<8} {article_stats['accuracy']:<10.2%} "
                      f"{article_stats['article_compaction_ratio']:<10.2f}")

        print(f"\n{'='*80}\n")

    def run_evaluation(
        self,
        dataset_name: str,
        compaction_methods: List[str],
        target_size: float,
        query_config: QueryConfig,
        n_articles: int = 1,
        start_article: int = 0,
        n_questions_per_article: Optional[int] = None,
        max_new_tokens: int = 2048,
        batch_size: Optional[int] = None,
        verbose_logging: bool = False,
        compute_perplexity: bool = False,
        method_kwargs: Optional[Dict[str, Dict]] = None,
        log_dir: str = 'logs/qa_evaluation',
        experiment_name: Optional[str] = None,
        algorithm_config_file: Optional[str] = None,
        query_config_file: Optional[str] = None,
    ) -> Dict:
        """Evaluate every method on every selected article and save the results as JSON.

        Parameters
        ----------
        target_size : float
            Compacted article length, a fraction of the article if in (0, 1).
        n_articles : int
            -1 for all articles from ``start_article``.
        n_questions_per_article : int, optional
            Use a fixed random subset of this many questions.
        method_kwargs : dict, optional
            {method_name: kwargs}.
        algorithm_config_file, query_config_file : str, optional
            Recorded in the output only.

        Returns
        -------
        dict
            Config, per-article results and ``overall_stats``.
        """
        from compaction.compaction_methods import get_compaction_method

        dataset = load_dataset(dataset_name)

        if n_articles == -1:
            article_indices = list(range(start_article, len(dataset)))
        else:
            end_article = min(start_article + n_articles, len(dataset))
            article_indices = list(range(start_article, end_article))

        print(f"\n{'='*60}")
        print(f"QA EVALUATION")
        print(f"{'='*60}")
        print(f"Methods: {', '.join(compaction_methods)}")
        print(f"Articles: {len(article_indices)}")
        print(f"Target size: {target_size}")
        print(f"Device: {self.device}")
        methods_str = ", ".join([f"{mc.method}({mc.fraction:.1%})" for mc in query_config.method_configs])
        print(f"Query config: methods=[{methods_str}], "
              f"max_query_vectors_per_kv_head={query_config.max_query_vectors_per_kv_head}")
        print(f"{'='*60}\n")

        # vLLM is needed for generating self-study queries, summaries,
        # full-context answers ('original') and reference answers.
        need_vllm = False

        # Only self-study specs that generate need vLLM (same check as
        # SelfStudyQueryGenerator.__init__).
        self_study_config = query_config.get_method_config('self_study')
        if self_study_config is not None:
            ss_cfg = getattr(self_study_config, 'config', None)
            specs = getattr(ss_cfg, 'conversation_specs', None) or []
            if not specs or any(not spec.is_prefill() or not spec.is_direct() for spec in specs):
                need_vllm = True

        if compute_perplexity:
            need_vllm = True

        for method_name in compaction_methods:
            if method_name == 'original':
                need_vllm = True
                break
            kwargs = (method_kwargs or {}).get(method_name, {})
            base_algorithm = kwargs.get('algorithm', method_name)
            if base_algorithm == 'summarize':
                need_vllm = True
                break

        if need_vllm and self.vllm_model is None:
            self.vllm_model = initialize_vllm(self.model_name, max_model_len=self.max_model_len,
                                              seed=self.seed)

        is_string_match_eval = is_string_match_dataset(dataset_name)
        if is_string_match_eval:
            print(f"Dataset '{dataset_name}' uses string-match evaluation")

        all_results = []

        for method_name in compaction_methods:
            kwargs = (method_kwargs or {}).get(method_name, {})
            compaction_method = get_compaction_method(method_name, kwargs)

            for article_idx in article_indices:
                article_data = dataset[article_idx]

                result = self.evaluate_compaction_on_article(
                    article_data=article_data,
                    compaction_method=compaction_method,
                    target_size=target_size,
                    query_config=query_config,
                    verbose_logging=verbose_logging,
                    compute_perplexity=compute_perplexity,
                    article_idx=article_idx,
                    max_new_tokens=max_new_tokens,
                    n_questions_per_article=n_questions_per_article,
                    batch_size=batch_size,
                    is_string_match_eval=is_string_match_eval,
                )

                all_results.append(result)

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            base_filename = f"{experiment_name}_{timestamp}"
        else:
            base_filename = f"qa_evaluation_{timestamp}"

        filepath = log_path / f"{base_filename}.json"
        suffix = 1
        while filepath.exists():
            filepath = log_path / f"{base_filename}_{suffix}.json"
            suffix += 1

        filtered_hyperparameters = {}
        if method_kwargs:
            for method_name in compaction_methods:
                if method_name in method_kwargs:
                    filtered_hyperparameters[method_name] = method_kwargs[method_name]

        query_config_dict = None
        if query_config:
            query_config_dict = asdict(query_config)
            # extraction_fn is a callable and not JSON-serializable.
            for mc in query_config_dict['method_configs']:
                if mc['method'] == 'self_study' and 'config' in mc and 'conversation_specs' in mc['config']:
                    for spec in mc['config']['conversation_specs']:
                        spec.pop('extraction_fn', None)

        output = {
            'timestamp': timestamp,
            'experiment_name': experiment_name,
            'algorithm_config_file': algorithm_config_file,
            'query_config_file': query_config_file,
            'hyperparameters': filtered_hyperparameters,
            'config': {
                'model_name': self.model_name,
                'dataset_name': dataset_name,
                'methods': compaction_methods,
                'seed': self.seed,
                'target_size': target_size,
                'n_articles': len(article_indices),
                'start_article': start_article,
                'article_indices': article_indices,
                'n_questions_per_article': n_questions_per_article,
                'query_config': query_config_dict,
                'compute_perplexity': compute_perplexity,
            },
            'results': all_results
        }

        overall_stats = self._compute_overall_stats(all_results)
        output['overall_stats'] = overall_stats

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        self._log_overall_results(overall_stats, compaction_methods)

        print(f"\n{'='*60}")
        print(f"Evaluation complete! Results saved to:")
        print(f"{filepath}")
        if algorithm_config_file:
            print(f"Algorithm config file: {algorithm_config_file}")
        if query_config_file:
            print(f"Query config file: {query_config_file}")
        print(f"{'='*60}\n")

        return output
