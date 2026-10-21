"""Per-head influence curves: change in log-perplexity as one head's compaction ratio varies."""
import torch
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

from compaction.algorithms.highest_attention_keys import HighestAttentionKeysCompaction
from compaction.algorithms.omp import OMPCompaction
from compaction.query_generation import QueryConfig, QueryGenerator
from evaluation.utils import (
    format_question,
    get_or_generate_reference_answers,
    compute_perplexity_on_compacted_cache,
)
from models.generate import get_sliding_layer_info


@dataclass
class HeadCompactionResult:
    """Result of compacting a single head."""
    C1: torch.Tensor  # (t, head_dim)
    beta: torch.Tensor  # (t,)
    C2: torch.Tensor  # (t, head_dim)
    selected_indices: List[int]
    # OMP only: full greedy selection order, reused when varying the ratio
    cached_selection_order: Optional[List[int]] = None


class HeadInfluenceComputer:
    """Compute per-head influence curves (log-perplexity vs. one head's ratio)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        query_config: QueryConfig,
        algorithm_kwargs: Optional[Dict] = None,
        algorithm_name: str = "highest_attention_keys",
        device: str = "cuda",
        vllm_model: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        algorithm_name : str
            'highest_attention_keys' or 'omp'.
        vllm_model : optional
            Pre-initialized vLLM model for self-study query generation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.query_config = query_config
        self.algorithm_kwargs = algorithm_kwargs or {}
        self.algorithm_name = algorithm_name
        self.device = device
        self.vllm_model = vllm_model

        self.num_layers = model.config.num_hidden_layers
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = getattr(
            model.config, 'head_dim',
            model.config.hidden_size // model.config.num_attention_heads
        )

        # Only global (non-sliding) layers get optimized budgets.
        self.sliding_layer_indices, self.sliding_window = get_sliding_layer_info(model)
        self.global_layer_indices = [
            i for i in range(self.num_layers) if i not in self.sliding_layer_indices
        ]
        self.num_global_layers = len(self.global_layer_indices)

    def _create_algorithm(self) -> Union[HighestAttentionKeysCompaction, OMPCompaction]:
        if self.algorithm_name == 'omp':
            return OMPCompaction(**self.algorithm_kwargs)
        else:
            return HighestAttentionKeysCompaction(**self.algorithm_kwargs)

    def compute_baseline_and_queries(
        self,
        past_key_values: Tuple,
        formatted_context: str,
        article_indices: range,
        baseline_proportions: Dict[str, float],
        target_ratio: float,
        eval_ratios: Optional[Union[List[float], Dict[str, List[float]]]] = None,
    ) -> Tuple[Dict[str, HeadCompactionResult], torch.Tensor, Dict]:
        """
        Generate queries once and compact every global head at its baseline budget.

        Parameters
        ----------
        baseline_proportions : dict
            Per-head fraction of the total budget, keyed "L{layer}H{head}".
        eval_ratios : list or dict of head -> list, optional
            Ratios to be evaluated later. For OMP this caps the cached selection order;
            None means the full article length.

        Returns
        -------
        head_results : dict of head key -> HeadCompactionResult
        queries : torch.Tensor, (num_layers, num_kv_heads, n_queries, head_dim)
        query_stats : dict
        """
        ref_layer_idx = self.global_layer_indices[0] if self.global_layer_indices else 0
        batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape
        article_len = len(article_indices)

        print("Generating queries for baseline...")
        if self.sliding_layer_indices:
            print(f"Note: Skipping {len(self.sliding_layer_indices)} sliding window layers, "
                  f"optimizing {self.num_global_layers} global layers only")
        query_gen_start = time.time()

        generator = QueryGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.query_config,
            device=self.device,
            dtype=past_key_values[0][0].dtype,
            vllm_model=self.vllm_model,
        )

        queries, query_stats, _ = generator.generate_queries(
            formatted_context=formatted_context,
            past_key_values=past_key_values,
            indices=article_indices,
        )

        query_gen_time = time.time() - query_gen_start
        print(f"Query generation took {query_gen_time:.2f}s")
        print(f"Generated {query_stats['final_n_queries_per_kv_head']} queries per KV head")

        print(f"Computing baseline compaction for global attention heads (algorithm={self.algorithm_name})...")
        compact_start = time.time()

        algorithm = self._create_algorithm()
        head_results = {}

        # Proportions sum to 1 over global heads.
        total_budget = int(target_ratio * article_len * self.num_global_layers * self.num_kv_heads)

        for layer_idx in self.global_layer_indices:
            keys_layer = past_key_values[layer_idx][0][0]  # (num_heads, seq_len, head_dim)
            values_layer = past_key_values[layer_idx][1][0]

            for head_idx in range(self.num_kv_heads):
                head_key = f"L{layer_idx}H{head_idx}"
                proportion = baseline_proportions.get(head_key, 1.0 / (self.num_global_layers * self.num_kv_heads))

                head_budget = max(1, int(proportion * total_budget))

                K = keys_layer[head_idx, list(article_indices), :]  # (article_len, head_dim)
                V = values_layer[head_idx, list(article_indices), :]

                queries_head = queries[layer_idx, head_idx, :, :]  # (n_queries, head_dim)

                # OMP: cache the selection order up to the largest ratio to be evaluated.
                cached_selection_order = None
                if self.algorithm_name == 'omp':
                    if eval_ratios is not None:
                        if isinstance(eval_ratios, dict):
                            max_ratio_for_head = max(eval_ratios.get(head_key, [1.0]))
                        else:
                            max_ratio_for_head = max(eval_ratios)
                        omp_max_keys = min(article_len, int(max_ratio_for_head * article_len) + 1)
                    else:
                        omp_max_keys = article_len

                    cached_selection_order = algorithm.get_full_selection_order(
                        K, queries_head, max_keys=omp_max_keys
                    )

                if self.algorithm_name == 'omp' and cached_selection_order is not None:
                    C1, beta, C2, selected_indices = algorithm.compute_compacted_cache(
                        K, V, queries_head, head_budget,
                        cached_selection_order=cached_selection_order
                    )
                else:
                    C1, beta, C2, selected_indices = algorithm.compute_compacted_cache(
                        K, V, queries_head, head_budget
                    )

                head_results[head_key] = HeadCompactionResult(
                    C1=C1, beta=beta, C2=C2, selected_indices=selected_indices,
                    cached_selection_order=cached_selection_order
                )

        compact_time = time.time() - compact_start
        print(f"Baseline compaction took {compact_time:.2f}s for {len(head_results)} heads")

        return head_results, queries, query_stats

    def build_compacted_cache(
        self,
        past_key_values: Tuple,
        head_results: Dict[str, HeadCompactionResult],
        article_indices: range,
        max_compacted_size: Optional[int] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """
        Build a ((C1, beta, C2), ...) cache from per-head results for global layers.

        Heads are padded to max_compacted_size with beta=-inf. Sliding layers keep
        their window unchanged with beta=0.
        """
        ref_layer_idx = self.global_layer_indices[0] if self.global_layer_indices else 0
        batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape
        device = past_key_values[ref_layer_idx][0].device
        dtype = past_key_values[ref_layer_idx][0].dtype

        all_indices = set(range(seq_len))
        article_set = set(article_indices)
        keep_indices = sorted(all_indices - article_set)

        article_start = article_indices.start
        article_end = article_indices.stop
        keep_before = [i for i in keep_indices if i < article_start]
        keep_after = [i for i in keep_indices if i >= article_end]

        if max_compacted_size is None:
            max_compacted_size = max(hr.C1.shape[0] for hr in head_results.values())

        compacted_layers = []

        for layer_idx in range(self.num_layers):
            keys_layer = past_key_values[layer_idx][0][0]  # (num_heads, seq_len, head_dim)
            values_layer = past_key_values[layer_idx][1][0]

            if layer_idx in self.sliding_layer_indices:
                sliding_seq_len = keys_layer.shape[1]
                C1_layer = keys_layer.unsqueeze(0)  # (1, num_heads, sliding_seq_len, head_dim)
                beta_layer = torch.zeros(1, self.num_kv_heads, sliding_seq_len, device=device, dtype=dtype)
                C2_layer = values_layer.unsqueeze(0)  # (1, num_heads, sliding_seq_len, head_dim)
                compacted_layers.append((C1_layer, beta_layer, C2_layer))
                continue

            C1_heads = []
            beta_heads = []
            C2_heads = []

            for head_idx in range(self.num_kv_heads):
                head_key = f"L{layer_idx}H{head_idx}"
                hr = head_results[head_key]

                K_keep_before = keys_layer[head_idx, keep_before, :] if keep_before else keys_layer.new_zeros(0, head_dim)
                V_keep_before = values_layer[head_idx, keep_before, :] if keep_before else values_layer.new_zeros(0, head_dim)
                K_keep_after = keys_layer[head_idx, keep_after, :] if keep_after else keys_layer.new_zeros(0, head_dim)
                V_keep_after = values_layer[head_idx, keep_after, :] if keep_after else values_layer.new_zeros(0, head_dim)

                beta_keep_before = keys_layer.new_zeros(len(keep_before))
                beta_keep_after = keys_layer.new_zeros(len(keep_after))

                C1_compact = hr.C1
                beta_compact = hr.beta
                C2_compact = hr.C2

                if C1_compact.shape[0] < max_compacted_size:
                    pad_len = max_compacted_size - C1_compact.shape[0]
                    C1_compact = torch.cat([C1_compact, C1_compact.new_zeros(pad_len, head_dim)], dim=0)
                    C2_compact = torch.cat([C2_compact, C2_compact.new_zeros(pad_len, head_dim)], dim=0)
                    beta_compact = torch.cat([beta_compact, beta_compact.new_full((pad_len,), float('-inf'))], dim=0)

                C1 = torch.cat([K_keep_before, C1_compact, K_keep_after], dim=0)
                beta = torch.cat([beta_keep_before, beta_compact, beta_keep_after], dim=0)
                C2 = torch.cat([V_keep_before, C2_compact, V_keep_after], dim=0)

                C1_heads.append(C1.unsqueeze(0).unsqueeze(0))  # (1, 1, t, head_dim)
                beta_heads.append(beta.unsqueeze(0).unsqueeze(0))  # (1, 1, t)
                C2_heads.append(C2.unsqueeze(0).unsqueeze(0))  # (1, 1, t, head_dim)

            C1_layer = torch.cat(C1_heads, dim=1)  # (1, num_heads, t, head_dim)
            beta_layer = torch.cat(beta_heads, dim=1)  # (1, num_heads, t)
            C2_layer = torch.cat(C2_heads, dim=1)  # (1, num_heads, t, head_dim)

            compacted_layers.append((C1_layer, beta_layer, C2_layer))

        return tuple(compacted_layers)

    def swap_head_in_cache(
        self,
        compacted_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        layer_idx: int,
        head_idx: int,
        new_hr: HeadCompactionResult,
        keep_before_len: int,
        max_compacted_size: int,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Return a copy of compacted_cache with one head's compacted slice replaced."""
        C1_layer, beta_layer, C2_layer = compacted_cache[layer_idx]
        head_dim = C1_layer.shape[-1]

        C1_new = new_hr.C1
        beta_new = new_hr.beta
        C2_new = new_hr.C2

        if C1_new.shape[0] < max_compacted_size:
            pad_len = max_compacted_size - C1_new.shape[0]
            C1_new = torch.cat([C1_new, C1_new.new_zeros(pad_len, head_dim)], dim=0)
            C2_new = torch.cat([C2_new, C2_new.new_zeros(pad_len, head_dim)], dim=0)
            beta_new = torch.cat([beta_new, beta_new.new_full((pad_len,), float('-inf'))], dim=0)

        C1_layer_new = C1_layer.clone()
        beta_layer_new = beta_layer.clone()
        C2_layer_new = C2_layer.clone()

        compact_start = keep_before_len
        compact_end = keep_before_len + max_compacted_size

        C1_layer_new[0, head_idx, compact_start:compact_end, :] = C1_new
        beta_layer_new[0, head_idx, compact_start:compact_end] = beta_new
        C2_layer_new[0, head_idx, compact_start:compact_end, :] = C2_new

        new_cache_list = list(compacted_cache)
        new_cache_list[layer_idx] = (C1_layer_new, beta_layer_new, C2_layer_new)

        return tuple(new_cache_list)

    def compute_perplexity_for_head_ratio(
        self,
        past_key_values: Tuple,
        article_indices: range,
        baseline_compacted_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        queries: torch.Tensor,
        head_key: str,
        new_ratio: float,
        reference_answers: List[Tuple[str, List[int], str]],
        questions: List[Dict],
        original_seq_len: int,
        keep_before_len: int,
        max_compacted_size: int,
        algorithm: Union[HighestAttentionKeysCompaction, OMPCompaction],
        cached_selection_order: Optional[List[int]] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Recompact one head at new_ratio, swap it into the baseline cache and score it.

        Returns (avg_perplexity, avg_log_perplexity, compact_time, ppl_time) over the
        reference answers.
        """
        article_len = len(article_indices)

        layer_idx = int(head_key[1:].split('H')[0])
        head_idx = int(head_key.split('H')[1])

        compact_start = time.time()

        keys_layer = past_key_values[layer_idx][0][0]
        values_layer = past_key_values[layer_idx][1][0]

        K = keys_layer[head_idx, list(article_indices), :]
        V = values_layer[head_idx, list(article_indices), :]

        new_budget = max(1, int(new_ratio * article_len))

        queries_head = queries[layer_idx, head_idx, :, :]

        if self.algorithm_name == 'omp' and cached_selection_order is not None:
            C1, beta, C2, selected_indices = algorithm.compute_compacted_cache(
                K, V, queries_head, new_budget,
                cached_selection_order=cached_selection_order
            )
        else:
            C1, beta, C2, selected_indices = algorithm.compute_compacted_cache(
                K, V, queries_head, new_budget
            )

        new_hr = HeadCompactionResult(
            C1=C1, beta=beta, C2=C2, selected_indices=selected_indices
        )

        compacted_cache = self.swap_head_in_cache(
            compacted_cache=baseline_compacted_cache,
            layer_idx=layer_idx,
            head_idx=head_idx,
            new_hr=new_hr,
            keep_before_len=keep_before_len,
            max_compacted_size=max_compacted_size,
        )

        compact_time = time.time() - compact_start

        ppl_start = time.time()
        perplexities = []
        for i, (question_id, gen_token_ids, gen_text) in enumerate(reference_answers):
            q = questions[i]
            question_text = q['question']
            options = q.get('options', None)

            question_formatted = format_question(
                self.tokenizer, question_text, options,
                self.model.config._name_or_path if hasattr(self.model.config, '_name_or_path') else None
            )

            ppl, log_ppl = compute_perplexity_on_compacted_cache(
                model=self.model,
                tokenizer=self.tokenizer,
                compacted_cache=compacted_cache,
                generated_token_ids=gen_token_ids,
                question_prompt=question_formatted,
                device=self.device,
                original_seq_len=original_seq_len,
            )
            perplexities.append((ppl, log_ppl))

        ppl_time = time.time() - ppl_start

        avg_ppl = sum(p for p, _ in perplexities) / len(perplexities)
        avg_log_ppl = sum(lp for _, lp in perplexities) / len(perplexities)
        return avg_ppl, avg_log_ppl, compact_time, ppl_time

    def compute_head_influence_curve(
        self,
        past_key_values: Tuple,
        article_indices: range,
        baseline_compacted_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        queries: torch.Tensor,
        head_key: str,
        eval_ratios: List[float],
        reference_answers: List[Tuple[str, List[int], str]],
        questions: List[Dict],
        original_seq_len: int,
        baseline_log_perplexity: float,
        keep_before_len: int,
        max_compacted_size: int,
        algorithm: Union[HighestAttentionKeysCompaction, OMPCompaction],
        cached_selection_order: Optional[List[int]] = None,
    ) -> Tuple[List[Tuple[float, float]], float, float]:
        """Return ([(ratio, delta log-perplexity vs. baseline)], compact_time, ppl_time) for one head."""
        curve = []
        total_compact_time = 0.0
        total_ppl_time = 0.0

        for ratio in eval_ratios:
            avg_ppl, avg_log_ppl, compact_time, ppl_time = self.compute_perplexity_for_head_ratio(
                past_key_values=past_key_values,
                article_indices=article_indices,
                baseline_compacted_cache=baseline_compacted_cache,
                queries=queries,
                head_key=head_key,
                new_ratio=ratio,
                reference_answers=reference_answers,
                questions=questions,
                original_seq_len=original_seq_len,
                keep_before_len=keep_before_len,
                max_compacted_size=max_compacted_size,
                algorithm=algorithm,
                cached_selection_order=cached_selection_order,
            )

            total_compact_time += compact_time
            total_ppl_time += ppl_time

            delta_log_ppl = avg_log_ppl - baseline_log_perplexity
            curve.append((ratio, delta_log_ppl))

        return curve, total_compact_time, total_ppl_time

    def compute_all_head_curves_for_article(
        self,
        article_data: Dict,
        past_key_values: Tuple,
        article_indices: range,
        formatted_context: str,
        baseline_proportions: Dict[str, float],
        target_ratio: float,
        eval_ratios: Union[List[float], Dict[str, List[float]]],
        model_name: str,
        max_new_tokens: int = 2048,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Compute influence curves for all global heads on one article.

        eval_ratios is either one list for all heads or a dict of head key -> list.
        Returns a dict of head key -> [(ratio, delta log-perplexity)].
        """
        ref_layer_idx = self.global_layer_indices[0] if self.global_layer_indices else 0
        original_seq_len = past_key_values[ref_layer_idx][0].shape[2]
        questions = article_data['questions']

        print("Getting reference answers...")
        reference_answers = get_or_generate_reference_answers(
            article_id=article_data['article_id'],
            model=self.model,
            tokenizer=self.tokenizer,
            formatted_context=formatted_context,
            questions=questions,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            device=self.device,
            vllm_model=self.vllm_model,
        )

        baseline_head_results, queries, query_stats = self.compute_baseline_and_queries(
            past_key_values=past_key_values,
            formatted_context=formatted_context,
            article_indices=article_indices,
            baseline_proportions=baseline_proportions,
            target_ratio=target_ratio,
            eval_ratios=eval_ratios,
        )

        article_start = article_indices.start
        seq_len = past_key_values[ref_layer_idx][0].shape[2]
        all_indices = set(range(seq_len))
        article_set = set(article_indices)
        keep_indices = sorted(all_indices - article_set)
        keep_before_len = len([i for i in keep_indices if i < article_start])

        # Pad to the full article length so any ratio up to 1.0 can be swapped in.
        article_len = len(article_indices)
        max_compacted_size = article_len

        print("Building baseline compacted cache...")
        cache_build_start = time.time()
        baseline_compacted_cache = self.build_compacted_cache(
            past_key_values, baseline_head_results, article_indices,
            max_compacted_size=max_compacted_size
        )
        cache_build_time = time.time() - cache_build_start
        print(f"Baseline cache built in {cache_build_time:.2f}s")

        print("Computing baseline perplexity...")
        baseline_perplexities = []
        for i, (question_id, gen_token_ids, gen_text) in enumerate(reference_answers):
            q = questions[i]
            question_formatted = format_question(
                self.tokenizer, q['question'], q.get('options', None),
                model_name
            )
            ppl, log_ppl = compute_perplexity_on_compacted_cache(
                model=self.model,
                tokenizer=self.tokenizer,
                compacted_cache=baseline_compacted_cache,
                generated_token_ids=gen_token_ids,
                question_prompt=question_formatted,
                device=self.device,
                original_seq_len=original_seq_len,
            )
            baseline_perplexities.append((ppl, log_ppl))

        baseline_perplexity = sum(p for p, _ in baseline_perplexities) / len(baseline_perplexities)
        baseline_log_perplexity = sum(lp for _, lp in baseline_perplexities) / len(baseline_perplexities)
        print(f"Baseline perplexity: {baseline_perplexity:.4f} (log: {baseline_log_perplexity:.4f})")

        algorithm = self._create_algorithm()
        per_head_eval_ratios = isinstance(eval_ratios, dict)

        head_curves = {}
        total_global_heads = self.num_global_layers * self.num_kv_heads
        total_compact_time_all = 0.0
        total_ppl_time_all = 0.0
        head_count = 0

        for layer_idx in self.global_layer_indices:
            for head_idx in range(self.num_kv_heads):
                head_key = f"L{layer_idx}H{head_idx}"
                head_count += 1

                if per_head_eval_ratios:
                    head_eval_ratios = eval_ratios[head_key]
                else:
                    head_eval_ratios = eval_ratios

                cached_selection_order = None
                if self.algorithm_name == 'omp' and head_key in baseline_head_results:
                    cached_selection_order = baseline_head_results[head_key].cached_selection_order

                curve, compact_time, ppl_time = self.compute_head_influence_curve(
                    past_key_values=past_key_values,
                    article_indices=article_indices,
                    baseline_compacted_cache=baseline_compacted_cache,
                    queries=queries,
                    head_key=head_key,
                    eval_ratios=head_eval_ratios,
                    reference_answers=reference_answers,
                    questions=questions,
                    original_seq_len=original_seq_len,
                    baseline_log_perplexity=baseline_log_perplexity,
                    keep_before_len=keep_before_len,
                    max_compacted_size=max_compacted_size,
                    algorithm=algorithm,
                    cached_selection_order=cached_selection_order,
                )

                head_curves[head_key] = curve
                total_compact_time_all += compact_time
                total_ppl_time_all += ppl_time

                print(f"  {head_key} ({head_count}/{total_global_heads}): "
                      f"compact={compact_time:.2f}s, ppl={ppl_time:.2f}s, "
                      f"curve={[(r, f'{d:.3f}') for r, d in curve]}")

        print(f"\nTotal time: compact={total_compact_time_all:.1f}s, ppl={total_ppl_time_all:.1f}s")

        return head_curves


def aggregate_head_curves(
    all_article_curves: List[Dict[str, List[Tuple[float, float]]]],
) -> Dict[str, List[Tuple[float, float]]]:
    """Average per-article head curves pointwise (all curves must share the same ratios)."""
    if not all_article_curves:
        return {}

    head_keys = list(all_article_curves[0].keys())

    aggregated = {}

    for head_key in head_keys:
        curves = [article_curves[head_key] for article_curves in all_article_curves]
        ratios = [point[0] for point in curves[0]]

        averaged_curve = []
        for i, ratio in enumerate(ratios):
            deltas = [curve[i][1] for curve in curves]
            avg_delta = sum(deltas) / len(deltas)
            averaged_curve.append((ratio, avg_delta))

        aggregated[head_key] = averaged_curve

    return aggregated


def save_head_curves(
    head_curves: Dict[str, List[Tuple[float, float]]],
    output_path: str,
    metadata: Optional[Dict] = None,
):
    """Save head curves and optional metadata to JSON."""
    output = {
        'metadata': metadata or {},
        'head_curves': head_curves,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved head curves to {output_path}")


def load_head_curves(input_path: str) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict]:
    """Load (head_curves, metadata) from JSON."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    head_curves = {}
    for head_key, curve in data['head_curves'].items():
        head_curves[head_key] = [(point[0], point[1]) for point in curve]

    return head_curves, data.get('metadata', {})


def load_and_aggregate_article_curves(
    article_curves_dir: str,
) -> Tuple[Dict[str, List[Tuple[float, float]]], List[Dict]]:
    """Load all article_*.json curves in a directory; return (averaged curves, per-file metadata)."""
    from pathlib import Path
    import glob

    dir_path = Path(article_curves_dir)
    pattern = str(dir_path / 'article_*.json')
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"No article curve files found in {article_curves_dir}")

    all_curves = []
    all_metadata = []

    for filepath in files:
        curves, metadata = load_head_curves(filepath)
        all_curves.append(curves)
        all_metadata.append(metadata)

    print(f"Loaded curves from {len(files)} articles")

    aggregated = aggregate_head_curves(all_curves)
    return aggregated, all_metadata
