"""Evaluation helpers: model and vLLM loading, KV cache extraction, chat formatting and answer parsing."""
import re
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.generate import get_generation_params

QWEN_USER_START = "<|im_start|>user"
QWEN_USER_END = "<|im_end|>"
LLAMA_USER_START = "<|start_header_id|>user"
LLAMA_USER_END = "<|eot_id|>"
GEMMA_USER_START = "<start_of_turn>user"
GEMMA_USER_END = "<end_of_turn>"


def detect_user_tags(formatted_context: str) -> Tuple[str, str]:
    """Return the (user_start_tag, user_end_tag) of the chat template used in ``formatted_context``."""
    if LLAMA_USER_START in formatted_context:
        return LLAMA_USER_START, LLAMA_USER_END
    if QWEN_USER_START in formatted_context:
        return QWEN_USER_START, QWEN_USER_END
    if GEMMA_USER_START in formatted_context:
        return GEMMA_USER_START, GEMMA_USER_END
    raise ValueError("Could not detect user tags from formatted context.")


def compute_article_indices(
    tokenizer,
    formatted_context: str,
    article_text: str,
) -> range:
    """Token index range of ``article_text`` within the chat-formatted context."""
    user_start_tag, _ = detect_user_tags(formatted_context)

    user_start_pos = formatted_context.find(user_start_tag)
    if user_start_pos == -1:
        raise ValueError(f"Could not find '{user_start_tag}' tag in formatted context")

    article_text_start = formatted_context.find('\n', user_start_pos + len(user_start_tag))
    if article_text_start == -1:
        raise ValueError("Could not find newline after user start tag")
    article_text_start += 1

    article_text_end = article_text_start + len(article_text)

    prefix_text = formatted_context[:article_text_start]
    prefix_tokens = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    article_start_idx = prefix_tokens['input_ids'].shape[1]

    article_with_prefix = formatted_context[:article_text_end]
    article_with_prefix_tokens = tokenizer(article_with_prefix, return_tensors="pt", add_special_tokens=False)
    article_end_idx = article_with_prefix_tokens['input_ids'].shape[1]

    return range(article_start_idx, article_end_idx)


def load_model_and_tokenizer(
    model_name: str,
    device: str = None,
    dtype: Optional[torch.dtype] = None,
    max_model_len: Optional[int] = None,
):
    """Load the HF model (the project's modeling classes where available) and tokenizer.

    ``max_model_len`` beyond the native context enables YaRN rope scaling.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {
        "device_map": device if device == "cuda" else None,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa"
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    if max_model_len is not None:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        original_max_pos = getattr(config, 'max_position_embeddings', 32768)
        if max_model_len > original_max_pos:
            model_kwargs["rope_scaling"] = {
                "rope_type": "yarn",
                "factor": max_model_len / original_max_pos,
                "original_max_position_embeddings": original_max_pos,
            }
            print(f"Enabling YaRN rope scaling: {original_max_pos} -> {max_model_len} (factor={max_model_len/original_max_pos:.1f}x)")

    # The project's modeling classes support compacted caches with a bias term.
    if "qwen" in model_name.lower():
        from transformers import AutoConfig
        qwen_model_type = AutoConfig.from_pretrained(model_name).model_type
        if qwen_model_type == "qwen3_moe":
            from models.qwen3_moe import Qwen3MoeForCausalLM
            model = Qwen3MoeForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        else:
            from models.qwen3 import Qwen3ForCausalLM
            model = Qwen3ForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
    elif "llama" in model_name.lower():
        from models.llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    elif "gemma" in model_name.lower():
        from models.gemma3 import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    else:
        print("Warning: Using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

    if device == "cpu":
        model = model.to(device)

    model.eval()
    print(f"Model loaded successfully")
    return model, tokenizer


def offload_model_to_cpu(model) -> None:
    """Move the model to CPU to free GPU memory."""
    if model is not None:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def reload_model_to_gpu(model, device: str = "cuda") -> None:

    if model is not None:
        model.to(device)
        model.eval()


def initialize_vllm(
    model_name: str,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Initialize the vLLM engine used for sampling queries, summaries and reference answers.

    With two or more visible GPUs the engine gets the last one to itself and is
    kept awake. Otherwise it shares the GPU with the HF model and starts asleep.

    Parameters
    ----------
    gpu_memory_utilization : float, optional
        None -> 0.9 on a dedicated GPU, 0.5 on a shared one.
    max_model_len : int, optional
        None -> model default. Beyond the native context, enables YaRN rope scaling.
    seed : int, optional
        Engine-level sampling seed.
    """
    print("Initializing vLLM model (will be reused across all articles)...")

    # 'spawn' avoids CUDA fork issues.
    import os
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    try:
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "vLLM is required for this run."
        )

    # With 2+ GPUs the engine gets the last one and the HF model stays on
    # cuda:0. The engine runs in a spawned process, so restricting
    # CUDA_VISIBLE_DEVICES during construction pins it without affecting
    # this process's CUDA context.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [d.strip() for d in visible.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(torch.cuda.device_count())]
    dedicated = len(devices) >= 2

    if gpu_memory_utilization is None:
        # On a shared GPU leave room for the HF model.
        gpu_memory_utilization = 0.9 if dedicated else 0.5

    llm_kwargs = {
        "model": model_name,
        "tokenizer": model_name,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_sleep_mode": not dedicated,
    }
    if seed is not None:
        llm_kwargs["seed"] = int(seed)

    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        original_max_pos = getattr(config, 'max_position_embeddings', 32768)
        if max_model_len > original_max_pos:
            llm_kwargs["hf_overrides"] = {
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": max_model_len / original_max_pos,
                    "original_max_position_embeddings": original_max_pos,
                }
            }
            print(f"Using extended context: max_model_len={max_model_len} with YaRN rope scaling (factor={max_model_len/original_max_pos:.1f}x)")
        else:
            print(f"Setting max_model_len={max_model_len} (within native {original_max_pos})")

    if dedicated:
        target = devices[-1]
        print(f"Two GPUs visible: pinning the vLLM engine to CUDA device "
              f"{target} (HF model keeps the first device)")
        os.environ["CUDA_VISIBLE_DEVICES"] = target
        try:
            vllm_model = LLM(**llm_kwargs)
        finally:
            if visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible
    else:
        vllm_model = LLM(**llm_kwargs)

    # On a dedicated GPU make sleep()/wake_up() no-ops so callers need not
    # distinguish the two regimes.
    vllm_model.dedicated_gpu = dedicated
    if dedicated:
        vllm_model.sleep = lambda *args, **kwargs: None
        vllm_model.wake_up = lambda *args, **kwargs: None
        print(f"vLLM initialized on its own GPU and kept awake (model: {model_name})")
    else:
        vllm_model.sleep()
        print(f"vLLM initialized and put to sleep (model: {model_name})")

    return vllm_model


def extract_full_kv_cache(
    model,
    tokenizer,
    article_text: str,
    device: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[int, Tuple, range, str, int]:
    """Prefill the chat-formatted article and return its full KV cache.

    Returns
    -------
    (seq_len, past_key_values, article_indices, formatted_context, original_token_length)
    """
    formatted_context = format_context(tokenizer, article_text, system_prompt, model_name)

    # The chat template already adds <bos>.
    inputs = tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
    original_token_length = inputs['input_ids'].shape[1]

    input_ids = inputs['input_ids'].to(device)
    seq_len = input_ids.shape[1]

    article_indices = compute_article_indices(tokenizer, formatted_context, article_text)

    with torch.no_grad():
        if seq_len > CHUNKED_PREFILL_MIN_TOKENS:
            # A single forward materialises (seq, seq) masks and scores on
            # sliding-window layers; chunking bounds them at (chunk, seq) and
            # yields the same cache up to bf16 accumulation order.
            from models.generate import chunked_prefill
            outputs = chunked_prefill(
                model,
                input_ids=input_ids,
                chunk_size=CHUNKED_PREFILL_CHUNK_TOKENS,
            )
        else:
            outputs = model(
                input_ids,
                use_cache=True
            )
    return seq_len, outputs.past_key_values, article_indices, formatted_context, original_token_length


# Contexts longer than this are prefilled in chunks by extract_full_kv_cache.
CHUNKED_PREFILL_MIN_TOKENS = 65536
CHUNKED_PREFILL_CHUNK_TOKENS = 8192


def get_default_system_prompt(model_name: Optional[str] = None) -> str:
    """Default QA system prompt; empty for Gemma, which has no system role."""
    if model_name and "gemma" in model_name.lower():
        return ""
    return "You are a helpful assistant. Answer a question based on the provided context."


def format_context(
    tokenizer,
    article: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Apply the chat template to the article as a user turn (after the system prompt, if any)."""
    if system_prompt is None:
        system_prompt = get_default_system_prompt(model_name)

    if system_prompt == "":
        context_messages = [
            {"role": "user", "content": article}
        ]
    else:
        context_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": article}
        ]
    return tokenizer.apply_chat_template(
        context_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )


def format_question(
    tokenizer,
    question: str,
    options: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    enable_thinking: bool = True,
    answer_prefix: Optional[str] = None,
) -> str:
    """Format a question (with lettered options if given) as a chat prompt ready for generation.

    Parameters
    ----------
    enable_thinking : bool
        False for string-match tasks (lca), where thinking would interfere with scoring.
    answer_prefix : str, optional
        Appended after the generation prompt (lca opens a code block).
    """
    is_qwen = model_name and "qwen" in model_name.lower()
    is_gemma = model_name and "gemma" in model_name.lower()

    if options:
        num_options = len(options)
        option_labels = [chr(65 + i) for i in range(num_options)]  # 65 is 'A'
        formatted_options = '\n'.join([
            f"{label}. {option}"
            for label, option in zip(option_labels, options)
        ])

        if num_options <= 4:
            option_list = "A, B, C, or D"
        elif num_options == 5:
            option_list = "A, B, C, D, or E"
        else:
            option_list = ", ".join(option_labels[:-1]) + f", or {option_labels[-1]}"

        if is_qwen:
            if model_name == "Qwen/Qwen3-4B-Instruct-2507": # need to specify step by step
                instruction = f"Please think step by step very briefly and then, on a new line, respond with the letter ({option_list}) of the correct option. If you are not sure, still make a guess."
            else:
                instruction = f"Please think very briefly and then respond with **only** the letter ({option_list}) of the correct option. If you are not sure, still make a guess."
        elif is_gemma: # with "briefly" Gemma skips thinking
            instruction = f"Please think step by step and then, on a new line, respond with **only** the letter ({option_list}) of the correct option. If you are not sure, still make a guess."
        else:
            instruction = f"Please think step by step briefly and then, on a new line, respond with **only** the letter ({option_list}) of the correct option. If you are not sure, still make a guess."

        content = f"{question}\n\n{formatted_options}\n\n{instruction}"
    else:
        content = question

    question_messages = [
        {"role": "user", "content": content}
    ]

    result = tokenizer.apply_chat_template(
        question_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    # The context already starts with <bos>.
    if is_gemma:
        result = result[len("<bos>"):]

    if answer_prefix:
        result = result + answer_prefix

    return result


def parse_model_choice(answer_text: str, max_options: int = 5) -> Optional[int]:
    """Parse the chosen option as a 1-indexed int (1=A), or None if unparseable.

    Looks after ``</think>`` if present (None if thinking is unfinished), and
    accepts forms like "A", "the answer is: D." or a standalone capital letter.
    """
    has_open_think = '<think>' in answer_text
    has_close_think = '</think>' in answer_text

    if has_open_think and not has_close_think:
        return None

    if has_close_think:
        parts = answer_text.split('</think>')
        if len(parts) > 1:
            answer_text = parts[-1].strip()
    elif not has_open_think:
        # Without thinking, use the last non-empty line.
        lines = answer_text.splitlines()
        for line in reversed(lines):
            if line.strip():
                answer_text = line.strip()
                break

    answer_lower = answer_text.lower()
    for pattern in ['answer:', 'answer is', 'correct option is']:
        if pattern in answer_lower:
            parts = answer_lower.split(pattern)
            if len(parts) > 1:
                candidate = parts[-1].strip().lstrip(':').strip().rstrip('.').strip()
                candidate = candidate.replace('*', '')
                if candidate and candidate[0] in tuple(chr(97 + i) for i in range(max_options)):  # a-e (or more)
                    return ord(candidate[0]) - ord('a') + 1

    answer_clean = answer_lower.split(".")[0].strip()
    answer_clean = answer_clean.replace('*', '')

    valid_options = tuple(chr(97 + i) for i in range(max_options))  # a, b, c, d, e, ...
    if answer_clean in valid_options:
        return ord(answer_clean) - ord('a') + 1

    # Fallback: the last standalone capital option letter.
    valid_upper = tuple(chr(65 + i) for i in range(max_options))  # A, B, C, D, E, ...
    pattern = r'(?<![a-zA-Z])([' + ''.join(valid_upper) + r'])(?![a-zA-Z])'
    matches = re.findall(pattern, answer_text)
    if matches:
        letter = matches[-1]
        return ord(letter) - ord('A') + 1

    return None


def compute_cache_memory_stats(
    original_cache: Tuple,
    compacted_cache: Tuple,
) -> Dict:
    """Element counts of the original (K, V) and compacted (C1, beta, C2) caches and the reduction in percent."""
    original_cache_size = sum([k.numel() + v.numel() for k, v in original_cache])
    compacted_cache_size = sum([(c1.numel() + beta.numel() + c2.numel()) for c1, beta, c2 in compacted_cache])

    memory_reduction_pct = 100 * (1 - compacted_cache_size / original_cache_size)

    return {
        'original_cache_size': original_cache_size,
        'compacted_cache_size': compacted_cache_size,
        'memory_reduction_pct': memory_reduction_pct,
    }


def print_query_generation_stats(compaction_stats: Dict):

    if 'query_generation' not in compaction_stats:
        return

    qstats = compaction_stats['query_generation']
    query_generation_time = qstats.get('query_generation_time', 0.0)
    final_n_queries = qstats.get('final_n_queries_per_kv_head')

    # Text-based methods generate no queries.
    if final_n_queries is None:
        return

    print(f"\nQuery generation statistics:")
    print(f"  Queries per KV head: {final_n_queries}")
    print(f"  Query generation time: {query_generation_time:.2f}s")

    if 'methods_used' in qstats:
        for method, method_info in qstats['methods_used'].items():
            print(f"\n  {method}:")
            print(f"    Fraction: {method_info['fraction']:.1%}")
            print(f"    Queries requested: {method_info['n_queries_requested_per_kv_head']}")
            print(f"    Queries actual: {method_info['n_queries_actual_per_kv_head']}")

            if method == 'self_study' and 'stats' in method_info:
                details = method_info['stats']
                print(f"    Conversations generated: {details.get('n_conversations', 0)}")
                print(f"    Tokens extracted: {details.get('n_self_study_tokens_extracted', 0)}")
                if 'n_self_study_tokens_subsampled' in details:
                    print(f"    Tokens subsampled to: {details.get('n_self_study_tokens_subsampled', 0)}")


def check_reference_answers_cached(
    article_id: str,
    questions: List[Dict],
    model_name: str,
    cache_dir: str = "data/reference_answers_cache",
) -> bool:
    """True if a reference-answer cache file exists for exactly these questions."""
    import json
    import hashlib
    from pathlib import Path

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return False

    # Same file naming as get_or_generate_reference_answers.
    question_ids = [q.get('question_unique_id', f'q_{i}') for i, q in enumerate(questions)]
    question_ids_str = '_'.join(sorted(question_ids))
    question_ids_hash = hashlib.md5(question_ids_str.encode()).hexdigest()[:16]

    model_short_name = model_name.split('/')[-1]

    cache_file = cache_path / f"{model_short_name}_{article_id}_{question_ids_hash}.json"
    if not cache_file.exists():
        return False

    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        cached_question_ids = [item['question_id'] for item in cache_data['generations']]
        return cached_question_ids == question_ids
    except (json.JSONDecodeError, KeyError):
        return False


def get_or_generate_reference_answers(
    article_id: str,
    model,
    tokenizer,
    formatted_context: str,
    questions: List[Dict],
    model_name: str,
    max_new_tokens: int = 4096,
    cache_dir: str = "data/reference_answers_cache",
    device: str = "cuda",
    vllm_model=None,
) -> List[Tuple[str, List[int], str]]:
    """Load or generate (and cache) full-context answers, used as perplexity targets.

    Generates with ``vllm_model`` if given, else with the HF ``model``. The cache
    file is keyed by model, article and a hash of the question IDs.

    Returns
    -------
    list of (question_id, token_ids, generated_text)
    """
    import json
    import hashlib
    from pathlib import Path

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    question_ids = [q.get('question_unique_id', f'q_{i}') for i, q in enumerate(questions)]
    question_ids_str = '_'.join(sorted(question_ids))
    question_ids_hash = hashlib.md5(question_ids_str.encode()).hexdigest()[:16]

    model_short_name = model_name.split('/')[-1]

    cache_file = cache_path / f"{model_short_name}_{article_id}_{question_ids_hash}.json"

    if cache_file.exists():
        print(f"Loading cached reference answers for article {article_id} ({len(questions)} questions)")
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        cached_question_ids = [item['question_id'] for item in cache_data['generations']]
        if cached_question_ids == question_ids:
            results = []
            for item in cache_data['generations']:
                token_ids = tokenizer.encode(item['generated_text'], add_special_tokens=False)
                results.append((item['question_id'], token_ids, item['generated_text']))
            return results
        else:
            print(f"Warning: Cached questions don't match requested questions, regenerating...")

    print(f"Generating reference answers for article {article_id} ({len(questions)} questions)")

    question_ids = [q.get('question_unique_id', f'q_{i}') for i, q in enumerate(questions)]
    full_prompts = []

    for i, q in enumerate(questions):
        question_text = q['question']
        options = q.get('options', None)

        question_formatted = format_question(tokenizer, question_text, options, model_name)

        full_prompt = formatted_context + question_formatted
        full_prompts.append(full_prompt)

    if vllm_model is not None:
        from models.generate import generate_with_vllm_batch
        print(f"  Generating answers in batch using vLLM...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vllm_model.wake_up()
        gen_params = get_generation_params(model)
        generated_texts = generate_with_vllm_batch(
            vllm_model=vllm_model,
            full_prompts=full_prompts,
            max_new_tokens=max_new_tokens,
            temperature=gen_params['temperature'],
            top_k=gen_params['top_k'],
            top_p=gen_params['top_p'],
        )
        vllm_model.sleep()
    else:
        from models.generate import generate_with_full_context_batch
        print(f"  Generating answers in batch using HuggingFace...")
        generated_texts = generate_with_full_context_batch(
            model=model,
            tokenizer=tokenizer,
            full_prompts=full_prompts,
            max_new_tokens=max_new_tokens,
        )

    results = []
    for question_id, generated_text in zip(question_ids, generated_texts):
        gen_token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
        results.append((question_id, gen_token_ids, generated_text))

    # Only text is cached; token IDs are recomputed on load.
    cache_data = {
        'model_name': model_name,
        'article_id': article_id,
        'question_ids': question_ids,
        'question_ids_hash': question_ids_hash,
        'max_new_tokens': max_new_tokens,
        'generations': [
            {
                'question_id': qid,
                'generated_text': text
            }
            for qid, tids, text in results
        ]
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    total_tokens = sum(len(tids) for _, tids, _ in results)
    print(f"Cached reference answers for article {article_id} ({len(questions)} questions, {total_tokens} total tokens)")

    return results


def compute_perplexity_on_compacted_cache(
    model,
    tokenizer,
    compacted_cache: Tuple,
    generated_token_ids: List[int],
    question_prompt: str,
    device: str = "cuda",
    original_seq_len: Optional[int] = None,
) -> Tuple[float, float]:
    """Perplexity of the answer tokens given the question prompt, under the compacted cache.

    ``compacted_cache`` holds (C1, beta, C2) per layer. ``original_seq_len`` sets
    the RoPE position of the new tokens. Returns (perplexity, mean NLL).
    """
    question_token_ids = tokenizer.encode(question_prompt, add_special_tokens=False)

    full_token_ids = question_token_ids + generated_token_ids
    full_token_ids_tensor = torch.tensor([full_token_ids], dtype=torch.long, device=device)

    from models.cache import CompactedPrefixCache
    from models.generate import get_sliding_layer_info

    sliding_layer_indices, sliding_window = get_sliding_layer_info(model)

    moved_layers = []
    for (C1, beta, C2) in compacted_cache:
        moved_layers.append((
            C1.to(device=device, dtype=model.dtype),
            beta.to(device=device, dtype=model.dtype),
            C2.to(device=device, dtype=model.dtype),
        ))

    cache = CompactedPrefixCache(
        tuple(moved_layers),
        original_seq_len=original_seq_len,
        sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
        sliding_window=sliding_window,
    )

    # Feed the tokens in slices: the beta-bias mask forces SDPA onto the math
    # path, which materialises a (q_len, kv_len) buffer per head.
    ppl_prefill_chunk = 256
    logits_slices = []
    with torch.no_grad():
        for s in range(0, full_token_ids_tensor.size(1), ppl_prefill_chunk):
            outputs = model(
                input_ids=full_token_ids_tensor[:, s:s + ppl_prefill_chunk],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            logits_slices.append(outputs.logits)

    # logits: (batch_size, seq_len, vocab_size)
    logits = torch.cat(logits_slices, dim=1) if len(logits_slices) > 1 \
        else logits_slices[0]

    # Score only the answer tokens.
    num_question_tokens = len(question_token_ids)
    seq_len = full_token_ids_tensor.size(1)

    start_idx = max(num_question_tokens - 1, 0)

    # logits[num_question_tokens - 1] predicts answer token 0.
    answer_logits = logits[:, start_idx:seq_len-1, :].contiguous()
    answer_labels = full_token_ids_tensor[:, start_idx+1:seq_len].contiguous()

    if answer_labels.numel() == 0:
        return float("nan"), float("nan")

    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(answer_logits.view(-1, answer_logits.size(-1)), answer_labels.view(-1))

    log_perplexity = loss.item()

    perplexity = torch.exp(loss).item()

    return perplexity, log_perplexity
