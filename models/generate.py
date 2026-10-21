from typing import List, Tuple, Optional, Dict, Set, Any, TYPE_CHECKING

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache

if TYPE_CHECKING:
    from vllm import LLM

Tensor = torch.Tensor

from .cache import CompactedPrefixCache


def get_generation_params(model) -> Dict[str, Optional[float]]:
    """Return the model's default sampling 'temperature', 'top_k', 'top_p' (None where unset)."""
    gen_config = getattr(model, 'generation_config', None)

    temperature = None
    top_k = None
    top_p = None

    if gen_config is not None:
        if getattr(gen_config, 'do_sample', False):
            temp = getattr(gen_config, 'temperature', None)
            if temp is not None:
                temperature = temp

            tk = getattr(gen_config, 'top_k', None)
            if tk is not None and tk > 0:
                top_k = tk

            tp = getattr(gen_config, 'top_p', None)
            if tp is not None and tp < 1.0:
                top_p = tp

    return {
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
    }


@torch.inference_mode()
def chunked_prefill(
    model,
    input_ids: Tensor,
    past_key_values: Optional[Cache] = None,
    chunk_size: int = 4096,
    **model_kwargs,
) -> Any:
    """
    Prefill in chunks of chunk_size tokens to avoid OOM on large attention masks.

    Works with CompactedPrefixCache or DynamicCache; the cache is updated in place.
    Returns the model output of the last chunk.
    """
    device = input_ids.device
    batch_size, input_len = input_ids.shape

    # Create the cache here so it persists across chunks; config-aware so sliding layers
    # are window-cropped as in a single full forward.
    if past_key_values is None:
        past_key_values = DynamicCache(config=model.config)

    start_pos = past_key_values.get_seq_length()

    outputs = None

    for chunk_start in range(0, input_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, input_len)
        chunk_input_ids = input_ids[:, chunk_start:chunk_end]

        chunk_cache_position = torch.arange(
            start_pos + chunk_start,
            start_pos + chunk_end,
            device=device,
            dtype=torch.long,
        )

        # Covers the existing cache, earlier chunks and the current chunk.
        current_kv_len = start_pos + chunk_end
        chunk_attention_mask = torch.ones(
            (batch_size, current_kv_len),
            device=device,
            dtype=torch.long,
        )

        outputs = model(
            input_ids=chunk_input_ids,
            past_key_values=past_key_values,
            cache_position=chunk_cache_position,
            attention_mask=chunk_attention_mask,
            use_cache=True,
            **model_kwargs,
        )

    return outputs


def get_sliding_layer_info(model) -> Tuple[Set[int], Optional[int]]:
    """Return (sliding-window layer indices, window size or None) from the model config."""
    sliding_layer_indices = set()
    sliding_window = None

    config = getattr(model, 'config', None)
    if config is not None:
        layer_types = getattr(config, 'layer_types', None)
        sliding_window = getattr(config, 'sliding_window', None)
        if layer_types is not None:
            for layer_idx, layer_type in enumerate(layer_types):
                if layer_type == "sliding_attention":
                    sliding_layer_indices.add(layer_idx)

    return sliding_layer_indices, sliding_window

@torch.inference_mode()
def generate_with_full_context_batch(
    model,
    tokenizer: AutoTokenizer,
    full_prompts: List[str],
    max_new_tokens: int = 128,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> List[str]:
    """
    Sample answers for chat-formatted prompts with the full context (no cache reuse).

    Unset sampling parameters fall back to the model's defaults. Returns only the
    generated text.
    """
    device = next(model.parameters()).device

    gen_params = get_generation_params(model)
    if temperature is None:
        temperature = gen_params['temperature'] if gen_params['temperature'] is not None else 1.0
    if top_k is None:
        top_k = gen_params['top_k']  # None means disabled
    if top_p is None:
        top_p = gen_params['top_p'] if gen_params['top_p'] is not None else 1.0

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prompts already contain <bos>.
    inputs = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    ).to(device)

    tokenizer.padding_side = original_padding_side

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    answer_texts = []
    input_lengths = inputs.input_ids.shape[1]  # Length includes padding
    for i, output_ids in enumerate(outputs):
        gen_only = output_ids[input_lengths:]
        answer_text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        answer_texts.append(answer_text)

    return answer_texts


@torch.inference_mode()
def generate_with_compacted_cache_batch(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    compacted_cache: Tuple[Tuple[Tensor, Tensor, Tensor], ...],
    max_new_tokens: int = 128,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    original_seq_len: Optional[int] = None,
) -> List[str]:
    """
    Sample answers for a batch of left-padded prompts on top of a shared compacted prefix.

    compacted_cache holds (C1, beta, C2) per layer; sliding layers store their raw window
    with beta=0. original_seq_len is the uncompacted prefix length, used for the RoPE offset.
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    gen_params = get_generation_params(model)
    if temperature is None:
        temperature = gen_params['temperature'] if gen_params['temperature'] is not None else 1.0
    if top_k is None:
        top_k = gen_params['top_k']  # None means disabled
    if top_p is None:
        top_p = gen_params['top_p'] if gen_params['top_p'] is not None else 1.0

    sliding_layer_indices, sliding_window = get_sliding_layer_info(model)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    input_attn_mask = enc["attention_mask"]
    
    batch_size = input_ids.shape[0]
    tokenizer.padding_side = original_padding_side

    pad_counts = (input_attn_mask == 0).sum(dim=1)

    # Expand the prefix from (1, KV, t, D) or (KV, t, D) to (B, KV, t, D).
    expanded_layers = []
    def expand_kv_tensor(t: Tensor) -> Tensor:
        t = t.to(device=device, dtype=model_dtype)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.shape[0] == batch_size:
            return t
        if t.shape[0] == 1 and batch_size > 1:
            return t.expand(batch_size, *t.shape[1:]).contiguous()
        return t

    def expand_beta_tensor(beta: Tensor) -> Tensor:
        beta = beta.to(device=device, dtype=model_dtype)
        if beta.dim() == 2:
            beta = beta.unsqueeze(0)
        if beta.shape[0] == batch_size:
            return beta
        if beta.shape[0] == 1 and batch_size > 1:
            return beta.expand(batch_size, *beta.shape[1:]).contiguous()
        return beta

    for (C1, beta, C2) in compacted_cache:
        expanded_layers.append((
            expand_kv_tensor(C1),
            expand_beta_tensor(beta),
            expand_kv_tensor(C2),
        ))

    cache = CompactedPrefixCache(
        tuple(expanded_layers),
        original_seq_len=original_seq_len,
        pad_counts=pad_counts,
        sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
        sliding_window=sliding_window,
    )
    past_seen_tokens = cache.get_seq_length() # Length of the compacted prefix (t)

    # (B, prefix_len + prompt_len): prefix always attended, prompt follows the tokenizer mask.
    if past_seen_tokens > 0:
        prefix_mask = torch.ones(
            (batch_size, past_seen_tokens),
            device=device,
            dtype=input_attn_mask.dtype,
        )
        attention_mask = torch.cat([prefix_mask, input_attn_mask], dim=1)
    else:
        attention_mask = input_attn_mask

    input_len = input_ids.shape[1]
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + input_len,
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id
        )

    answer_texts = []
    for i, output_ids in enumerate(outputs):
        gen_only = output_ids[input_len:]
        answer_text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        answer_texts.append(answer_text)

    return answer_texts


def generate_with_vllm_batch(
    vllm_model: "LLM",
    full_prompts: List[str],
    max_new_tokens: int = 128,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> List[str]:
    """Sample full-context answers with vLLM (unset sampling parameters: 1.0, disabled, 1.0)."""
    from vllm import SamplingParams

    if temperature is None:
        temperature = 1.0
    if top_k is None:
        top_k = -1
    if top_p is None:
        top_p = 1.0

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    outputs = vllm_model.generate(full_prompts, sampling_params)

    answer_texts = [output.outputs[0].text.strip() for output in outputs]

    return answer_texts
