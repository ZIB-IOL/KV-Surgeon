"""Registry of predefined conversation specs for self-study query generation."""

from typing import Dict, List
from .config import ConversationSpec


def extract_after_thinking_then_split(text: str) -> List[str]:
    """Split the text after </think> into items; returns [] if there are more than 5.

    Tries dash separator lines, then numbered items, then blank lines.
    """
    import re

    if "</think>" in text:
        parts = text.split("</think>", 1)
        content = parts[1].strip() if len(parts) > 1 else text
    else:
        content = text.strip()

    if not content:
        return []

    items = re.split(r'\n+\s*-{3,}\s*\n+', content)

    if len(items) == 1:
        numbered_items = re.split(r'\n+(?=\d+[\.\)]\s)', content)
        if len(numbered_items) > 1:
            items = numbered_items

    if len(items) == 1:
        items = content.split("\n\n")

    items = [item.strip() for item in items if item.strip()]

    # Rejoin questions with option blocks ("A) ...") that the split separated.
    if len(items) > 5:
        options_pattern = re.compile(r'^A[\)\.]')
        might_be_pairs = all(
            options_pattern.match(items[i])
            for i in range(1, len(items), 2)
        )
        if might_be_pairs and len(items) % 2 == 0:
            merged = []
            for i in range(0, len(items), 2):
                merged.append(items[i] + "\n\n" + items[i + 1])
            items = merged

    if len(items) > 5:
        return []
    return items


CONVERSATION_SPEC_REGISTRY: Dict[str, ConversationSpec] = {
    "3_question": ConversationSpec(
        seed_prompt=(
            "Write 3 questions that test understanding of different parts of the context. "
            "Answer with just the 3 questions and options (do not say the correct answer), each one separated with 2 newlines."
        ),
        extraction_fn=extract_after_thinking_then_split,
        enable_thinking_a=True,
        max_tokens_a=8192,
        enable_thinking_b=True,
        max_tokens_b=4096,
    ),
    "repeat": ConversationSpec(
        conversation_starter="Repeat the previous context verbatim.",
        prefill_with_article=True,
    ),
    "summarize": ConversationSpec(
        conversation_starter="Summarize the main points of the context.",
        enable_thinking_b=False,
        max_tokens_b=2048,
    ),
    "structure_json": ConversationSpec(
        conversation_starter="Structure the information in JSON form and include all important details like dates, times, names, and numerical values.",
        enable_thinking_b=False,
        max_tokens_b=2048,
    ),
    "aggregate": ConversationSpec(
        conversation_starter="Aggregate all the key facts mentioned in the context.",
        enable_thinking_b=False,
        max_tokens_b=2048,
    ),
}


def get_spec(spec_key: str) -> ConversationSpec:
    """Look up a conversation spec by key."""
    if spec_key not in CONVERSATION_SPEC_REGISTRY:
        raise KeyError(
            f"Unknown conversation spec: {spec_key}. "
            f"Available specs: {list(CONVERSATION_SPEC_REGISTRY.keys())}"
        )
    return CONVERSATION_SPEC_REGISTRY[spec_key]


def get_specs(spec_keys: List[str]) -> List[ConversationSpec]:
    """Look up several conversation specs by key."""
    return [get_spec(key) for key in spec_keys]


def repeat_specs(spec_config: List[tuple[str, int]]) -> List[ConversationSpec]:
    """Expand [(spec_key, count), ...] into a list of specs, e.g. [("3_question", 2), ("repeat", 1)]."""
    result = []
    for spec_key, count in spec_config:
        spec = get_spec(spec_key)
        result.extend([spec] * count)
    return result
