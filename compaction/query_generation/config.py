"""Configuration for query generation strategies."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable


@dataclass
class ConversationSpec:
    """
    One self-study conversation type.

    Model A turns seed_prompt into one or more conversation starters (split by
    extraction_fn); alternatively conversation_starter is used directly.
    Exactly one of the two must be set. Model B answers each starter.

    Parameters
    ----------
    prefill_with_article : bool
        Use the article itself as Model B's answer instead of generating one.
    enable_thinking_a, max_tokens_a : optional
        Model A settings; only valid with seed_prompt.
    enable_thinking_b, max_tokens_b : optional
        Model B settings; ignored with prefill_with_article.
    """
    seed_prompt: Optional[str] = None
    conversation_starter: Optional[str] = None
    extraction_fn: Optional[Callable[[str], List[str]]] = None
    prefill_with_article: bool = False
    enable_thinking_a: Optional[bool] = None
    max_tokens_a: Optional[int] = None
    enable_thinking_b: Optional[bool] = False
    max_tokens_b: Optional[int] = 4096

    def __post_init__(self):
        """Validate that exactly one of seed_prompt or conversation_starter is provided."""
        modes_set = sum([
            self.seed_prompt is not None,
            self.conversation_starter is not None,
        ])

        if modes_set == 0:
            raise ValueError(
                "ConversationSpec: exactly one of 'seed_prompt' or 'conversation_starter' must be provided"
            )
        if modes_set > 1:
            raise ValueError(
                "ConversationSpec: only one of 'seed_prompt' or 'conversation_starter' can be provided"
            )

        if self.conversation_starter is not None and self.extraction_fn is not None:
            raise ValueError(
                "ConversationSpec: extraction_fn cannot be used with direct conversation_starter"
            )
        if self.conversation_starter is not None and self.enable_thinking_a is not None:
            raise ValueError(
                "ConversationSpec: enable_thinking_a cannot be used with direct conversation_starter"
            )
        if self.conversation_starter is not None and self.max_tokens_a is not None:
            raise ValueError(
                "ConversationSpec: max_tokens_a cannot be used with direct conversation_starter"
            )

    def is_direct(self) -> bool:
        """Return True if this is a direct conversation starter (no Model A)."""
        return self.conversation_starter is not None

    def uses_extraction(self) -> bool:
        """Return True if this spec uses extraction to create multiple starters."""
        return self.extraction_fn is not None

    def is_prefill(self) -> bool:
        """Return True if this spec uses prefill mode (bypasses Model B generation)."""
        return self.prefill_with_article


@dataclass
class SelfStudyConfig:
    """Self-study query generation: a list of conversation specs."""
    conversation_specs: List[ConversationSpec] = field(default_factory=list)

    def __post_init__(self):
        if not self.conversation_specs:
            raise ValueError("conversation_specs cannot be empty")


@dataclass
class ContextPrefillConfig:
    """
    Context-prefill query generation: queries from one prefill pass over the article.

    Parameters
    ----------
    last_n : int
        If > 0, keep only the queries of the last last_n article positions
        (an observation window as in SnapKV). 0 keeps all positions.
    """
    last_n: int = 0


@dataclass
class QueryMethodConfig:
    """
    One query generation method and its share of the query budget.

    Parameters
    ----------
    method : {'self_study', 'context_prefill'}
    fraction : float
        Share of the queries produced by this method.
    config : SelfStudyConfig or ContextPrefillConfig
    """
    method: str
    fraction: float
    config: Any

    def __post_init__(self):
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(
                f"fraction must be between 0.0 and 1.0, got {self.fraction}"
            )


@dataclass
class QueryConfig:
    """
    Query generation setup: a mix of methods whose fractions sum to 1.

    Parameters
    ----------
    method_configs : List[QueryMethodConfig]
    max_query_vectors_per_kv_head : int
        Total query budget per KV head, split across methods by fraction.
    """
    method_configs: List[QueryMethodConfig] = field(default_factory=list)
    max_query_vectors_per_kv_head: int = 10000
    verbose: bool = False

    def __post_init__(self):
        if self.max_query_vectors_per_kv_head < 1:
            raise ValueError(
                f"max_query_vectors_per_kv_head must be at least 1, "
                f"got {self.max_query_vectors_per_kv_head}"
            )

        if not self.method_configs:
            raise ValueError("method_configs cannot be empty")

        total_fraction = sum(mc.fraction for mc in self.method_configs)
        if not (0.99 <= total_fraction <= 1.01):
            raise ValueError(
                f"method_configs fractions should sum to 1.0, got {total_fraction}"
            )

        methods = [mc.method for mc in self.method_configs]
        if len(methods) != len(set(methods)):
            raise ValueError(f"Duplicate methods found in method_configs: {methods}")

    def get_method_config(self, method: str) -> Optional[QueryMethodConfig]:
        """Get configuration for a specific method."""
        for mc in self.method_configs:
            if mc.method == method:
                return mc
        return None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QueryConfig':
        """Rebuild a QueryConfig from its asdict() form."""
        method_configs = []
        for mc_dict in config_dict['method_configs']:
            method = mc_dict['method']
            fraction = mc_dict['fraction']

            if method == 'self_study':
                config_data = mc_dict['config'].copy()

                if 'conversation_specs' in config_data and config_data['conversation_specs']:
                    conversation_specs = []
                    for spec_dict in config_data['conversation_specs']:
                        spec_dict_copy = spec_dict.copy()

                        # extraction_fn is not serializable and is dropped.
                        if 'extraction_fn' in spec_dict_copy:
                            del spec_dict_copy['extraction_fn']

                        conversation_specs.append(ConversationSpec(**spec_dict_copy))
                    config_data['conversation_specs'] = conversation_specs

                method_config = SelfStudyConfig(**config_data)
            elif method == 'context_prefill':
                method_config = ContextPrefillConfig(**mc_dict['config'])
            else:
                method_config = mc_dict['config']

            method_configs.append(QueryMethodConfig(
                method=method,
                fraction=fraction,
                config=method_config
            ))

        return cls(
            method_configs=method_configs,
            max_query_vectors_per_kv_head=config_dict['max_query_vectors_per_kv_head'],
            verbose=config_dict.get('verbose', False)
        )
