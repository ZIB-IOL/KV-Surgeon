# head_budget_optimization/
"""
Nonuniform per-head budgets for KV cache compaction.

Measures per-head influence curves (perplexity vs. one head's budget), averages them over
articles, and solves for one set of head proportions. See run.py for the recipe.
"""

from .influence import (
    HeadInfluenceComputer,
    aggregate_head_curves,
    save_head_curves,
    load_head_curves,
    load_and_aggregate_article_curves,
)
from .solver import HeadBudgetSolver, analyze_head_curves

__all__ = [
    'HeadInfluenceComputer',
    'HeadBudgetSolver',
    'aggregate_head_curves',
    'save_head_curves',
    'load_head_curves',
    'load_and_aggregate_article_curves',
    'analyze_head_curves',
]
