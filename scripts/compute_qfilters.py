#!/usr/bin/env python3
"""
Compute per-(layer, KV head) Q-Filters (Godey et al. 2025, arXiv:2503.02812).

Each filter is the sign-fixed top right singular vector of post-RoPE queries on Pile
calibration text (authors' protocol: 20 x 2048 tokens, 3000 sampled queries per head,
GQA groups averaged). We store v1+ and qfilters.py keeps the highest <k, v1+>. The
authors' released tensors store -v1+; --from-hf imports and negates them.

Output: qfilters/pile/<model basename>.pt (or --out).

Usage:
    python scripts/compute_qfilters.py --model Qwen/Qwen3-4B
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.utils import load_model_and_tokenizer          # noqa: E402
from compaction.query_generation.context_prefill import (      # noqa: E402
    ContextPrefillQueryGenerator)
from compaction.query_generation.config import (               # noqa: E402
    ContextPrefillConfig)


def import_official(repo, out, model_name):
    """Import an authors' released filter file, negating into our convention."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    path = snapshot_download(repo)
    sd = load_file(os.path.join(path, 'model.safetensors'))
    key = 'q_filters' if 'q_filters' in sd else list(sd)[0]
    # Released tensors store -v1+ (make_filters.py negates); flip to +v1+.
    filters = -sd[key].float()
    L, H, d = filters.shape
    print(f"imported {repo}: {(L, H, d)}, negated into the keep-highest "
          f"convention")
    torch.save({
        'filters': filters.contiguous(), 'model_name': model_name,
        'num_layers': L, 'num_kv_heads': H, 'head_dim': d,
        'calibration': f'official:{repo}', 'gqa': 'average',
        'n_tokens': None,
    }, out)
    print(f"wrote {out}")


def pile_sequences(tokenizer, n_samples, sample_len, dataset):
    """n_samples token sequences of exactly sample_len tokens from the Pile."""
    from datasets import load_dataset as hf_load
    ds = hf_load(dataset, split='train')
    seqs, i = [], 0
    while len(seqs) < n_samples and i < len(ds):
        ids = tokenizer(ds[i]['text'], return_tensors='pt',
                        add_special_tokens=True).input_ids
        if ids.shape[1] >= sample_len:
            seqs.append(ids[:, :sample_len])
        i += 1
    if len(seqs) < n_samples:
        raise SystemExit(f"only {len(seqs)} documents of >= {sample_len} "
                         f"tokens in the first {i} rows of {dataset}")
    print(f"pile: {len(seqs)} sequences x {sample_len} tokens "
          f"(scanned {i} documents)")
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default=None,
                    help='default: qfilters/pile/<model basename>.pt')
    ap.add_argument('--from-hf', default=None,
                    help="import an authors' released filter repo instead")
    # Calibration (defaults follow the paper).
    ap.add_argument('--pile-dataset', default='NeelNanda/pile-10k')
    ap.add_argument('--n-samples', type=int, default=20)
    ap.add_argument('--sample-len', type=int, default=2048)
    ap.add_argument('--svd-samples', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out = args.out or os.path.join(
        'qfilters', 'pile', args.model.rstrip('/').split('/')[-1] + '.pt')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    if args.from_hf:
        return import_official(args.from_hf, out, args.model)

    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model,
                                                dtype=torch.bfloat16)
    model.eval()
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, 'num_key_value_heads', num_heads)
    head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // num_heads)
    hpk = num_heads // num_kv_heads
    print(f"{num_layers} layers, {num_heads} q-heads -> {num_kv_heads} KV "
          f"heads (group {hpk}), head_dim {head_dim}")

    gen = ContextPrefillQueryGenerator(
        model, tokenizer, ContextPrefillConfig(), dtype=torch.bfloat16)

    seqs = pile_sequences(tokenizer, args.n_samples, args.sample_len,
                          args.pile_dataset)
    per_seq = -(-args.svd_samples // len(seqs))   # ceil

    buf = []
    n_tokens = 0
    for i, ids in enumerate(seqs):
        q = gen._extract_query_vectors_from_prefill(
            input_ids=ids.to(model.device), head_dim=head_dim)
        if q is None:
            raise RuntimeError(f"query extraction failed on sample {i}")
        # q: (num_layers, num_attention_heads, n_tok, head_dim)
        n_tok = q.shape[2]
        k = min(per_seq, n_tok)
        sel = torch.randperm(n_tok, device=q.device)[:k]
        buf.append(q[:, :, sel, :].to(torch.float32).cpu())   # (L, H_q, k, d)
        n_tokens += n_tok
        del q
        torch.cuda.empty_cache()
        print(f"sample {i}: {n_tok} tokens ({n_tokens} total)", flush=True)

    X = torch.cat(buf, dim=2)[:, :, :args.svd_samples, :]       # (L,H_q,n,d)
    n_used = X.shape[2]
    print(f"SVD on {n_used} representations per head")
    # Per q-head SVD; sign by the majority sign of u1 (reference impl).
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    v1 = Vh[..., 0, :]                                          # (L, H_q, d)
    u1 = U[..., :, 0]                                           # (L, H_q, n)
    sign = ((u1 > 0).float().mean(-1) > 0.5).float() * 2 - 1
    v1 = sign.unsqueeze(-1) * v1                                # = v1+
    share = (S[..., 0] ** 2 / (S ** 2).sum(-1)).mean()
    # GQA: average the per-q-head filters within each group.
    filters = v1.reshape(num_layers, num_kv_heads, hpk, head_dim).mean(2)
    print(f"top-direction energy share: mean {share:.3f}")
    print(f"filter norms: mean {filters.norm(dim=-1).mean():.4f} "
          f"(< 1 under GQA averaging = within-group agreement)")

    torch.save({
        'filters': filters.contiguous(),
        'model_name': args.model,
        'num_layers': num_layers, 'num_kv_heads': num_kv_heads,
        'head_dim': head_dim, 'n_tokens': n_tokens,
        'calibration': 'pile', 'gqa': 'average',
        'svd_samples': n_used,
        'articles': f"{args.pile_dataset}[{args.n_samples}x{args.sample_len}]",
        'energy_share_mean': float(share),
    }, out)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
