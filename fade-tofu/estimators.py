"""
KL estimators between two autoregressive models on sequences sampled from the first.

For an answer y = (y_1, ..., y_N) sampled from p(. | x), with N including the
terminal (EOS) token when the answer ended on one:

    Monte Carlo (MC):       sum_n  log p(y_n | x, y_<n) - log q(y_n | x, y_<n)
    Rao-Blackwellized (RB): sum_n  KL( p(. | x, y_<n) || q(. | x, y_<n) )

Both are unbiased for KL(p(.|x) || q(.|x)). RB replaces each position's single sampled
log-ratio with its exact expectation over the full vocabulary, which can only lower the
variance, and since every term is itself a KL it is non-negative by construction
(Amini, Vieira & Cotterell, "Better Estimation of the KL Divergence Between Language
Models", arXiv:2504.10637, Eq. 9 / Theorem 2). MC comes out of the same forward
passes, so both are computed.
"""

import torch


@torch.no_grad()
def kl_terms_from_logits(logits_p, logits_q, targets, chunk_size=1024):
    """
    Per-position RB and MC terms for a flat set of positions.

    logits_p, logits_q: [N, V] next-token logits of p and q at the same N prefixes.
    targets:            [N] the token p actually sampled at each of those positions.
    Returns (rb, mc), each [N] float32. rb is clamped at 0 to absorb float rounding.
    """
    rb_out, mc_out = [], []
    for start in range(0, logits_p.shape[0], chunk_size):
        lp = logits_p[start:start + chunk_size].float().log_softmax(-1)
        lq = logits_q[start:start + chunk_size].float().log_softmax(-1)
        tgt = targets[start:start + chunk_size, None]
        rb_out.append((lp.exp() * (lp - lq)).sum(-1).clamp_min(0.0))
        mc_out.append((lp.gather(-1, tgt) - lq.gather(-1, tgt)).squeeze(-1))
    return torch.cat(rb_out), torch.cat(mc_out)


def _answer_logits(model, input_ids, attention_mask, positions):
    """Logits only at `positions` (a [N, 2] (row, col) index), skipping the full
    [B, T, V] logits tensor -- the vocab is 128k for Llama 3."""
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return model.lm_head(hidden[positions[:, 0], positions[:, 1]])


@torch.no_grad()
def sequence_kl(model_p, model_q, prompts, answers, pad_token_id, max_tokens_per_batch=16384):
    """
    RB and MC estimates of KL(p || q) for each (prompt, answer) pair, answers ~ p.

    prompts, answers: lists of token-id lists (answer includes its terminal token, if any).
    Returns (rb, mc): float lists, one value per pair (summed over answer positions).
    """
    device = next(model_p.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]) + len(answers[i]))
    rb_all = [0.0] * len(prompts)
    mc_all = [0.0] * len(prompts)

    batch = []
    def flush():
        seqs = [prompts[i] + answers[i] for i in batch]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), width), dtype=torch.long)
        pos, tgt, owner = [], [], []
        for row, (i, s) in enumerate(zip(batch, seqs)):
            input_ids[row, :len(s)] = torch.tensor(s)
            attention_mask[row, :len(s)] = 1
            start = len(prompts[i])
            for col in range(start, len(s)):  # logits at col-1 predict token at col
                pos.append((row, col - 1))
                tgt.append(s[col])
                owner.append(i)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        if pos:
            pos_t = torch.tensor(pos, device=device)
            logits_p = _answer_logits(model_p, input_ids, attention_mask, pos_t)
            logits_q = _answer_logits(model_q, input_ids, attention_mask, pos_t)
            rb, mc = kl_terms_from_logits(logits_p, logits_q, torch.tensor(tgt, device=device))
            for i, r, m in zip(owner, rb.tolist(), mc.tolist()):
                rb_all[i] += r
                mc_all[i] += m
        batch.clear()

    for i in order:
        width = len(prompts[i]) + len(answers[i])
        if batch and width * (len(batch) + 1) > max_tokens_per_batch:
            flush()
        batch.append(i)
    if batch:
        flush()
    return rb_all, mc_all
