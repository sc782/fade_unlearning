"""
Checks estimators.py against the exact KL of two toy bigram language models, small
enough that every possible answer can be enumerated.

    python -m pytest tests/test_estimators.py
"""

import itertools
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from estimators import kl_terms_from_logits, sequence_kl  # noqa: E402

VOCAB, EOS, MAX_NEW = 4, 3, 4  # tokens 0..3, token 3 ends the answer


class Bigram(torch.nn.Module):
    """next-token logits depend only on the current token; exposes the .model /
    .lm_head structure sequence_kl expects from an HF causal LM."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.table = torch.randn(VOCAB, VOCAB, generator=g) * 1.5
        self.lm_head = lambda h: h
        self.model = lambda input_ids, attention_mask: SimpleNamespace(
            last_hidden_state=self.table[input_ids])
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # gives the module a device

    def next_logprobs(self, token):
        return self.table[token].log_softmax(-1)


def answers_with_prob(model, prompt_last):
    """Every possible answer (stopping at EOS or MAX_NEW tokens) with its probability."""
    out = []
    for n in range(1, MAX_NEW + 1):
        for seq in itertools.product(range(VOCAB), repeat=n):
            if EOS in seq[:-1] or (n < MAX_NEW and seq[-1] != EOS):
                continue
            logp, prev = 0.0, prompt_last
            for tok in seq:
                logp += model.next_logprobs(prev)[tok].item()
                prev = tok
            out.append((list(seq), logp))
    return out


def exact_kl(p, q, prompt_last):
    lq = dict((tuple(s), lp) for s, lp in answers_with_prob(q, prompt_last))
    return sum(torch.tensor(lp).exp().item() * (lp - lq[tuple(s)])
               for s, lp in answers_with_prob(p, prompt_last))


def sample(model, prompt_last, n, g):
    answers = []
    for _ in range(n):
        seq, prev = [], prompt_last
        while len(seq) < MAX_NEW:
            prev = torch.multinomial(model.next_logprobs(prev).exp(), 1, generator=g).item()
            seq.append(prev)
            if prev == EOS:
                break
        answers.append(seq)
    return answers


def test_per_position_terms():
    torch.manual_seed(0)
    lp, lq = torch.randn(5, 7), torch.randn(5, 7)
    rb, mc = kl_terms_from_logits(lp, lq, torch.arange(5), chunk_size=2)
    expected = torch.nn.functional.kl_div(lq.log_softmax(-1), lp.log_softmax(-1),
                                          log_target=True, reduction="none").sum(-1)
    assert torch.allclose(rb, expected, atol=1e-6)
    assert (rb >= 0).all()
    ref_mc = lp.log_softmax(-1)[range(5), range(5)] - lq.log_softmax(-1)[range(5), range(5)]
    assert torch.allclose(mc, ref_mc, atol=1e-6)


def test_unbiased_and_lower_variance():
    p, q = Bigram(1), Bigram(2)
    prompt = [0, 1, 2]
    truth = exact_kl(p, q, prompt[-1])
    answers = sample(p, prompt[-1], 20000, torch.Generator().manual_seed(0))
    rb, mc = sequence_kl(p, q, [prompt] * len(answers), answers, pad_token_id=0,
                         max_tokens_per_batch=4096)
    rb, mc = torch.tensor(rb), torch.tensor(mc)
    se = mc.std().item() / len(answers) ** 0.5
    assert abs(rb.mean().item() - truth) < 4 * se, (rb.mean(), truth)
    assert abs(mc.mean().item() - truth) < 4 * se, (mc.mean(), truth)
    assert rb.var() <= mc.var()
    assert (rb >= 0).all()


def test_batching_does_not_change_values():
    p, q = Bigram(3), Bigram(4)
    answers = sample(p, 2, 50, torch.Generator().manual_seed(1))
    prompts = [[1, 2], [0, 0, 2]] * 25  # mixed prompt lengths -> padding inside batches
    one = sequence_kl(p, q, prompts, answers, 0, max_tokens_per_batch=1)
    many = sequence_kl(p, q, prompts, answers, 0, max_tokens_per_batch=10 ** 6)
    assert torch.allclose(torch.tensor(one), torch.tensor(many), atol=1e-5)
