"""
HF Trainers for fine-tuning and for the 5 unlearning objectives used in the paper
(see models.METHODS for the short names):

    grad_ascent      -forget NLL
    grad_diff        -forget NLL + retain NLL
    npo_grad_diff    NPO on forget (reference = the starting model) + retain NLL
    idk              NLL of refusals to forget questions + retain NLL
    grad_diff_hinge  inverted hinge loss on forget + retain NLL  (IHL, Cha et al. 2025)
"""

import torch
import torch.nn.functional as F
from transformers import Trainer


def get_batch_loss(logits, labels):
    """Summed next-token NLL per sequence, ignoring -100 labels."""
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss = F.cross_entropy(logits.transpose(-1, -2), shifted_labels, ignore_index=-100,
                           reduction="none")
    return loss.sum(dim=-1)


def inverted_hinge_loss(logits, labels):
    """mean over answer tokens of max(0, 1 + p(y) - max_{v != y} p(v))."""
    shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[..., 1:].reshape(-1)
    keep = shift_labels != -100
    probs = shift_logits[keep].softmax(-1)
    target = shift_labels[keep]
    p_target = probs.gather(-1, target[:, None]).squeeze(-1)
    p_other = probs.scatter(-1, target[:, None], float("-inf")).max(-1).values
    return (1.0 + p_target - p_other).clamp_min(0).mean()


class FinetuneTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        return outputs.loss, outputs.logits, labels


class ForgetTrainer(FinetuneTrainer):
    LOSSES = ("grad_ascent", "grad_diff", "npo_grad_diff", "idk", "grad_diff_hinge")

    def __init__(self, *args, forget_loss, beta=0.1, npo_coeff=1.0, grad_diff_coeff=1.0, **kwargs):
        if forget_loss not in self.LOSSES:
            raise ValueError(f"unknown forget_loss {forget_loss!r}; expected one of {self.LOSSES}")
        super().__init__(*args, **kwargs)
        self.forget_loss = forget_loss
        self.beta = beta
        self.npo_coeff = npo_coeff
        self.grad_diff_coeff = grad_diff_coeff

    def _reference_forget_nll(self, model, input_ids, labels, attention_mask):
        """NPO's reference policy is the model unlearning started from. The trainable
        weights are a LoRA adapter on top of it, so disabling the adapter recovers it
        exactly without keeping a second copy of the model in memory."""
        with torch.no_grad(), model.disable_adapter():
            logits = model(input_ids, attention_mask=attention_mask).logits
        return get_batch_loss(logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        (f_ids, f_labels, f_mask), (r_ids, r_labels, r_mask) = inputs
        loss_type = self.forget_loss

        if loss_type == "idk":
            # refusal answers to forget questions and retain QA are both plain SFT targets
            outputs = model(torch.cat([f_ids, r_ids]), labels=torch.cat([f_labels, r_labels]),
                            attention_mask=torch.cat([f_mask, r_mask]))
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        outputs = model(f_ids, labels=f_labels, attention_mask=f_mask)
        if loss_type in ("grad_ascent", "grad_diff"):
            forget_loss = -outputs.loss
        elif loss_type == "grad_diff_hinge":
            forget_loss = inverted_hinge_loss(outputs.logits, f_labels)
        else:  # npo_grad_diff
            neg_log_ratios = (get_batch_loss(outputs.logits, f_labels)
                              - self._reference_forget_nll(model, f_ids, f_labels, f_mask))
            forget_loss = self.npo_coeff * (-F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta)

        if loss_type == "grad_ascent":
            loss = forget_loss
        else:
            retain_loss = model(r_ids, labels=r_labels, attention_mask=r_mask).loss
            coeff = self.grad_diff_coeff if loss_type == "npo_grad_diff" else 1.0
            loss = forget_loss + coeff * retain_loss
        return (loss, outputs) if return_outputs else loss
