"""HIP top-1 reference sampling; does not own KV or communication buffers.

The exact-history path is intentionally a correctness reference: rebuilding
counts on CPU costs more than incremental bookkeeping but avoids silently
reusing an incomplete overlap history. Only forward-local penalty snapshots
are changed; the scheduler's request output and penalizers are read-only.
"""
import logging

import torch

logger = logging.getLogger(__name__)
_logged_sampling = False
_logged_history = False


def sample_target_ids(logits, temperatures, top_ks, top_ps, min_ps=None,
                      sampling_seed=None, positions=None):
    global _logged_sampling
    if not _logged_sampling:
        logger.warning('HIP MTP target sampling: Torch FP32 temperature/top-k/top-p; draft-match acceptance')
        _logged_sampling = True
    x = logits.float()
    temperatures = temperatures.reshape(-1, 1)
    top_ks = top_ks.reshape(-1, 1)
    top_ps = top_ps.reshape(-1, 1)
    greedy = (temperatures <= 0) | (top_ks == 1)
    scaled = x / torch.where(temperatures > 0, temperatures, 1.0)
    sorted_logits, ids = scaled.sort(dim=-1, descending=True)
    ranks = torch.arange(x.shape[-1], device=x.device).reshape(1, -1)
    sorted_logits = sorted_logits.masked_fill(
        ~((top_ks <= 0) | (ranks < top_ks)), float('-inf'))
    probs = torch.softmax(sorted_logits, dim=-1)
    keep = ((probs.cumsum(-1)-probs) < top_ps) | (ranks == 0)
    if min_ps is not None:
        keep &= (probs >= probs[:, :1]*min_ps.reshape(-1, 1)) | (ranks == 0)
    probs = probs.masked_fill(~keep, 0.)
    if sampling_seed is None:
        selected = torch.multinomial(probs, 1)
    else:
        from sglang.srt.sampling.sampler import multinomial_with_seed
        if positions is None:
            raise ValueError('Seeded target sampling requires token positions')
        selected = multinomial_with_seed(probs.double().log(), sampling_seed, positions)
    sampled = ids.gather(1, selected).squeeze(1)
    return torch.where(greedy.squeeze(1), x.argmax(-1), sampled)


def pending_output_by_rid(pending_results, current_rids):
    """Read previous D2H results without committing them or advancing grammar."""
    pending = {}
    for previous_batch, result in pending_results:
        if not any(req.rid in current_rids for req in previous_batch.reqs):
            continue
        if result.copy_done is not None:
            result.copy_done.synchronize()
        ids = result.next_token_ids
        if ids is None:
            continue
        if torch.is_tensor(ids):
            if not ids.is_cpu:
                raise RuntimeError('Pending MTP output must be copied to CPU before penalty snapshot')
            ids = ids.reshape(-1).tolist()
        if result.accept_lens is not None:
            if not result.accept_lens.is_cpu:
                raise RuntimeError('Pending accept lengths must be on CPU')
            lengths = result.accept_lens.tolist()
            stride = result.speculative_num_draft_tokens
        else:
            lengths = [1]*len(previous_batch.reqs)
            stride = 1
        if stride is None or len(ids) != len(previous_batch.reqs)*stride:
            raise RuntimeError('Unexpected pending MTP result layout')
        for i, req in enumerate(previous_batch.reqs):
            if req.rid not in current_rids:
                continue
            if req.rid in pending:
                raise RuntimeError('More than one uncommitted MTP result for the same request')
            n = lengths[i]
            if not 0 <= n <= stride:
                raise RuntimeError('Invalid pending MTP accepted length')
            pending[req.rid] = ids[i*stride:i*stride+n]
    return pending


def refresh_exact_penalty_snapshot(batch, orchestrator, pending_results):
    """Complete committed+pending frequency/presence penalties in a fresh copy.

    Called AFTER copy_for_forward and BEFORE the worker starts. Only the two
    supported penalty contributions are replaced; normal decoding is untouched.
    """
    global _logged_history
    reqs = batch.reqs
    if not reqs or not any(r.sampling_params.frequency_penalty != 0 or
                           r.sampling_params.presence_penalty != 0 for r in reqs):
        return
    if any(getattr(r.sampling_params, 'repetition_penalty', 1.) != 1. or
           getattr(r.sampling_params, 'min_new_tokens', 0) != 0 for r in reqs):
        raise ValueError('HIP exact MTP reference currently supports frequency/presence penalties; '
                         'repetition_penalty must be 1 and min_new_tokens must be 0')
    from sglang.srt.sampling.penaltylib import BatchedFrequencyPenalizer, BatchedPresencePenalizer
    info = batch.sampling_info
    pending = pending_output_by_rid(pending_results, {r.rid for r in reqs})
    counts = torch.zeros((len(reqs), info.vocab_size), dtype=torch.float32)
    for i, req in enumerate(reqs):
        history = list(req.output_ids) + pending.get(req.rid, [])
        if history:
            ids = torch.tensor(history, dtype=torch.int64)
            if ids.min() < 0 or ids.max() >= info.vocab_size:
                raise RuntimeError('Output history contains an invalid token')
            counts[i] = torch.bincount(ids, minlength=info.vocab_size)
    counts = counts.to(device=info.temperatures.device)
    frequency = torch.tensor([r.sampling_params.frequency_penalty for r in reqs],
                             dtype=torch.float32, device=counts.device).unsqueeze(1)
    presence = torch.tensor([r.sampling_params.presence_penalty for r in reqs],
                            dtype=torch.float32, device=counts.device).unsqueeze(1)
    additive = (info.acc_additive_penalties.clone() if info.acc_additive_penalties is not None
                else torch.zeros_like(counts))
    if orchestrator is not None:
        # Remove these two relaxed contributions, retaining other additive terms.
        for cls in (BatchedFrequencyPenalizer, BatchedPresencePenalizer):
            penalizer = orchestrator.penalizers.get(cls)
            if penalizer is not None and penalizer.is_prepared():
                contribution = torch.zeros_like(counts)
                penalizer.apply(contribution)
                additive.sub_(contribution)
    additive.sub_(frequency*counts + presence*counts.gt(0))
    info.acc_additive_penalties = additive
    info.speculative_frequency_penalties = frequency
    info.speculative_presence_penalties = presence
    info.speculative_presence_mask = counts.gt(0)
    if not _logged_history:
        logger.warning('HIP MTP penalty reference: complete committed + pending accepted history; '
                       'forward-local snapshot; top-1 prefix penalties enabled')
        _logged_history = True


def apply_top1_prefix_penalties(next_token_logits, candidates, sampling_info, draft_token_num):
    frequency = sampling_info.speculative_frequency_penalties
    presence = sampling_info.speculative_presence_penalties
    if frequency is None and presence is None:
        return
    logits = next_token_logits.reshape(candidates.shape[0], draft_token_num, -1)
    tokens = candidates[:, 1:].long()
    if tokens.numel() == 0:
        return
    delta = -frequency.expand_as(tokens)
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    earlier = positions.reshape(1, 1, -1) < positions.reshape(1, -1, 1)
    appeared = (tokens.unsqueeze(2).eq(tokens.unsqueeze(1)) & earlier).any(2)
    seen = sampling_info.speculative_presence_mask.gather(1, tokens)
    delta = delta - presence*(~(appeared | seen))
    for pos in range(1, draft_token_num):
        logits[:, pos].scatter_add_(1, tokens[:, :pos], delta[:, :pos].to(logits.dtype))
