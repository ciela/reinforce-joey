# coding: utf-8
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from joeynmt.decoders import TransformerDecoder
from joeynmt.model import Model
from joeynmt.batch import Batch
from joeynmt.helpers import tile
from logzero import logger as log

__all__ = [
    "greedy",
    "transformer_greedy",
    "beam_search",
    "fcfs_beam_search",
    "vanilla_beam_search",
    "run_batch",
]


def greedy(src_mask: Tensor, max_output_length: int, model: Model,
           encoder_output: Tensor, encoder_hidden: Tensor)\
        -> (np.array, np.array):
    """
    Greedy decoding. Select the token word highest probability at each time
    step. This function is a wrapper that calls recurrent_greedy for
    recurrent decoders and transformer_greedy for transformer decoders.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :return:
    """

    if isinstance(model.decoder, TransformerDecoder):
        # Transformer greedy decoding
        greedy_fun = transformer_greedy
    else:
        # Recurrent greedy decoding
        greedy_fun = recurrent_greedy

    return greedy_fun(
        src_mask, max_output_length, model, encoder_output, encoder_hidden)


def recurrent_greedy(
        src_mask: Tensor, max_output_length: int, model: Model,
        encoder_output: Tensor, encoder_hidden: Tensor) -> (np.array, np.array):
    """
    Greedy decoding: in each step, choose the word that gets highest score.
    Version for recurrent decoder.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    bos_index = model.bos_index
    eos_index = model.eos_index
    batch_size = src_mask.size(0)
    prev_y = src_mask.new_full(size=[batch_size, 1], fill_value=bos_index,
                               dtype=torch.long)
    output = []
    attention_scores = []
    hidden = None
    prev_att_vector = None
    finished = src_mask.new_zeros((batch_size, 1)).byte()

    # pylint: disable=unused-variable
    for t in range(max_output_length):
        # decode one single step
        with torch.no_grad():
            logits, hidden, att_probs, prev_att_vector = model(
                return_type="decode",
                trg_input=prev_y,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                decoder_hidden=hidden,
                att_vector=prev_att_vector)
            # logits: batch x time=1 x vocab (logits)

        # greedy decoding: choose arg max over vocabulary in each step
        next_word = torch.argmax(logits, dim=-1)  # batch x time=1
        output.append(next_word.squeeze(1).detach().cpu().numpy())
        prev_y = next_word
        attention_scores.append(att_probs.squeeze(1).detach().cpu().numpy())
        # batch, max_src_length

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        # stop predicting if <eos> reached for all elements in batch
        if (finished >= 1).sum() == batch_size:
            break

    stacked_output = np.stack(output, axis=1)  # batch, time
    stacked_attention_scores = np.stack(attention_scores, axis=1)
    return stacked_output, stacked_attention_scores


# pylint: disable=unused-argument
def transformer_greedy(
        src_mask: Tensor, max_output_length: int, model: Model,
        encoder_output: Tensor, encoder_hidden: Tensor) -> (np.array, None):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder final state (unused in Transformer)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    bos_index = model.bos_index
    eos_index = model.eos_index
    batch_size = src_mask.size(0)

    # start with BOS-symbol for each sentence in the batch
    ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones([1, 1, 1])
    if isinstance(model, torch.nn.DataParallel):
        trg_mask = torch.stack(
            [src_mask.new_ones([1, 1]) for _ in model.device_ids])

    finished = src_mask.new_zeros(batch_size).byte()

    for _ in range(max_output_length):
        # pylint: disable=unused-variable
        with torch.no_grad():
            logits, _, _, _ = model(
                return_type="decode",
                trg_input=ys, # model.trg_embed(ys) # embed the previous tokens
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                decoder_hidden=None,
                trg_mask=trg_mask
            )

            logits = logits[:, -1]
            _, next_word = torch.max(logits, dim=1)
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        # stop predicting if <eos> reached for all elements in batch
        if (finished >= 1).sum() == batch_size:
            break

    ys = ys[:, 1:]  # remove BOS-symbol
    return ys.detach().cpu().numpy(), None


# pylint: disable=too-many-statements,too-many-branches
def beam_search(model: Model, size: int,
                encoder_output: Tensor, encoder_hidden: Tensor,
                src_mask: Tensor, max_output_length: int,
                alpha: float, n_best: int = 1) -> (np.array, np.array):
    """
    Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.

    In each decoding step, find the k most likely partial hypotheses.

    :param model:
    :param size: size of the beam
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    assert size > 0, 'Beam size must be >0.'
    assert n_best <= size, 'Can only return {} best hypotheses.'.format(size)

    # init
    bos_index = model.bos_index
    eos_index = model.eos_index
    pad_index = model.pad_index
    trg_vocab_size = model.decoder.output_size
    device = encoder_output.device
    transformer = isinstance(model.decoder, TransformerDecoder)
    batch_size = src_mask.size(0)
    att_vectors = None  # not used for Transformer

    # Recurrent models only: initialize RNN hidden state
    # pylint: disable=protected-access
    if not transformer:
        hidden = model.decoder._init_hidden(encoder_hidden)
    else:
        hidden = None

    # tile encoder states and decoder initial states beam_size times
    if hidden is not None:
        hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size

    encoder_output = tile(encoder_output.contiguous(), size,
                          dim=0)  # batch*k x src_len x enc_hidden_size
    src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

    # Transformer only: create target mask
    if transformer:
        trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
        if isinstance(model, torch.nn.DataParallel):
            trg_mask = torch.stack(
                [src_mask.new_ones([1, 1]) for _ in model.device_ids])
    else:
        trg_mask = None

    # numbering elements in the batch
    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

    # numbering elements in the extended batch, i.e. beam size copies of each
    # batch element
    beam_offset = torch.arange(
        0,
        batch_size * size,
        step=size,
        dtype=torch.long,
        device=device)

    # keeps track of the top beam size hypotheses to expand for each element
    # in the batch to be further decoded (that are still "alive")
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=device)

    # Give full probability to the first beam on the first step.
    topk_log_probs = torch.zeros(batch_size, size, device=device)
    topk_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):
        # This decides which part of the predicted sentence we feed to the
        # decoder to make the next prediction.
        # For Transformer, we feed the complete predicted sentence so far.
        # For Recurrent models, only feed the previous target word prediction
        if transformer:  # Transformer
            decoder_input = alive_seq  # complete prediction so far
        else:  # Recurrent
            decoder_input = alive_seq[:, -1].view(-1, 1)  # only the last word

        # expand current hypotheses
        # decode one single step
        # logits: logits for final softmax
        # pylint: disable=unused-variable
        with torch.no_grad():
            logits, hidden, att_scores, att_vectors = model(
                return_type="decode",
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                trg_input=decoder_input, #trg_embed = embed(decoder_input)
                decoder_hidden=hidden,
                att_vector=att_vectors,
                unroll_steps=1,
                trg_mask=trg_mask  # subsequent mask for Transformer only
            )

        # For the Transformer we made predictions for all time steps up to
        # this point, so we only want to know about the last time step.
        if transformer:
            logits = logits[:, -1]  # keep only the last time step
            hidden = None           # we don't need to keep it for transformer

        # batch*k x trg_vocab
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

        # multiply probs by the beam probability (=add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * trg_vocab_size)

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.floor_divide(trg_vocab_size)
        topk_ids = topk_ids.fmod(trg_vocab_size)

        # map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_index
            + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
        select_indices = batch_index.view(-1)

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
             topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(True)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    # Check if the prediction has more than one EOS.
                    # If it has more than one EOS, it means that the
                    # prediction should have already been added to
                    # the hypotheses, so you don't have to add them again.
                    if (predictions[i, j, 1:] == eos_index).nonzero(
                            as_tuple=False).numel() < 2:
                        # ignore start_token
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 1:])
                        )
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(False).nonzero(
                as_tuple=False).view(-1)
            # if all sentences are translated, no need to go further
            # pylint: disable=len-as-condition
            if len(non_finished) == 0:
                break
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished) \
                .view(-1, alive_seq.size(-1))

        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

        if hidden is not None and not transformer:
            if isinstance(hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                hidden = hidden.index_select(1, select_indices)

        if att_vectors is not None:
            att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                       pad_value=pad_index)

    return final_outputs, None


# pylint: disable=too-many-statements,too-many-branches
def fcfs_beam_search(model: Model, beam_size: int,
                encoder_output: Tensor, encoder_hidden: Tensor,
                src_mask: Tensor, max_output_length: int,
                alpha: float, n_best: int = 1) -> (np.array, np.array):
    """
    FCFS Beam search with size k.
    based on beam_search and [this paper](https://arxiv.org/abs/2204.05424),
    adapted for Transformer.

    In each decoding step, find the k most likely partial hypotheses.

    :param model:
    :param beam_size: size of the beam
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    assert beam_size > 0, 'Beam size must be >0.'
    assert n_best <= beam_size, f"Can only return {beam_size} best hypotheses."

    # init
    bos_index = model.bos_index
    eos_index = model.eos_index
    pad_index = model.pad_index
    trg_vocab_size = model.decoder.output_size
    device = encoder_output.device
    batch_size = src_mask.size(0)

    # Recurrent models only: initialize RNN hidden state
    # pylint: disable=protected-access
    encoder_output = tile(encoder_output.contiguous(), beam_size,
                          dim=0)  # (batch_size * beam_size, src_len, enc_hidden_size)
    src_mask = tile(src_mask, beam_size, dim=0)  # (batch_size * beam_size, 1, src_len)

    # Transformer only: create target mask
    trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
    if isinstance(model, torch.nn.DataParallel):
        trg_mask = torch.stack(
            [src_mask.new_ones([1, 1]) for _ in model.device_ids])

    # numbering elements in the batch
    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

    # numbering elements in the extended batch, i.e. beam_size copies of each batch element
    beam_offset = torch.arange(
        0,
        batch_size * beam_size,
        step=beam_size,
        dtype=torch.long,
        device=device)

    # keeps track of the top beam size hypotheses to expand for each element
    # in the batch to be further decoded (that are still "alive")
    alive_seq = torch.full(
        [batch_size * beam_size, 1],
        bos_index,
        dtype=torch.long,
        device=device)  # (batch_size * beam_size, hyp_len) ... now hyp_len = 1

    # Give full probability (=zero in log space) to the first beam on the first step,
    # since the only option of the first token is the BOS token.
    topk_log_probs = torch.zeros(batch_size, beam_size, device=device)  # (batch_size, beam_size)
    topk_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    # init remaining_batch_size
    remaining_batch_size = len(batch_offset)

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):
        # This decides which part of the predicted sentence we feed to the
        # decoder to make the next prediction.
        # For Transformer, we feed the complete predicted sentence so far.
        # For Recurrent models, only feed the previous target word prediction
        decoder_input = alive_seq  # complete prediction so far; (remaining_batch_size * beam_size, step+1)

        # expand current hypotheses
        # decode one single step
        with torch.no_grad():
            # logits: scores before final softmax, (remaining_batch_size * beam_size, step+1, trg_vocab_size)
            logits, _, _, _ = model(
                return_type="decode",
                encoder_output=encoder_output,
                encoder_hidden=None,  # only for initializing decoder_hidden
                src_mask=src_mask,
                trg_input=decoder_input,  # trg_embed = embed(decoder_input)
                decoder_hidden=None,  # don't need to keep it for transformer
                att_vector=None,   # don't need to keep it for transformer
                unroll_steps=1,
                trg_mask=trg_mask  # subsequent mask for Transformer only
            )

        # For the Transformer we made predictions for all time steps up to
        # this point, so we only want to know about the last time step.
        logits = logits[:, -1]  # (remaining_batch_size * beam_size, trg_vocab_size)

        # compute log probability over trg vocab given a previous beam sequence
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)  # (remaining_batch_size * beam_size, trg_vocab_size)

        # multiply the vocab probs by the previous beam sequence probability (= add log_probs)
        # 'topk_log_probs' shape: (remaining_batch_size, beam_size) -> (remaining_batch_size*beam_size, 1)
        log_probs += topk_log_probs.reshape(-1, 1)  # (remaining_batch_size * beam_size, trg_vocab_size)
        curr_scores = log_probs.clone()

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, beam_size * trg_vocab_size)  # (remaining_batch_size, beam_size, trg_vocab_size)

        # pick currently best top 2*k hypotheses (flattened order)
        # `topk_scores` and `topk_ids` shape: (remaining_batch_size, 2*beam_size)
        top2k_scores, top2k_ids = curr_scores.topk(2*beam_size, dim=-1)

        if alpha > -1:
            # recover original log probs
            top2k_log_probs = top2k_scores * length_penalty  # (remaining_batch_size, 2*beam_size)
        else:
            top2k_log_probs = top2k_scores.clone()  # (remaining_batch_size, 2*beam_size)

        # reconstruct beam origin and true vocab ids from flattened order
        top2k_beam_origin_ids = top2k_ids.floor_divide(trg_vocab_size)  # (remaining_batch_size, 2*beam_size)
        top2k_vocab_ids = top2k_ids.fmod(trg_vocab_size)  # (remaining_batch_size, 2*beam_size)

        # map beam_index to batch_index in the flat representation
        batch_index = (
            top2k_beam_origin_ids                               # (remaining_batch_size, 2*beam_size)
            + beam_offset[:remaining_batch_size].unsqueeze(1)   # (remaining_batch_size, 1)
        )  # (remaining_batch_size, 2*beam_size)
        select_ids = batch_index.view(-1)  # (remaining_batch_size * 2*beam_size)

        # generate topk_seqs by appending the latest prediction to alive_seq
        top2k_seqs = torch.cat([
            alive_seq.index_select(dim=0, index=select_ids),   # (remaining_batch_size * 2 * beam_size, step+1)
            top2k_vocab_ids.view(-1, 1)                        # (remaining_batch_size * 2 * beam_size, 1)
            ], -1).reshape(remaining_batch_size, -1, step+2)   # (remaining_batch_size, 2 * beam_size, step+2)

        # init `alive_seq` and `topk_batch_index` for the following processes
        alive_seq = torch.zeros(remaining_batch_size, beam_size, step+2, dtype=torch.long, device=device)
        topk_batch_index = torch.zeros([remaining_batch_size, beam_size], dtype=torch.long, device=device)

        # save finished hypotheses, renew `alive_seq` and `topk_log_probs`,
        # and check whether `end_condition` is True
        end_condition = torch.full([remaining_batch_size], fill_value=False, device=device)
        for b in range(remaining_batch_size):
            b_org = batch_offset[b]
            alive_seq_list = []
            log_prob_list = []
            batch_index_list = []
            is_list_full = False
            for seq, log_prob, score, b_index in zip(top2k_seqs[b], top2k_log_probs[b], top2k_scores[b], batch_index[b]):
                if (seq[-1]==eos_index).item() or (step+1==max_output_length):
                    hypotheses[b_org].append(
                        (score, seq[1:])  # ignore start_token
                    )
                    if len(hypotheses[b_org]) == beam_size:
                        end_condition[b] = True
                        break
                else:
                    alive_seq_list.append(seq)
                    log_prob_list.append(log_prob)
                    batch_index_list.append(b_index)
                    if len(alive_seq_list) == beam_size:
                        is_list_full = True
                        break

            # if the batch reached the end, save the n_best hypotheses
            if end_condition[b]:
                best_hyp = sorted(
                    hypotheses[b_org], key=lambda x: x[0], reverse=True)
                for n, (score, seq) in enumerate(best_hyp):
                    if n >= n_best:
                        break
                    results["scores"][b_org].append(score)
                    results["predictions"][b_org].append(seq)

            else:  # the batch did not reach the end
                # if `*_list` are not full, fill them up
                if not is_list_full:
                    for i in range(beam_size-len(alive_seq_list)):
                        alive_seq_list.append(alive_seq_list[-1])
                        log_prob_list.append(float("-inf"))
                        batch_index_list.append(batch_index_list[-1])
                # renew alive_seq and topk_log_probs
                alive_seq[b] = torch.stack(alive_seq_list)  # `alive_seq[b]` shape: (beam_size, hyp_len)
                topk_log_probs[b] = torch.stack(log_prob_list)  # `topk_log_probs[b]` shape: (beam_size)
                topk_batch_index[b] = torch.stack(batch_index_list)  # `topk_batch_index[b]` shape: (beam_size)

        # batch indices of the examples which contain unfinished candidates
        unfinished = end_condition.eq(False).nonzero(as_tuple=False).view(-1)
        # if all examples are translated, no need to go further
        if len(unfinished) == 0:
            break

        # remove finished examples for the next step
        topk_batch_index = topk_batch_index.index_select(0, unfinished)  # (remaining_batch_size, beam_size)
        topk_log_probs = topk_log_probs.index_select(0, unfinished)      # (remaining_batch_size, beam_size)
        batch_offset = batch_offset.index_select(0, unfinished)          # (remaining_batch_size)
        alive_seq = alive_seq.index_select(0, unfinished)                # (remaining_batch_size, beam_size, hyp_len)

        # update remaining_batch_size
        remaining_batch_size = len(batch_offset)

        # reshape `alive_seq` to its original size
        alive_seq = alive_seq.reshape(remaining_batch_size * beam_size, step+2)  # (remaining_batch_size*beam_size, hyp_len)

        # reorder indices, outputs and masks
        select_ids = topk_batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_ids)
        src_mask = src_mask.index_select(0, select_ids)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps(
        [r[0].cpu().numpy() for r in results["predictions"]],
        pad_value=pad_index)

    return final_outputs, None


# pylint: disable=too-many-statements,too-many-branches
def vanilla_beam_search(model: Model, beam_size: int,
                encoder_output: Tensor, encoder_hidden: Tensor,
                src_mask: Tensor, max_output_length: int,
                alpha: float, n_best: int = 1) -> (np.array, np.array):
    """
    Vanilla beam search with size k.
    based on beam_search and [this paper](https://arxiv.org/abs/2204.05424),
    adapted for Transformer.

    In each decoding step, find the k most likely partial hypotheses.

    :param model:
    :param beam_size: size of the beam
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    assert beam_size > 0, 'Beam size must be >0.'
    assert n_best <= beam_size, f'Can only return {beam_size} best hypotheses.'

    # init
    bos_index = model.bos_index
    eos_index = model.eos_index
    pad_index = model.pad_index
    trg_vocab_size = model.decoder.output_size
    device = encoder_output.device
    batch_size = src_mask.size(0)

    # Recurrent models only: initialize RNN hidden state
    # pylint: disable=protected-access
    encoder_output = tile(encoder_output.contiguous(), beam_size,
                          dim=0)  # (batch_size * beam_size, src_len, enc_hidden_size)
    src_mask = tile(src_mask, beam_size, dim=0)  # (batch_size * beam_size, 1, src_len)

    # Transformer only: create target mask
    trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
    if isinstance(model, torch.nn.DataParallel):
        trg_mask = torch.stack(
            [src_mask.new_ones([1, 1]) for _ in model.device_ids])

    # numbering elements in the batch
    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

    # numbering elements in the extended batch, i.e. beam size copies of each
    # batch element
    beam_offset = torch.arange(
        0,
        batch_size * beam_size,
        step=beam_size,
        dtype=torch.long,
        device=device)

    # keeps track of the top beam size hypotheses to expand for each element
    # in the batch to be further decoded (that are still "alive")
    alive_seq = torch.full(
        [batch_size * beam_size, 1],
        bos_index,
        dtype=torch.long,
        device=device)  # (batch_size * beam_size, hyp_len) ... now hyp_len = 1

    # Give full probability to the first beam on the first step; score := log prob.
    # since the only option of the first token is the BOS token.
    topk_scores = torch.zeros(batch_size, beam_size, device=device)  # (batch_size, beam_size)
    topk_scores[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    # init remaining_batch_size
    remaining_batch_size = len(batch_offset)

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    # indicator if the generation is finished
    # `is_finished` shape: (batch_size, beam_size)
    is_finished = torch.full((batch_size, beam_size),
                             False,
                             dtype=torch.bool,
                             device=device)

    for step in range(max_output_length):
        # This decides which part of the predicted sentence we feed to the
        # decoder to make the next prediction.
        # For Transformer, we feed the complete predicted sentence so far.
        # For Recurrent models, only feed the previous target word prediction
        decoder_input = alive_seq  # complete prediction so far; (remaining_batch_size * beam_size, step+1)

        # expand current hypotheses
        # decode one single step
        with torch.no_grad():
            # logits: scores before final softmax; (remaining_batch_size * beam_size, step+1, trg_vocab_size)
            logits, _, _, _ = model(
                return_type="decode",
                encoder_output=encoder_output,
                encoder_hidden=None,      # only for initializing decoder_hidden
                src_mask=src_mask,
                trg_input=decoder_input,  # trg_embed = embed(decoder_input)
                decoder_hidden=None,      # don't need to keep it for transformer
                att_vector=None,          # don't need to keep it for transformer
                unroll_steps=1,
                trg_mask=trg_mask         # subsequent mask for Transformer only
            )

        # For the Transformer we made predictions for all time steps up to
        # this point, so we only want to know about the last time step.
        logits = logits[:, -1]  # (remaining_batch_size * beam_size, trg_vocab_size)

        # compute log probability over trg vocab given a previous sequence
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)   # (remaining_batch_size * beam_size, trg_vocab_size)

        # compute length penalty
        if alpha > -1:
            if step == 0:
                length_penalty_prev = 1.0
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            score_adjust_coeff = length_penalty_prev / length_penalty
        else:
            length_penalty = 1.0
            score_adjust_coeff = 1.0

        # correct `log_probs` and `score_adjust_coeff` for the score calculation
        if is_finished.any():
            # `is_finished` shape : (remaining_batch_size, beam_size)
            finished_ids = is_finished.reshape(-1).nonzero().reshape(-1)  # (remaining_batch_size * beam_size)

            # correct `log_probs` so that the finished sequence never gets a vocab other than EOS
            log_probs[finished_ids] = float('-inf')
            log_probs[finished_ids, eos_index] = 0.0

            # correct `score_adjust_coeff` so that the scores of the finished sequences do not change
            # `score_adjust_coeff` shape: (1) -> (remaining_batch_size * beam_size)
            score_adjust_coeff *= torch.ones((remaining_batch_size*beam_size,1), device=device)
            score_adjust_coeff[finished_ids] = 1.0

        # calc corr_scores
        # 'topk_scores': (remaining_batch_size, beam_size) -> (remaining_batch_size*beam_size, 1)
        topk_scores = topk_scores.reshape(-1, 1)
        curr_scores = score_adjust_coeff * topk_scores + 1/length_penalty * log_probs

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, beam_size * trg_vocab_size)  # (remaining_batch_size, beam_size, trg_vocab_size)

        # pick currently best top k hypotheses (flattened order)
        # `topk_scores` and `topk_ids` shape: (remaining_batch_size, beam_size)
        topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_origin_ids = topk_ids.floor_divide(trg_vocab_size)  # (remaining_batch_size, beam_size)
        topk_ids = topk_ids.fmod(trg_vocab_size)  # (remaining_batch_size, beam_size)

        # map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_origin_ids                                # (remaining_batch_size, beam_size)
            + beam_offset[:remaining_batch_size].unsqueeze(1)   # (remaining_batch_size, 1)
        )  # (remaining_batch_size, beam_size)
        select_ids = batch_index.view(-1)  # (remaining_batch_size * beam_size)

        # append the latest prediction
        topk_seqs = torch.cat([
            alive_seq.index_select(0, select_ids),        # (remaining_batch_size * beam_size, step+1)
            topk_ids.view(-1, 1)                          # (remaining_batch_size * beam_size, 1)
        ], -1).reshape(remaining_batch_size, -1, step+2)  # (remaining_batch_size, beam_size, step+2)

        # update `is_finished`; (remaining_batch_size, beam_size)
        is_finished = is_finished.view(-1).index_select(0, select_ids).reshape(remaining_batch_size, beam_size)
        is_finished = topk_ids.eq(eos_index) | is_finished | topk_scores.eq(-np.inf)
        if step + 1 == max_output_length:
            is_finished.fill_(True)

        # end condition is whether the all beam is finished
        end_condition = is_finished.all(dim=-1)  # (remaining_batch_size)

        # save finished hypotheses
        if end_condition.any():
            # this is a redundant process and may be a good fix in the future
            # (we don't need 'hypotheses' and can directory save score and seq to `best_hyp`.)
            for b in [x.item() for x in end_condition.nonzero()]:
                b_org = batch_offset[b]
                for score, seq in zip(topk_scores[b], topk_seqs[b]):
                    if seq.eq(eos_index).count_nonzero().item() >= 2:
                        seq = seq[:seq.eq(eos_index).nonzero()[0].item()+1]
                    hypotheses[b_org].append(
                        (score, seq[1:])  # ignore start_token
                    )
                best_hyp = sorted(
                    hypotheses[b_org], key=lambda x: x[0], reverse=True)
                for n, (score, seq) in enumerate(best_hyp):
                    if n >= n_best:
                        break
                    results["scores"][b_org].append(score)
                    results["predictions"][b_org].append(seq)

        # batch indices of the examples which contain unfinished candidates
        unfinished = end_condition.eq(False).nonzero(as_tuple=False).view(-1)

        # if all examples are translated, no need to go further
        if len(unfinished) == 0:
            break

        # remove finished examples for the next step
        batch_index = batch_index.index_select(0, unfinished)  # (remaining_batch_size, beam_size)
        topk_scores = topk_scores.index_select(0, unfinished)  # (remaining_batch_size, beam_size)
        batch_offset = batch_offset.index_select(0, unfinished)  # (remaining_batch_size)
        alive_seq = topk_seqs.index_select(0, unfinished)  # (remaining_batch_size, beam_size, hyp_len)
        is_finished = is_finished.index_select(0, unfinished)  # (remaining_batch_size, beam_size)

        # update remaining_batch_size
        remaining_batch_size = len(batch_offset)

        # reshape `alive_seq` to its original size
        alive_seq = alive_seq.reshape(remaining_batch_size * beam_size, step + 2)  # (remaining_batch_size*beam_size, hyp_len)

        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

        # update previous length prenalty with current one
        length_penalty_prev = length_penalty

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps(
        [r[0].cpu().numpy() for r in results["predictions"]],
        pad_value=pad_index)

    return final_outputs, None


def run_batch(model: Model, batch: Batch, max_output_length: int,
              beam_size: int, beam_alpha: float) -> (np.array, np.array):
    """
    Get outputs and attentions scores for a given batch

    :param model: Model class
    :param batch: batch to generate hypotheses for
    :param max_output_length: maximum length of hypotheses
    :param beam_size: size of the beam for beam search, if 0 use greedy
    :param beam_alpha: alpha value for beam search
    :return: stacked_output: hypotheses for batch,
        stacked_attention_scores: attention scores for batch
    """
    with torch.no_grad():
        encoder_output, encoder_hidden, _, _ = model(
            return_type="encode", src=batch.src,
            src_length=batch.src_length,
            src_mask=batch.src_mask)

    # if maximum output length is not globally specified, adapt to src len
    if max_output_length is None:
        max_output_length = int(max(batch.src_length.cpu().numpy()) * 1.5)

    # greedy decoding
    if beam_size < 2:
        stacked_output, stacked_attention_scores = greedy(
            src_mask=batch.src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden)
        # batch, time, max_src_length
    else:  # beam search
        stacked_output, stacked_attention_scores = vanilla_beam_search(
            model=model,
            beam_size=beam_size,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=batch.src_mask,
            max_output_length=max_output_length,
            alpha=beam_alpha)

    return stacked_output, stacked_attention_scores
