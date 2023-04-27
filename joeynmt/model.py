# coding: utf-8
"""
Module to represents whole models
"""
from typing import Callable
from collections import deque

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical, Gumbel, Uniform
from logzero import logger as log
import numpy as np

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder, CriticDecoder, CriticTransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.helpers import ConfigurationError, log_peakiness, join_strings, tile
from joeynmt.metrics import bleu


class RewardRegressionModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.l1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.l2=nn.Linear(H, D_out)

    def forward(self, X):
        return self.l2(self.relu(self.l1(X)))

class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super().__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]
        self._loss_function = None # set by the TrainManager

    @property
    def loss_function(self):
        return self._x

    @loss_function.setter
    def loss_function(self, loss_function: Callable):
        self._loss_function = loss_function

    def reinforce(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
            src_length: Tensor, temperature: float, topk: int, log_probabilities: False, pickle_logs:False):

        """ Computes forward pass for Policy Gradient aka REINFORCE

        Encodes source, then step by step decodes and samples token from output distribution.
        Calls the loss function to compute the BLEU and loss

        :param max_output_length: max output length
        :param src: source input
        :param trg: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param temperature: softmax temperature
        :param topk: consider top-k parameters for logging
        :param log_probabilities: log probabilities
        :return: loss, logs
        """

        encoder_output, encoder_hidden = self._encode(src, src_length,
            src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        trg_mask = src_mask.new_ones([1, 1, 1])
        distributions = []
        log_probs = 0
        # init hidden state in case of using rnn decoder
        hidden = self.decoder._init_hidden(encoder_hidden) \
            if hasattr(self.decoder,'_init_hidden') else 0
        attention_vectors = None
        finished = src_mask.new_zeros((batch_size)).byte()
        # decode tokens
        for _ in range(max_output_length):
            previous_words = ys[:, -1].view(-1, 1) if hasattr(self.decoder,'_init_hidden') else ys
            logits, hidden, _, attention_vectors = self.decoder(
                trg_embed=self.trg_embed(previous_words),
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=hidden,
                prev_att_vector=attention_vectors,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            next_word = distrib.sample()
            log_probs += distrib.log_prob(next_word)
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            # prevent early stopping in decoding when logging gold token
            if not pickle_logs:
                # check if previous symbol was <eos>
                is_eos = torch.eq(next_word, self.eos_index)
                finished += is_eos
                # stop predicting if <eos> reached for all elements in batch
                if (finished >= 1).sum() == batch_size:
                    break
        ys = ys[:, 1:]
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        # get reinforce loss
        batch_loss, rewards, old_bleus = self.loss_function(predicted_strings, gold_strings,  log_probs)
        return (batch_loss, log_peakiness(self.pad_index, self.trg_vocab, topk, distributions,
        trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)) \
        if log_probabilities else (batch_loss, [])

    def mrt(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor, src_length: Tensor,
            temperature: float, samples: int, alpha: float, topk: int, add_gold=False, log_probabilities=False, pickle_logs=False):
        """ Computes forward pass for MRT

        Encodes source, samples multiple output sequences.
        Coputes rewards and MRT-loss

        :param max_output_length: max output length
        :param src: source input
        :param trg: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param temperature: softmax temperature
        :param samples: number of sampled sentences for MRT
        :param alpha: smootheness of MRT
        :param topk: consider top-k parameters for logging
        :param add_gold: add gold translation
        :param log_probabilities: log probabilities
        :return: loss, probability logs
        """
        if add_gold:
            samples = samples+1
        encoder_output, encoder_hidden = self._encode(src, src_length,
                    src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        trg_mask = src_mask.new_ones([1, 1, 1])
        total_prob = 0
        distributions = []
        attention_vectors = None
        encoder_output = encoder_output.repeat(samples,1,1)
        if hasattr(self.decoder,'_init_hidden'):
            hidden = self.decoder._init_hidden(encoder_hidden)
            if len(hidden)==2:
                hidden = (hidden[0].repeat(1,samples,1), hidden[1].repeat(1,samples,1))
            else:
                hidden = hidden.repeat(1,samples,1)
        else:
            hidden = (0,0)
        # repeat tensor for vectorized solution
        ys = ys.repeat(samples, 1)
        src_mask = src_mask.repeat(samples,1,1)
        finished = src_mask.new_zeros((batch_size*samples)).byte()
        # decode tokens
        for i in range(max_output_length):
            previous_words = ys[:, -1].view(-1, 1) if hasattr(self.decoder,'_init_hidden') else ys
            logits, hidden, _, attention_vectors = self.decoder(
                trg_embed=self.trg_embed(previous_words),
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=hidden,
                prev_att_vector=attention_vectors,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            next_word = distrib.sample()
            if add_gold:
                if i < trg.shape[1]:
                    ith_column = trg[:,i]
                else:
                    tensor = torch.ones((batch_size,), dtype=torch.int64)
                    data = [self.pad_index]*batch_size
                    ith_column = tensor.new_tensor(data)
                next_word[-batch_size:] = ith_column
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            total_prob += distrib.log_prob(next_word)
            # prevent early stopping in decoding when logging gold token
            if not pickle_logs:
                # check if previous symbol was <eos>
                is_eos = torch.eq(next_word, self.eos_index)
                finished += is_eos
                # stop predicting if <eos> reached for all elements in batch
                if (finished >= 1).sum() == batch_size*samples:
                    break
        ys = ys[:, 1:]
        all_sequences = torch.stack(torch.split(ys, batch_size))
        sentence_probabs= list(torch.split(total_prob, batch_size))
        predicted_outputs = [self.trg_vocab.arrays_to_sentences(arrays=sequ,
                                                        cut_at_eos=True) for sequ in all_sequences]
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_sentences = [[join_strings(wordlist) for wordlist in predicted_output]
            for predicted_output in predicted_outputs]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        all_gold_sentences = [gold_strings]*samples
        # Simon's trick
        list_of_Qs = torch.softmax(torch.stack(sentence_probabs)*alpha, 0)
        # calculate loss
        batch_loss = 0
        for index, Q in enumerate(list_of_Qs):
            for prediction, gold_ref, Q_iter in zip(predicted_sentences[index], all_gold_sentences[index], Q):
                batch_loss -= bleu([prediction], [gold_ref])*Q_iter
        rewards = [bleu([prediction], [gold_ref]) for prediction, gold_ref in zip(predicted_sentences[-1], all_gold_sentences[-1])]
        # currently unused
        Qs_to_return = [q.tolist() for q in list_of_Qs]
        return (batch_loss, log_peakiness(self.pad_index, self.trg_vocab, topk, distributions, \
            trg, batch_size, max_output_length, gold_strings, predicted_sentences, \
                Qs_to_return, rewards, mrt=True, samples=samples)) \
                if log_probabilities else (batch_loss, [])

    def soft_beam_policy_off(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
            src_length: Tensor, temperature: float, topk: int, log_probabilities: False, pickle_logs:False,
            alpha: float = 1., max_adoption_size: int = 100, beam_size: int = 5,
            gumbel_loc: float = 0., gumbel_scale: float = 1., tau_op: float = 0.5):
        """ Computes forward pass for Soft Beam Search

        Encodes source, then step by step decodes and samples token from output distribution.
        Calls the loss function to compute the BLEU and loss

        :param max_output_length: max output length
        :param src: source input
        :param trg: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param temperature: softmax temperature
        :param topk: consider top-k parameters for logging
        :param log_probabilities: log probabilities
        :param alpha: length normalization controller
        :param gumbel_loc: loc parameter of gumbel distribution
        :param gumbel_scale: scale parameter of gumbel distribution
        :param max_adoption_size: maximum size of adoption set size
        :param tau_op: off-policy adjustment coefficients
        :return: loss, logs
        """
        dev = src.device
        uniform_dist = Uniform(
            low=torch.tensor([0.], device=dev),
            high=torch.tensor([1.], device=dev),
        )
        gumbel_dist = Gumbel(
            torch.tensor([gumbel_loc], device=dev),
            torch.tensor([gumbel_scale], device=dev),
            validate_args=False,
        )

        def adoption_model(log_prob: Tensor, tau: Tensor) -> Tensor:
            return 1 - gumbel_dist.cdf(-(log_prob - tau))

        encoder_output, encoder_hidden = self._encode(src, src_length, src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        # define sets of sequences and scores (cumulative sum of score function)
        ys_tokens = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        ys_scores = encoder_output.new_zeros([batch_size, 1])
        ys_iws = encoder_output.new_ones([batch_size, 1])
        trg_mask = src_mask.new_ones([1, 1, 1])
        log_probs = 0
        # init hidden state in case of using rnn decoder
        hidden = self.decoder._init_hidden(encoder_hidden) \
            if hasattr(self.decoder,'_init_hidden') else 0
        attention_vectors = None
        finished = src_mask.new_zeros([0], dtype=torch.long)
        initial_finished = src_mask.new_zeros([0], dtype=torch.long)
        length_norms = encoder_output.new_zeros([batch_size, 1])
        alive_batches = torch.arange(batch_size, device=dev)

        # run beam search and get thresholds
        with torch.no_grad():
            thresholds, beam_sets = self._compute_threshold_by_vanilla_beam_search(
                beam_size, encoder_output, encoder_hidden, src_mask, temperature, alpha
            )

        # decode tokens with soft beam search
        for l in range(1, max_output_length):
            # eval start
            previous_words = ys_tokens[:, -1].view(-1, 1) if hasattr(self.decoder,'_init_hidden') else ys_tokens
            logits, hidden, _, attention_vectors = self.decoder(
                trg_embed=self.trg_embed(previous_words),
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=hidden,
                prev_att_vector=attention_vectors,
                trg_mask=trg_mask,
                finished=finished,
                eos_index=self.eos_index,
            )
            logits = logits[:, -1] / temperature
            # sampling probability of pg
            log_probs = log_probs + torch.log_softmax(logits, dim=1)
            # find length normalization mask (True for not EOSed sequence)
            ln_mask = ~(logits[:, self.eos_index] == 0).unsqueeze(1)
            # apply length normalization with current length l
            length_norms[ln_mask] = (5 + l) ** alpha / (5 + 1) ** alpha
            log_probs_norm = log_probs / length_norms
            # eval end

            # adoption start
            # compute adoption probability of token in behavioral policy
            adoption_prob = adoption_model(log_probs_norm, (thresholds[:, l] - tau_op).unsqueeze(1))  # (batch_size, token_size)
            # select targets to be prob 1.
            beam_l = beam_sets[1]
            prob_one_idx = torch.tensor([
                [abi, beam[-1]]
                for batch_i, batch_beams in enumerate(beam_l)
                for beam in batch_beams
                for abi, token in zip(
                    (alive_batches == batch_i).nonzero(),
                    ys_tokens[(alive_batches == batch_i).nonzero()],
                )
                if (beam[:-1] == token.squeeze(0)).all()
            ]).T  # don't want to use loops..
            if prob_one_idx.size(0) > 0:
                assert prob_one_idx.size(1) <= batch_size * beam_size, f"Illegal `prob_one_idx` size: {prob_one_idx.size(1)}"
                adoption_prob[prob_one_idx[0], prob_one_idx[1]] = 1.
            to_adopt = adoption_prob >= uniform_dist.sample(adoption_prob.size()).squeeze(-1)  # (batch_size, token_size)
            # filter adopted indexes and tokens
            filtered_indexes = to_adopt.nonzero()
            adopted_indexes = filtered_indexes[:, 0]
            if (adoption_size := adopted_indexes.size(0)) == 0:
                break
            # initialize maximum size exceeded
            exceeded = adoption_size > batch_size * max_adoption_size
            log.info(f'{l=:02d}: Adopted token set size {adoption_size}')
            if exceeded:
                log.warning(f'{l=:02d}: Adopted token set size {adoption_size} exceeds {batch_size=} * {max_adoption_size=}')
                resampled = torch.randperm(adoption_size)[:batch_size * max_adoption_size]
                filtered_indexes = filtered_indexes[resampled.sort().values]
                adopted_indexes = filtered_indexes[:, 0]
                to_adopt[:, :] = False
                to_adopt[adopted_indexes, filtered_indexes[:, 1]] = True
            prev_ys_tokens = ys_tokens.index_select(0, adopted_indexes)
            next_ys_tokens = filtered_indexes[:, 1].unsqueeze(1)
            # get scores for scores and iws
            score = adoption_model(log_probs_norm, thresholds[:, l].unsqueeze(1))
            prev_ys_scores = ys_scores.index_select(0, adopted_indexes)
            next_ys_scores = score[to_adopt].unsqueeze(1)
            prev_ys_iws = ys_iws.index_select(0, adopted_indexes)
            next_ys_iws = score[to_adopt] / adoption_prob[to_adopt]
            # append adopted tokens next to increased previous tokens
            ys_tokens = torch.cat((prev_ys_tokens, next_ys_tokens), dim=1)
            # add adoption scores to increased previous scores
            ys_scores = prev_ys_scores + next_ys_scores
            # multiply importance weights to increased previsous iws
            if exceeded:
                ys_iws = prev_ys_iws * adoption_size / (batch_size * max_adoption_size)
            else:
                ys_iws = prev_ys_iws * next_ys_iws.unsqueeze(1)
            # update other adopted tensors for next decoder I/O
            alive_batches = alive_batches.index_select(0, adopted_indexes)
            if len(beam_sets) > 2:
                beam_sets[2] = beam_sets[2].index_select(0, alive_batches.unique())
            del beam_sets[1]
            thresholds = thresholds.index_select(0, adopted_indexes)
            encoder_output = encoder_output.index_select(0, adopted_indexes)
            src_mask = src_mask.index_select(0, adopted_indexes)
            log_probs = log_probs[to_adopt].unsqueeze(dim=1)
            trg = trg.index_select(0, adopted_indexes)
            length_norms = length_norms.index_select(0, adopted_indexes)
            # adoption end

            # update finished if exists
            pre_finished = (next_ys_tokens == self.eos_index).nonzero()[:, 0]
            # re-initialize finished
            finished = initial_finished
            if pre_finished.size(0) > 0:
                finished = pre_finished

        ys_tokens = ys_tokens[:, 1:]
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys_tokens,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        # get reinforce loss
        batch_loss, rewards, old_bleus = self.loss_function(predicted_strings, gold_strings, ys_scores * ys_iws)
        return (batch_loss, log_peakiness(self.pad_index, self.trg_vocab, topk, _,
        trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)) \
        if log_probabilities else (batch_loss, [])

    def soft_beam_policy_on(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
            src_length: Tensor, temperature: float, topk: int, log_probabilities: False, pickle_logs:False,
            alpha: float = 1., max_adoption_size: int = 100, beam_size: int = 5,
            gumbel_loc: float = 0., gumbel_scale: float = 1., margin: float = 0.5, tau_op: float = None):
        """ Computes forward pass for Soft Beam Search

        Encodes source, then step by step decodes and samples token from output distribution.
        Calls the loss function to compute the BLEU and loss

        :param max_output_length: max output length
        :param src: source input
        :param trg: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param temperature: softmax temperature
        :param topk: consider top-k parameters for logging
        :param log_probabilities: log probabilities
        :param alpha: length normalization controller
        :param gumbel_loc: loc parameter of gumbel distribution
        :param gumbel_scale: scale parameter of gumbel distribution
        :param max_adoption_size: maximum size of adoption set size
        :param margin: margin from beam sequences
        :param tau_op: a dummy parameter
        :return: loss, logs
        """
        dev = src.device
        uniform_dist = Uniform(
            low=torch.tensor([0.], device=dev),
            high=torch.tensor([1.], device=dev),
        )
        gumbel_dist = Gumbel(
            torch.tensor([gumbel_loc], device=dev),
            torch.tensor([gumbel_scale], device=dev),
            validate_args=False,
        )

        def adoption_model(log_prob: Tensor, tau: Tensor) -> Tensor:
            return 1 - gumbel_dist.cdf(-(log_prob - tau))

        encoder_output, encoder_hidden = self._encode(src, src_length, src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        # define sets of sequences and scores (cumulative sum of score function)
        ys_tokens = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        ys_scores = encoder_output.new_zeros([batch_size, 1])
        trg_mask = src_mask.new_ones([1, 1, 1])
        log_probs = 0
        # init hidden state in case of using rnn decoder
        hidden = self.decoder._init_hidden(encoder_hidden) \
            if hasattr(self.decoder,'_init_hidden') else 0
        attention_vectors = None
        initial_finished = src_mask.new_zeros([0], dtype=torch.long)
        length_norms = encoder_output.new_zeros([batch_size, 1])

        # run beam search and get thresholds
        with torch.no_grad():
            thresholds, _, _ = self._compute_threshold_by_vanilla_beam_search(
                beam_size, encoder_output, encoder_hidden, src_mask, temperature, alpha
            )
            # padding with inf to match the longest seq
            thresholds = pad_sequence(thresholds, batch_first=True, padding_value=float("inf"))

        # queue for sequences to be processed later
        seq_queue = deque([(0, ys_tokens, ys_scores, thresholds, encoder_output, src_mask, trg, log_probs, length_norms)])
        decode_limit = batch_size * max_adoption_size

        # repeat queue processing when it is not empty
        log.info('Decoding start')
        tokens, scores, trgs = [], [], []
        while seq_queue:
            # process batch inside the queue until its empty
            l, ys_tokens, ys_scores, thresholds, encoder_output, src_mask, trg, log_probs, length_norms = seq_queue.popleft()
            log.info(f"\tBatch start, queue size: {len(seq_queue)}, batch size: {ys_tokens.size(0)} from step {l:02d}")

            # decode tokens with soft beam search
            beam_maxlen = thresholds.size(1)  # max beam length
            finished = initial_finished  # initialize finished sequences number
            while not (ys_tokens[:, -1] == self.eos_index).all().item():
                log.info(
                    f'\t\t{l=:02d}: Step start, '
                    f'queue size: {len(seq_queue)}, '
                    f'batch size: {ys_tokens.size(0)}, '
                    f'cuda mem: {torch.cuda.memory_allocated(0)/1024/1024/1024:.2f}GB'
                )
                # eval start
                previous_words = ys_tokens[:, -1].view(-1, 1) if hasattr(self.decoder,'_init_hidden') else ys_tokens
                logits, hidden, _, attention_vectors = self.decoder(
                    trg_embed=self.trg_embed(previous_words),
                    encoder_output=encoder_output,
                    encoder_hidden=encoder_hidden,
                    src_mask=src_mask,
                    unroll_steps=1,
                    hidden=hidden,
                    prev_att_vector=attention_vectors,
                    trg_mask=trg_mask,
                    finished=finished,
                    eos_index=self.eos_index,
                )
                logits = logits[:, -1] / temperature
                # sampling probability of pg
                log_probs = log_probs + torch.log_softmax(logits, dim=1)
                # find length normalization mask (True for not EOSed sequence)
                ln_mask = ~(logits[:, self.eos_index] == 0).unsqueeze(1)
                # apply length normalization with current length l
                length_norms[ln_mask] = (5 + l) ** alpha / (5 + 1) ** alpha
                log_probs_norm = log_probs / length_norms
                # eval end

                # if current length has reached to beam max length, continue with only greedy search
                if l >= beam_maxlen:
                    # just get argmax
                    next_log_probs, next_ys_tokens = log_probs.max(dim=1)
                    ys_tokens = torch.cat((ys_tokens, next_ys_tokens.unsqueeze(1)), dim=1)
                    log_probs = next_log_probs.unsqueeze(1)
                    # increment current length and continue
                    l += 1
                    log.info(f'\t\t{l=:02d}: Step end (all greedy)')
                    continue

                # devide batched tensors into max length reached and unreached
                greedy = (thresholds[:, l] == float("inf")).nonzero().squeeze(1)
                # for greedy (l > L)
                if use_greedy := greedy.size(0) > 0:
                    greedy_ys_tokens = ys_tokens[greedy]
                    greedy_ys_scores = ys_scores[greedy]
                    greedy_thresholds = thresholds[greedy]
                    greedy_log_probs = log_probs[greedy]
                    greedy_log_probs_norm = log_probs_norm[greedy]
                    greedy_length_norms = length_norms[greedy]
                    greedy_encoder_output = encoder_output[greedy]
                    greedy_src_mask = src_mask[greedy]
                    greedy_trg = trg[greedy]
                    # just get argmax
                    greedy_next_log_probs, greedy_next_ys_tokens = greedy_log_probs.max(dim=1)
                    greedy_ys_tokens = torch.cat((greedy_ys_tokens, greedy_next_ys_tokens.unsqueeze(1)), dim=1)
                    greedy_log_probs = greedy_next_log_probs.unsqueeze(1)

                # for soft beam policy (l <= L)
                sbp = (thresholds[:, l] != float("inf")).nonzero().squeeze(1)
                if use_sbp := sbp.size(0) > 0:
                    ys_tokens = ys_tokens[sbp]
                    ys_scores = ys_scores[sbp]
                    thresholds = thresholds[sbp]
                    log_probs = log_probs[sbp]
                    log_probs_norm = log_probs_norm[sbp]
                    length_norms = length_norms[sbp]
                    encoder_output = encoder_output[sbp]
                    src_mask = src_mask[sbp]
                    trg = trg[sbp]
                    # adopion start
                    score = adoption_model(log_probs_norm, thresholds[:, l].unsqueeze(1) - margin)  # (batch_size, token_size)
                    log.info(f'\t\t{l=:02d}: Threshold mean={thresholds[:, l].unsqueeze(1).mean().item():.5f}, '
                             f'std={thresholds[:, l].unsqueeze(1).std().item():.5f}')
                    to_adopt = score >= uniform_dist.sample(score.size()).squeeze(-1)  # (batch_size, token_size)
                    # filter adopted indexes and tokens
                    filtered_indexes = to_adopt.nonzero()
                    adopted_indexes = filtered_indexes[:, 0]
                    if (adopted_size := adopted_indexes.size(0)) == 0 and not use_greedy:
                        log.info(f'\t\t{l=:02d}: Step end with zero sample size')
                        break
                    if adopted_size > 0:
                        if adopted_size > decode_limit:
                            log.warning(f'\t\t{l=:02d}: Adopted token set size {adopted_size} exceeds {decode_limit}={batch_size}*{max_adoption_size}')
                            # enqueue overflowed batches
                            offset = decode_limit
                            for _ in range(adopted_size // decode_limit):
                                to_be_queued = adopted_indexes[offset:(offset + decode_limit)]
                                log.warning(f'\t\t{l=:02d}: Queued: {to_be_queued}')
                                seq_queue.append((
                                    l,
                                    ys_tokens.index_select(0, to_be_queued),
                                    ys_scores.index_select(0, to_be_queued),
                                    thresholds.index_select(0, to_be_queued),
                                    encoder_output.index_select(0, to_be_queued),
                                    src_mask.index_select(0, to_be_queued),
                                    trg.index_select(0, to_be_queued),
                                    log_probs.index_select(0, to_be_queued),
                                    length_norms.index_select(0, to_be_queued),
                                ))
                                offset += decode_limit
                            # just process first batch
                            adopted_indexes = adopted_indexes[:decode_limit]
                            filtered_indexes = filtered_indexes[:decode_limit]
                            to_adopt[:, :] = False
                            to_adopt[adopted_indexes, filtered_indexes[:, 1]] = True
                        prev_ys_tokens = ys_tokens.index_select(0, adopted_indexes)
                        next_ys_tokens = filtered_indexes[:, 1].unsqueeze(1)
                        prev_ys_scores = ys_scores.index_select(0, adopted_indexes)
                        next_ys_scores = score[to_adopt].unsqueeze(1)
                        # append adopted tokens next to increased previous tokens
                        ys_tokens = torch.cat((prev_ys_tokens, next_ys_tokens), dim=1)
                        # add adoption scores to increased previous scores
                        ys_scores = prev_ys_scores + next_ys_scores
                        # update other adopted tensors for next decoder I/O
                        thresholds = thresholds.index_select(0, adopted_indexes)
                        encoder_output = encoder_output.index_select(0, adopted_indexes)
                        src_mask = src_mask.index_select(0, adopted_indexes)
                        log_probs = log_probs[to_adopt].unsqueeze(dim=1)
                        trg = trg.index_select(0, adopted_indexes)
                        length_norms = length_norms.index_select(0, adopted_indexes)

                if use_greedy:
                    # if use sbp concatenate sbp and greedy batch tensors, if not use sbp assign greedy tensors directly
                    catsbp = use_sbp and adopted_size > 0
                    ys_tokens = torch.cat((ys_tokens, greedy_ys_tokens)) if catsbp else greedy_ys_tokens
                    ys_scores = torch.cat((ys_scores, greedy_ys_scores)) if catsbp else greedy_ys_scores
                    thresholds = torch.cat((thresholds, greedy_thresholds)) if catsbp else greedy_thresholds
                    log_probs = torch.cat((log_probs, greedy_log_probs)) if catsbp else greedy_log_probs
                    log_probs_norm = torch.cat((log_probs_norm, greedy_log_probs_norm)) if catsbp else greedy_log_probs_norm
                    length_norms = torch.cat((length_norms, greedy_length_norms)) if catsbp else greedy_length_norms
                    encoder_output = torch.cat((encoder_output, greedy_encoder_output)) if catsbp else greedy_encoder_output
                    src_mask = torch.cat((src_mask, greedy_src_mask)) if catsbp else greedy_src_mask
                    trg = torch.cat((trg, greedy_trg)) if catsbp else greedy_trg

                # update finished if exists
                pre_finished = (ys_tokens[:, -1] == self.eos_index).nonzero().squeeze(1)
                # re-initialize finished
                finished = initial_finished
                if pre_finished.size(0) > 0:
                    finished = pre_finished

                # increment current length
                log.info(f'\t\t{l=:02d}: Step end')
                l += 1
                # end of step

            log.info('\tBatch end')
            # append ys_tokens, ys_scores and trg
            tokens.append(ys_tokens)
            scores.append(ys_scores)
            trgs.append(trg)
            # end of decoding loop

        log.info('Decoding end')
        # end of seq queue processing

        ys_tokens = pad_sequence([t for token in tokens for t in token], batch_first=True, padding_value=self.pad_index)
        ys_tokens = ys_tokens[:, 1:]
        ys_scores = pad_sequence([s for score in scores for s in score], batch_first=True)
        trg = pad_sequence([t for trg in trgs for t in trg], batch_first=True)
        assert ys_tokens.size(0) == ys_scores.size(0)
        assert ys_tokens.size(0) == trg.size(0)
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys_tokens,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        # get reinforce loss
        batch_loss, rewards, old_bleus = self.loss_function(predicted_strings, gold_strings,  ys_scores)
        log.info(f'{batch_loss.item()=}')
        return (batch_loss, log_peakiness(self.pad_index, self.trg_vocab, topk, _,
        trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)) \
        if log_probabilities else (batch_loss, [])


    def _compute_threshold_by_vanilla_beam_search(self, beam_size: int,
                                                  encoder_output: Tensor, encoder_hidden: Tensor, src_mask: Tensor,
                                                  temperature: float, alpha: float = 1.0,
                                                  max_iteration: int = 100, smoothing_factor: float = 0.1) -> (np.array, np.array):
        """
        Compute thresholds for soft beam policy based on vanilla_beam_search with size k.

        :param beam_size: size of the beam
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param temperature: softmax temperature
        :param alpha: `alpha` factor for length penalty
        :param max_iteration: if the iteration does not end after 'max_iteration' times, the iteration is terminated
        :param smoothing_factor: If 'smoothing_factor'>0,
                                 hypotheses with score less than “threshold” but greater than "threshold - smoothing_factor"
                                 is adopted with probability "(threshold - smoothing_factor)/smoothing_factor"
        :return:
            - threshold_of_all_steps: [torch.tensor(batch=0), ..., torch.tensor(batch=batch_size-1)]
            - beam_seq_of_all_steps: [list(batch=0), ..., list(batch=batch_size-1)]
            - beam_prob_of_all_steps: [list(batch=0), ..., list(batch=batch_size-1)]
            Note: Each tensor or list contains information at each step (1~L),
                  where L is the maximum step size and may vary by batch index.
        """
        # don't use dropouts during beam search
        self.decoder.eval()

        # init
        bos_index = self.bos_index
        eos_index = self.eos_index
        trg_vocab_size = self.decoder.output_size
        device = encoder_output.device
        batch_size = src_mask.size(0)

        uniform_dist = Uniform(
            low=torch.tensor([0.], device=device),
            high=torch.tensor([1.], device=device),
        )

        # Transformer only: create target mask
        trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
        if isinstance(self, torch.nn.DataParallel):
            trg_mask = torch.stack(
                [src_mask.new_ones([1, 1]) for _ in self.device_ids])

        # numbering elements in the extended batch, i.e. beam size copies of each batch element
        beam_offset = torch.arange(0, batch_size * beam_size,
                                   step=beam_size,
                                   dtype=torch.long,
                                   device=device)

        # keeps track of the beam hypotheses to expand for each element
        beam_seq = torch.full(
            [batch_size * beam_size, 1],
            bos_index,
            dtype=torch.long,
            device=device)  # (batch_size * beam_size, hyp_len) ... now hyp_len = 1

        # keeps track of the scores of the beam hypotheses
        beam_score = torch.zeros(batch_size, beam_size, device=device)  # (batch_size, beam_size)
        # give full probability to the first beam on the first step; score := log 1 * coeff = 0,
        # since the only option of the first token is the BOS token.
        beam_score[:, 1:] = float("-inf")

        # keeps track of the adoption probs of the beam hypotheses
        beam_prob = torch.ones(batch_size*beam_size, device=device)

        # size of finished batch
        finished_batch_size = 0

        # keeps threshold of each step
        threshold_of_all_steps = [[] for b in range(batch_size)]

        # keeps beam hypotheses of each step
        beam_seq_of_all_steps = [[] for b in range(batch_size)]

        # keeps adoption probability of beam hypotheses of each step
        beam_prob_of_all_steps = [[] for b in range(batch_size)]

        # indicator if each beam seq is finished
        beam_finished = torch.full((batch_size, beam_size),
                                   False,
                                   dtype=torch.bool,
                                   device=device)  # (batch_size, beam_size)

        # indicator if all beam seqs in each batch are finished
        batch_finished = beam_finished.reshape(batch_size, -1).all(dim=-1)

        step = 0
        while not beam_finished.all() and step < max_iteration:
            step += 1

            # This decides which part of the predicted sentence we feed to the decoder to make the next prediction.
            # For Transformer, we feed the complete predicted sentence so far.
            # For Recurrent models, only feed the previous target word prediction
            decoder_input = beam_seq  # (batch_size * (beam_size+a), step)
            beam_size_ = int(beam_seq.shape[0] / batch_size) # = beam_size+a

            # expand current hypotheses
            # decode one single step
            # logits: scores before final softmax; (batch_size * beam_size + finished_batch_size, step, trg_vocab_size)
            logits, _, _, _ = self.decoder(
                trg_embed=self.trg_embed(decoder_input),  # trg_embed = embed(decoder_input)
                encoder_output=tile(encoder_output.contiguous(), beam_size_, dim=0),  # (batch_size * (beam_size+a), src_len, enc_hidden_size),
                src_mask=tile(src_mask, beam_size_, dim=0),  # (batch_size * (beam_size+a), 1, src_len),
                trg_mask=trg_mask,  # subsequent mask for Transformer only
                finished=beam_finished.reshape(-1).nonzero().squeeze(1),
                eos_index=self.eos_index,
            )

            # For the Transformer we made predictions for all time steps up to
            # this point, so we only want to know about the last time step.
            logits = logits[:, -1] / temperature  # (batch_size * (beam_size+a), trg_vocab_size)

            # compute log probability over trg vocab given a previous sequence
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1)  # (batch_size * (beam_size+a), trg_vocab_size)
            beam_vocab_score = log_probs

            # compute length penalty
            if alpha > 0:
                if step == 1:
                    length_penalty_prev = 1.0
                length_penalty = ((5.0 + step) / 6.0) ** alpha
                score_adjust_coeff = length_penalty_prev / length_penalty
            else:
                length_penalty = 1.0
                score_adjust_coeff = 1.0

            # correct `score_adjust_coeff` for  `beam_vocab_score`
            if beam_finished.any():
                # `beam_finished` shape : (batch_size, beam_size+a)
                finished_ids = beam_finished.reshape(-1).nonzero().reshape(-1)

                # correct `score_adjust_coeff` so that the scores of the finished sequences do not change
                # `score_adjust_coeff` shape: (1) -> (batch_size*(beam_size+a), 1)
                score_adjust_coeff *= torch.ones((beam_finished.numel(), 1), device=device)
                score_adjust_coeff[finished_ids] = 1.0

            # apply length penalty to `beam_vocab_score`
            # 'beam_score': (batch_size, beam_size+a) -> (batch_size*(beam_size+a), 1)
            beam_score = beam_score.reshape(-1, 1)
            # 'beam_vocab_score': (batch_size*(beam_size+a), trg_vocab_size)
            beam_vocab_score = score_adjust_coeff * beam_score + 1 / length_penalty * beam_vocab_score

            # flatten 'beam_vocab_score':  (batch_size*(beam_size+a), trg_vocab_size) -> (batch_size, (beam_size+a)*trg_vocab_size)
            beam_vocab_score = beam_vocab_score.reshape(batch_size, -1)

            # pick currently best top k hypotheses as beam set (flattened order)
            if smoothing_factor == 0:
                # `beam_score` and `beam_index` shape: (batch_size, beam_size)
                beam_score, beam_index = beam_vocab_score.topk(beam_size, dim=-1, sorted=True, largest=True)
                # calc threshold: (batch_size)
                threshold = beam_score[:,-1]

            else:  # stochastic sampling of hypothesis with score less than “threshold” but greater than "threshold - smoothing_factor"
                # sort: (batch_size, (beam_size+a)*trg_vocab_size) -> (batch_size, (beam_size+a)*trg_vocab_size)
                sorted_beam_vocab_score, sorted_beam_vocab_index = beam_vocab_score.sort(dim=1, descending=True)

                # calc threshold: (batch_size)
                threshold = sorted_beam_vocab_score[:, beam_size-1].detach().clone()

                # score adjustment
                sorted_beam_vocab_score += smoothing_factor - threshold.unsqueeze(-1)

                # delete unnecessary columns
                adoption_candidate_column = (sorted_beam_vocab_score >= 0).any(dim=0)
                sorted_beam_vocab_score = sorted_beam_vocab_score[:, adoption_candidate_column]
                sorted_beam_vocab_index = sorted_beam_vocab_index[:, adoption_candidate_column]

                # adoption
                adoption_prob = (sorted_beam_vocab_score/smoothing_factor).clamp(min=0, max=1)
                adoption_mask = adoption_prob >= uniform_dist.sample(adoption_prob.shape).squeeze(-1)
                sorted_beam_vocab_score[~adoption_mask] = -np.inf

                # delete unnecessary columns:  (batch_size, beam_size+a)
                adoption_column = adoption_mask.any(dim=0)
                beam_score = sorted_beam_vocab_score[:, adoption_column]
                beam_index = sorted_beam_vocab_index[:, adoption_column]
                adoption_prob = adoption_prob[:, adoption_column]
                adoption_mask = adoption_mask[:, adoption_column]

                # restore adjusted score to original score
                beam_score += - smoothing_factor + threshold.unsqueeze(-1)

            # reconstruct beam origin and true word ids from flattened order
            beam_origin_index = beam_index.floor_divide(trg_vocab_size)  # (batch_size, beam_size)
            word_index = beam_index.fmod(trg_vocab_size)  # (batch_size, beam_size)

            # compute `beam_finished`; (batch_size, beam_size)
            beam_finished = word_index.eq(eos_index) | beam_score.eq(-np.inf)

            # map beam_index to selected_index in the flat representation
            select_index = (
                beam_origin_index  # (batch_size, beam_size+a)
                + beam_offset.unsqueeze(1)  # (batch_size, 1)
            )  # (batch_size, beam_size+a)
            select_index = select_index.view(-1)  # (batch_size * (beam_size+a))

            # append the latest prediction
            beam_seq = torch.cat([
                beam_seq.index_select(0, select_index),  # (batch_size * (beam_size+a), step)
                word_index.view(-1, 1)  # (batch_size * (beam_size+a), 1)
            ], -1)  # (batch_size*(beam_size+1), step+1)

            # create a list-type variable of 'beam_seq' for output
            if smoothing_factor == 0:
                beam_seq_list = [beam_seq.reshape(batch_size,beam_size,step+1)[b] for b in range(batch_size)]

                # comp beam_prob (dummy)
                beam_prob_list = [beam_score.new_ones(beam_size) for b in range(batch_size)]

            else:
                beam_seq_list = [beam_seq.reshape(batch_size,-1,step+1)[b][adoption_mask[b]] for b in range(batch_size)]

                # update beam_prob
                beam_prob = beam_prob[select_index] * adoption_prob.view(-1)
                beam_prob_list = [beam_prob.reshape(batch_size,-1)[b,adoption_mask[b]] for b in range(batch_size)]

                # update beam_offset
                beam_offset = torch.arange(0, select_index.shape[0],
                                           step=select_index.shape[0]/batch_size,
                                           dtype=torch.long,
                                           device=device)

                # rewrite sequences that was not adopted into dummy eos sequences
                beam_seq[~adoption_mask.view(-1)] = eos_index

            # store threshold & beam_seq
            for b in range(batch_size):
                if ~batch_finished[b]:
                    # threshold
                    threshold_of_all_steps[b].append(threshold[b])
                    # beam_seq_list
                    beam_seq_of_all_steps[b].append(beam_seq_list[b])
                    # beam_prob_list
                    beam_prob_of_all_steps[b].append(beam_prob_list[b])

            # update 'batch_finished': (batch_size)
            batch_finished = beam_finished.reshape(batch_size, -1).all(dim=-1)

            # update previous length penalty with current one
            length_penalty_prev = length_penalty

        # reset decoder's status to training
        self.decoder.train()

        # adjust format
        threshold_of_all_steps = [torch.hstack(threshold_of_all_steps[b]) for b in range(batch_size)]

        return threshold_of_all_steps, beam_seq_of_all_steps, beam_prob_of_all_steps

    def _compute_threshold_by_vanilla_beam_search_obsolete(self, beam_size: int,
                                                encoder_output: Tensor, encoder_hidden: Tensor,
                                                src_mask: Tensor, temperature: float, max_output_length: int,
                                                alpha: float, n_best: int = None) -> (np.array, np.array):
        """
        [To BE DELETED]
        Compute thresholds for soft beam policy based on vanilla_beam_search with size k.
        :param model:
        :param beam_size: size of the beam
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param temperature: softmax temperature
        :param max_output_length:
        :param alpha: `alpha` factor for length penalty
        :param n_best: return this many hypotheses, <= beam
        :return:
            - thresholds: torch.tensor (max_output_length),
            - beam_seq_of_all_steps: [beam_seq(step=0), ..., beam_seq(step=max_output_length-1)]
        """
        # don't use dropouts during beam search
        self.decoder.eval()

        assert beam_size > 0, 'Beam size must be >0.'
        if n_best is None:
            n_best = beam_size
        else:
            assert n_best <= beam_size, f'Can only return {beam_size} best hypotheses.'

        # init
        bos_index = self.bos_index
        eos_index = self.eos_index
        pad_index = self.pad_index
        trg_vocab_size = self.decoder.output_size
        device = encoder_output.device
        batch_size = src_mask.size(0)

        # Recurrent models only: initialize RNN hidden state
        # pylint: disable=protected-access
        encoder_output_beam = tile(encoder_output.contiguous(), beam_size, dim=0)  # (batch_size * beam_size, src_len, enc_hidden_size)
        encoder_output_alive = encoder_output.contiguous()  # (batch_size, src_len, enc_hidden_size)
        src_mask_beam = tile(src_mask, beam_size, dim=0)  # (batch_size * beam_size, 1, src_len)
        src_mask_alive = src_mask  # (batch_size, 1, src_len)

        # Transformer only: create target mask
        trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
        if isinstance(self, torch.nn.DataParallel):
            trg_mask = torch.stack(
                [src_mask.new_ones([1, 1]) for _ in self.device_ids])

        # numbering elements in the extended batch, i.e. beam size copies of each batch element
        beam_offset = torch.arange(0, batch_size * beam_size,
                                step=beam_size,
                                dtype=torch.long,
                                device=device)

        # keeps track of the beam hypotheses to expand for each element
        beam_seq = torch.full(
            [batch_size * beam_size, 1],
            bos_index,
            dtype=torch.long,
            device=device)  # (batch_size * beam_size, hyp_len) ... now hyp_len = 1

        # keeps track of the scores of the beam hypotheses
        beam_score = torch.zeros(batch_size, beam_size, device=device)  # (batch_size, beam_size)
        # give full probability to the first beam on the first step; score := log 1 * coeff = 0,
        # since the only option of the first token is the BOS token.
        beam_score[:, 1:] = float("-inf")

        # keeps flag whether the all beam is finished
        are_all_beam_finished =torch.full(
            [batch_size], False, dtype=torch.bool, device=device
        )  # (batch_size)

        # size of finished batch
        finished_batch_size = 0

        # keeps track of unfinished hypotheses for the case that all beam hypotheses are finished
        alive_seq = torch.full([0,1], bos_index, dtype=torch.long, device=device)  # (finished_batch_size, hpy_len) ... for now (0,1)

        # keeps threshold of each step
        thresholds = torch.full(
            [batch_size, max_output_length], -float('inf'),
            dtype=torch.float,
            device=device)  # (batch_size, max_output_length)

        # keeps results of each step of beam hypotheses
        beam_seq_of_all_steps = [[] for _ in range(max_output_length)]  # [beam_seq at step 0, ... , beam_seq at step max_output_length-1]
        beam_seq_of_all_steps[0] = beam_seq.reshape(batch_size,beam_size,1)

        # indicator if each beam seq is finished
        beam_finished = torch.full((batch_size, beam_size),
                                False,
                                dtype=torch.bool,
                                device=device)  # (batch_size, beam_size)

        for step in range(1,max_output_length):
            # This decides which part of the predicted sentence we feed to the decoder to make the next prediction.
            # For Transformer, we feed the complete predicted sentence so far.
            # For Recurrent models, only feed the previous target word prediction
            encoder_output = torch.vstack([encoder_output_beam, encoder_output_alive[are_all_beam_finished]])
            src_mask = torch.vstack([src_mask_beam, src_mask_alive[are_all_beam_finished]])
            decoder_input = torch.vstack([beam_seq, alive_seq])  # (batch_size * beam_size + finished_batch_size, step)

            # expand current hypotheses
            # decode one single step
            # logits: scores before final softmax; (batch_size * beam_size + finished_batch_size, step, trg_vocab_size)
            logits, _, _, _ = self.decoder(
                trg_embed=self.trg_embed(decoder_input),  # trg_embed = embed(decoder_input)
                encoder_output=encoder_output,
                src_mask=src_mask,
                trg_mask=trg_mask,  # subsequent mask for Transformer only
                finished=beam_finished.reshape(batch_size * beam_size).nonzero().squeeze(1),
                eos_index=self.eos_index,
            )

            # For the Transformer we made predictions for all time steps up to
            # this point, so we only want to know about the last time step.
            logits = logits[:, -1] / temperature  # (batch_size * beam_size + finished_batch_size, trg_vocab_size)

            # compute log probability over trg vocab given a previous sequence
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1)  # (batch_size * beam_size + finished_batch_size, trg_vocab_size)
            beam_vocab_score = log_probs[:(batch_size * beam_size), :]
            alive_vocab_score = log_probs[(batch_size * beam_size):, :]

            # compute length penalty
            if alpha > 0:
                if step == 1:
                    length_penalty_prev = 1.0
                length_penalty = ((5.0 + step) / 6.0) ** alpha
                score_adjust_coeff = length_penalty_prev / length_penalty
            else:
                length_penalty = 1.0
                score_adjust_coeff = 1.0

            # apply length penalty to 'alive_vocab_score'
            if finished_batch_size > 0:
                # 'alive_vocab_score': (finished_batch_size, trg_vocab_size)
                alive_vocab_score = score_adjust_coeff * alive_score + 1/length_penalty * alive_vocab_score

            # correct `score_adjust_coeff` for  `beam_vocab_score`
            if beam_finished.any():
                # `beam_finished` shape : (batch_size, beam_size)
                finished_ids = beam_finished.reshape(-1).nonzero().reshape(-1)

                # correct `score_adjust_coeff` so that the scores of the finished sequences do not change
                # `score_adjust_coeff` shape: (1) -> (batch_size * beam_size + finished_batch_size)
                score_adjust_coeff *= torch.ones((batch_size*beam_size+finished_batch_size,1), device=device)
                score_adjust_coeff[finished_ids] = 1.0

            # apply length penalty to `beam_vocab_score`
            # 'beam_score': (batch_size, beam_size) -> (batch_size*beam_size, 1)
            beam_score = beam_score.reshape(-1, 1)
            # 'beam_vocab_score': (batch_size*beam_size, trg_vocab_size)
            beam_vocab_score = score_adjust_coeff * beam_score + 1/length_penalty * beam_vocab_score

            # flatten 'beam_vocab_score':  (batch_size*beam_size, trg_vocab_size) -> (batch_size, beam_size*trg_vocab_size)
            beam_vocab_score = beam_vocab_score.reshape(batch_size, beam_size * trg_vocab_size)

            # pick currently best top k hypotheses as beam set (flattened order)
            # `aug_beam_score` and `aug_beam_ids` shape: (batch_size, beam_size+1)
            # 'aug' is the abbreviation for 'augmented'.
            aug_beam_score, aug_beam_index = beam_vocab_score.topk(beam_size+1, dim=-1, sorted=True, largest=True)

            # reconstruct beam origin and true word ids from flattened order
            beam_origin_index = aug_beam_index.floor_divide(trg_vocab_size)  # (batch_size, beam_size+1)
            word_index = aug_beam_index.fmod(trg_vocab_size)  # (batch_size, beam_size+1)

            # compute `arg_beam_finished`; (batch_size, beam_size+1)
            aug_beam_finished = word_index.eq(eos_index) | aug_beam_score.eq(-np.inf)

            # map beam_index to selected_index in the flat representation
            select_index = (
                beam_origin_index           # (batch_size, beam_size+1)
                + beam_offset.unsqueeze(1)  # (batch_size, 1)
            )  # (batch_size, beam_size)
            select_index = select_index.view(-1)  # (batch_size * (beam_size+1))

            # append the latest prediction
            aug_beam_seq = torch.cat([
                beam_seq.index_select(0, select_index),  # (batch_size * (beam_size+1), step)
                word_index.view(-1, 1)                   # (batch_size * (beam_size+1), 1)
            ], -1).reshape(batch_size, beam_size+1, step + 1)    # (batch_size, beam_size+1, step+1)

            # separate results into 'beam_*' and 'runnerup_*'
            beam_seq_old = beam_seq  # this will be used in the process "calc thresholds"
            beam_score = aug_beam_score[:, :beam_size]        # (batch_size, beam_size)
            beam_finished = aug_beam_finished[:, :beam_size]  # (batch_size, beam_size)
            beam_seq = aug_beam_seq[:, :beam_size, :]         # (batch_size, beam_size, step+1)
            runnerup_score = aug_beam_score[:, -1]        # (batch_size)
            runnerup_finished = aug_beam_finished[:, -1]  # (batch_size)
            runnerup_seq = aug_beam_seq[:, -1, :]         # (batch_size, step+1)

            # backup beam_seq
            beam_seq_of_all_steps[step] = beam_seq

            # reshape `beam_seq` to its original size
            beam_seq = beam_seq.reshape(batch_size * beam_size, step+1)  # (batch_size*beam_size, hyp_len)

            # compute the flag whether the all beam is finished
            are_all_beam_finished_new = beam_finished.all(dim=-1)  # (batch_size)

            # calc thresholds
            if step < max_output_length-1 and n_best == beam_size:
                alive_index_old = 0
                alive_seq_old = alive_seq
                alive_seq = torch.full([0,step+1], bos_index, dtype=torch.long, device=device)
                alive_score = torch.zeros([0], device=device)
                for batch_index in range(batch_size):

                    if not are_all_beam_finished[batch_index]:
                        thresholds[batch_index, step] = (beam_score[batch_index,-1] + runnerup_score[batch_index]) /2

                        # find unfinished sequence
                        if are_all_beam_finished_new[batch_index]:
                            if not runnerup_finished[batch_index]:
                                seq = runnerup_seq[batch_index]
                            else:
                                sorted_index = beam_vocab_score[batch_index].argsort(descending=True)  # (beam_size*trg_vocab_size)
                                word_index = sorted_index.fmod(trg_vocab_size)
                                unfinished = ~ word_index.eq(eos_index)
                                first_unfinished_index = (unfinished * torch.arange(unfinished.shape[0],0,-1)).argmax()
                                beam_origin_index = sorted_index[first_unfinished_index].floor_divide(trg_vocab_size)
                                seq = torch.cat([
                                    beam_seq_old[beam_origin_index],
                                    word_index[first_unfinished_index].unsqueeze(-1)
                                ])  # (step+1)
                            alive_seq = torch.vstack([alive_seq, seq])

                    else:
                        # prep
                        score, word_index = alive_vocab_score[alive_index_old].sort(descending=True)  # (trg_vocab_size)
                        unfinished = ~ word_index.eq(eos_index)
                        first_unfinished_index = (unfinished * torch.arange(unfinished.shape[0], 0, -1, device=device)).argmax()
                        seq = torch.cat([
                            alive_seq_old[alive_index_old],
                            word_index[first_unfinished_index].unsqueeze(-1)
                        ])  # (step+1)
                        # comp threshold and alive_seq
                        th_up = beam_score[batch_index,-1]
                        # for the case that all scores are over the th_up
                        th_dn = th_up - 0.1 if (score > th_up).all() else score[score < th_up].max()
                        thresholds[batch_index, step] = (th_up + th_dn) /2
                        alive_seq = torch.vstack([alive_seq, seq])
                        alive_index_old += 1

            else:
                # calc threshold (Since this is the final step, there is no need for `alive_*` anymore.)
                thresholds[:, step] = beam_score[:,(n_best-1):(n_best+1)].mean(dim=-1)

            are_all_beam_finished = are_all_beam_finished_new

            # update previous length penalty with current one
            length_penalty_prev = length_penalty

        # reset decoder's status to training
        self.decoder.train()

        return thresholds, beam_seq_of_all_steps

    def ned_a2c(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
                        src_length: Tensor, temperature: float, critic: nn.Module, topk: int, log_probabilities=False, pickle_logs=False):
        """ Computes forward pass for NED-A2C

        Encodes source, step by step decodes and samples actor output.
        For each step decodes critic output given actor outputs as target
        Computes actor loss and critic loss

        :param max_output_length: max output length
        :param src: source input
        :param trg: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param temperature: softmax temperature
        :param critic: critic network
        :param topk: consider top-k parameters for logging
        :param log_probabilities: log probabilities
        :return: actor loss, critic loss, actor probability logs
        """

        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        trg_mask = src_mask.new_ones([1, 1, 1])
        # init actor parameters
        encoder_output, encoder_hidden = self._encode(
            src, src_length,
            src_mask)
        hidden = (self.decoder._init_hidden(encoder_hidden)) \
            if hasattr(self.decoder,'_init_hidden') else (0,0)
        attention_vectors = None
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        log_probs = 0
        distributions = []
        actor_log_probabs = []
        # init critic parameters
        critic_encoder_output, critic_encoder_hidden = critic._encode(
                src, src_length,
                src_mask)
        critic_hidden = (self.decoder._init_hidden(critic_encoder_hidden)) \
            if hasattr(self.decoder,'_init_hidden') else (0,0)
        critic_logits = []
        critic_sequence = critic_encoder_output.new_full(size=[batch_size, 1], fill_value=self.bos_index, dtype=torch.long)
        critic_attention_vectors = None
        # init dict to track eos
        eos_dict = {i:-1 for i in range(batch_size)}
        finished = src_mask.new_zeros((batch_size)).byte()
        # decode with actor
        for i in range(max_output_length):
            previous_words = ys[:, -1].view(-1, 1) if hasattr(self.decoder,'_init_hidden') else ys
            logits, hidden, _, attention_vectors = self.decoder(
                trg_embed=self.trg_embed(previous_words),
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=hidden,
                prev_att_vector=attention_vectors,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            sampled_word = distrib.sample()
            log_probs -= distrib.log_prob(sampled_word)
            ys = torch.cat([ys, sampled_word.unsqueeze(-1)], dim=1)
            actor_log_probabs.append(log_probs)
            sampled_word_list = sampled_word.tolist()
            for index in range(len(sampled_word_list)):
                if sampled_word_list[index] == self.eos_index:
                    if eos_dict[index] == -1:
                        eos_dict[index] = i
            # decode with critic, using actor as target
            critic_logit, critic_hidden, critic_attention_scores, critic_attention_vectors = critic.decoder(
                trg_embed=self.trg_embed(sampled_word.view(-1,1)),
                encoder_output=critic_encoder_output,
                encoder_hidden=critic_encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=critic_hidden,
                prev_att_vector=critic_attention_vectors,
                trg_mask=trg_mask
            )
            critic_logits.append(critic_logit)
            critic_distrib =  Categorical(logits = critic_logit.view(-1, critic_logit.size(-1)))
            critic_sample = critic_distrib.sample()
            critic_sequence = torch.cat([critic_sequence, critic_sample.view(-1, 1)], -1)
            # prevent early stopping in decoding when logging gold token
            if not pickle_logs:
                # check if previous symbol was <eos>
                is_eos = torch.eq(sampled_word, self.eos_index)
                finished += is_eos
                # stop predicting if <eos> reached for all elements in batch
                if (finished >= 1).sum() == batch_size:
                    break
        ys = ys[:, 1:]
        critic_sequence = critic_sequence[:, 1:]
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        # calculate rewards
        bleu_scores = []
        for prediction, gold_ref in zip(predicted_strings, gold_strings):
            bleu_scores.append(bleu([prediction], [gold_ref]))
        bleu_tensor = torch.FloatTensor(bleu_scores).unsqueeze(1)
        if torch.cuda.is_available():
            bleu_tensor = bleu_tensor.cuda()
        critic_logits_tensor = torch.stack(critic_logits)
        critic_logits_tensor = critic_logits_tensor.squeeze()
        if len(critic_logits_tensor.shape) == 1:
            critic_logits_tensor = critic_logits_tensor.unsqueeze(1)
        for dict_index in eos_dict:
            critic_logits_tensor[eos_dict[dict_index]:,dict_index] = 0
        critic_logits = torch.unbind(critic_logits_tensor)
        rewards = [(bleu_tensor-logit).squeeze(1) for logit in critic_logits]
        # calculate critic loss
        critic_loss = torch.cat([torch.pow(bleu_tensor-logit, 2) for logit in critic_logits]).sum()
        # calculate actor loss
        batch_loss = 0
        for log_prob, critic_logit in zip(actor_log_probabs, critic_logits):
            batch_loss += log_prob.unsqueeze(1)*(bleu_tensor-critic_logit)
        batch_loss = batch_loss.sum()
        return ([batch_loss, critic_loss], log_peakiness(self.pad_index, self.trg_vocab, topk, distributions, trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, bleu_scores)) \
        if log_probabilities else ([batch_loss, critic_loss], [])

    def forward(self, return_type: str = None, **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """ Interface for multi-gpu

        For DataParallel, We need to encapsulate all model call: model.encode(),
        model.decode(), and model.encode_decode() by model.__call__().
        model.__call__() triggers model.forward() together with pre hooks
        and post hooks, which take care of multi-gpu distribution.

        :param return_type: one of {"loss", "encode", "decode"}
        """
        if return_type is None:
            raise ValueError("Please specify return_type: "
                             "{`loss`, `encode`, `decode`}.")

        return_tuple = (None, None, None, None)
        if return_type == "loss":
            assert self.loss_function is not None
            out, _, _, _ = self._encode_decode(
                src=kwargs["src"],
                trg_input=kwargs["trg_input"],
                src_mask=kwargs["src_mask"],
                src_length=kwargs["src_length"],
                trg_mask=kwargs["trg_mask"])

            # compute log probs
            log_probs = F.log_softmax(out, dim=-1)

            # compute batch loss
            batch_loss = self.loss_function(log_probs, kwargs["trg"])

            # return batch loss
            #     = sum over all elements in batch that are not pad
            return_tuple = (batch_loss, None, None, None)
        elif return_type == "reinforce":
            loss, logging = self.reinforce(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            topk=kwargs['topk'],
            log_probabilities=kwargs["log_probabilities"],
            pickle_logs=kwargs["pickle_logs"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "mrt":
            loss, logging = self.mrt(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            alpha=kwargs["alpha"],
            samples=kwargs["samples"],
            topk=kwargs['topk'],
            add_gold=kwargs["add_gold"],
            log_probabilities=kwargs["log_probabilities"],
            pickle_logs=kwargs["pickle_logs"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "sbp":
            policy = self.soft_beam_policy_on if kwargs["sbp_policy"] == "on" else self.soft_beam_policy_off
            loss, logging = policy(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            alpha=kwargs["alpha"],
            topk=kwargs['topk'],
            log_probabilities=kwargs["log_probabilities"],
            pickle_logs=kwargs["pickle_logs"],
            max_adoption_size=kwargs["max_adoption_size"],
            beam_size=kwargs["beam_size"],
            gumbel_loc=kwargs.get("gumbel_loc", 0.),
            gumbel_scale=kwargs.get("gumbel_scale", 1.),
            margin=kwargs.get("margin", 0.5),
            tau_op=kwargs.get("tau_op", 0.5),
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "a2c":
            loss, logging = self.ned_a2c(
            critic=kwargs["critic"],
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            topk=kwargs['topk'],
            log_probabilities=kwargs["log_probabilities"],
            pickle_logs=kwargs["pickle_logs"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "encode":
            encoder_output, encoder_hidden = self._encode(
                src=kwargs["src"],
                src_length=kwargs["src_length"],
                src_mask=kwargs["src_mask"])

            # return encoder outputs
            return_tuple = (encoder_output, encoder_hidden, None, None)

        elif return_type == "decode":
            outputs, hidden, att_probs, att_vectors = self._decode(
                trg_input=kwargs["trg_input"],
                encoder_output=kwargs["encoder_output"],
                encoder_hidden=kwargs["encoder_hidden"],
                src_mask=kwargs["src_mask"],
                unroll_steps=kwargs["unroll_steps"],
                decoder_hidden=kwargs["decoder_hidden"],
                att_vector=kwargs.get("att_vector", None),
                trg_mask=kwargs.get("trg_mask", None),
                finished=kwargs.get("finished", None),
                eos_index=kwargs.get("eos_index", -1))

            # return decoder outputs
            return_tuple = (outputs, hidden, att_probs, att_vectors)
        return return_tuple

    # pylint: disable=arguments-differ
    def _encode_decode(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                       src_length: Tensor, trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self._encode(src=src,
                                                      src_length=src_length,
                                                      src_mask=src_mask)
        unroll_steps = trg_input.size(1)
        return self._decode(encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask, trg_input=trg_input,
                            unroll_steps=unroll_steps,
                            trg_mask=trg_mask)

    def _encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(self.src_embed(src), src_length, src_mask)

    def _decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
                src_mask: Tensor, trg_input: Tensor,
                unroll_steps: int, decoder_hidden: Tensor = None,
                att_vector: Tensor = None, trg_mask: Tensor = None,
                finished: Tensor = None, eos_index: int = -1) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param att_vector: previous attention vector (optional)
        :param trg_mask: mask for target steps
        :param finished: indexes of finished sequences (optional used only if decoder is TransformerDecoder)
        :param eos_index: index of eos-token (optional used only if decoder is TransformerDecoder and finished is not None)
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            prev_att_vector=att_vector,
                            trg_mask=trg_mask,
                            finished=finished,
                            eos_index=eos_index)

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed,
                                    self.trg_embed)


class _DataParallel(nn.DataParallel):
    """ DataParallel wrapper to pass through the model attributes """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None,
                is_critic: bool = False) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
            cfg["encoder"]["hidden_size"], \
            "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**cfg["encoder"],
                                    emb_size=src_embed.embedding_dim,
                                    emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**cfg["encoder"],
                                    emb_size=src_embed.embedding_dim,
                                    emb_dropout=enc_emb_dropout)

    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        if is_critic:
            decoder = CriticTransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        else:
            decoder = TransformerDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        if is_critic:
            decoder = CriticDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        else:
            decoder = RecurrentDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = Model(encoder=encoder, decoder=decoder,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)
    #if not False:
    # tie softmax layer with trg embeddings
    if not is_critic:
        if cfg.get("tied_softmax", False):
            if trg_embed.lut.weight.shape == \
                    model.decoder.output_layer.weight.shape:
                # (also) share trg embeddings and softmax layer:
                model.decoder.output_layer.weight = trg_embed.lut.weight
            else:
                raise ConfigurationError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
