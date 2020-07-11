# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copied from multihead_attention.py
# Change it for phrase level gaussian attention
# TODO:
# 1. Graph based function
# 2. Convlution based function

# Phrase_args
# 1. generate_function
# 2. parse_function
# 3. center_first
# 4. window_size
# Phrase_info
# Notimplemented yet

import math
from typing import Dict, Optional, Tuple
from math import ceil

import torch
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.incremental_decoding_utils import with_incremental_state

# from torch_geometric.nn import GATConv, GCNConv


class PhraseGenerator(nn.Module):
    """
    Phrase level representation generator
    1. Parsing the seqence for different function
    """

    def __init__(
        self,
        phrase_args,
    ):
        """
        init function

        Args:
            embed_dim ([int]): [the input dimension (is the same as output dimension)]
            generate_function ([str]): using different phrase generate functions
            center_first ([bool, default None]): whether let the 1st token to be the center of the phrase
        """
        super().__init__()
        generate_function = phrase_args.generate_function
        center_first = phrase_args.center_first
        self.__parse_func__ = PhraseBuilder(phrase_args)
        # Basic function
        if(generate_function == 'max-pooling'):
            self.__type__ = generate_function
            self.__repr_func__ = lambda tokens: torch.max(tokens, 1)[0]
        elif(generate_function == 'averate-pooling'):
            self.__type__ = generate_function
            self.__repr_func__ = lambda tokens: torch.mean(tokens, 1)[0]

        # Graph based function
        # Not implemented
        # Undone
        elif(generate_function == 'GAT'):
            assert type(center_first) == bool
            self.__type__ = generate_function
            raise NotImplementedError
            pass
        elif(generate_function == 'GCN'):
            assert type(center_first) == bool
            self.__type__ = generate_function
            raise NotImplementedError
            pass

        # Conv based function
        # Undone
        elif(generate_function == 'CNN'):
            raise NotImplementedError
            pass
        else:
            # Return first token as outputs
            self.__repr_func__ = lambda tokens: tokens[0]

        return

    def forward(
        self,
        x,
        phrase_info,
    ):
        """
        forward method

        Args:
            x ([Tensor]): [(seq_len, bsz, embed_dim) the tensor in attention layer]
            phrase_info ([dict]): [used for parsing]

        Returns:
            [Tensor]: [(phrase_num, bsz, embed_dim)]
        """
        parsed, phrase_info = self.__parse_func__(x, phrase_info)
        output = self.__repr_func__(parsed)
        return output, phrase_info


# Undone
# 1. fixed_window √
# 2. graph based ×
class PhraseBuilder:
    def __init__(self, phrase_args):
        """
        [Parsing the seq into Phrases, each sentence is parsed into multiple phrases]

        Args:
            phrase_args ([dict]): [used for parsing]

        Returns:
            [Tensor]: [phrase_len, phrase_num, bsz, embed_dim]
        """
        self.parse_function = phrase_args.parse_function
        if(self.parse_function == 'fixed_window'):
            assert 'window_size' in dir(phrase_args), (
                'Using fixed window, but the size of window is not indicated'
            )
            self.window_size = phrase_args.window_size

    def __call__(self, x, phrase_info):
        """
        [Parsing the seq into Phrases, each sentence is parsed into multiple phrases]

        Args:
            x ([Tensor]): (seq_len, bsz, embed_dim) the tensor in attention layer
            phrase_info ([dict]): [used for parsing and etc.]

        Returns:
            [Tensor]: [phrase_len, phrase_num, bsz, embed_dim]
        """

        if(self.parse_function == 'fixed_window'):
            seq_length = x.size(0)
            bsz = x.size(1)
            chunks = ceil(seq_length / self.window_size)
            pad = (0, chunks * self.window_size - seq_length)
            # Padding Zero to the Tensor X
            x = x.transpose(0, -1)
            x = F.pad(x, pad)
            x = x.transpose(0, -1)
            x = x.chunk(chunks, dim=0)
            result = torch.stack(x, dim=1)
            fixed_mu = torch.arange(
                self.window_size, seq_length, self.window_size)
            fixed_mu = fixed_mu.repeat(bsz, 1)
            fixed_sigam = torch.full((seq_length, bsz), self.window_size/4)
            phrase_info['fixed_mu'] = fixed_mu
            phrase_info['fixed_sigma'] = fixed_sigam

        return result, phrase_info


# Undone
# 1. reset para (for max/mean pooling there is no para ~~)
# 2. forward √
# 3. init √
@with_incremental_state
class MultiPhraseAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.

    Note:
        1. By default the torch version MHA is turned on in MultiHeadAttention, but it is deleted here
        2. The add_zero_attention is also deleted here, because i have no idea what it is
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        phrase_args=None,
        apply_phrase=False,
    ):
        super().__init__()

        # what ever mode is running, phrase args should be given
        assert phrase_args is not None
        self.phrase_args = phrase_args

        # if both attention is turned on, there will be two W_k and W_q (W_v will remain the same as origin)
        self.gaussian_attention = self.phrase_args.gaussian_attention
        self.multihead_attention = self.phrase_args.multihead_attention
        assert self.multihead_attention or self.gaussian_attention, (
            'At least one attention should be added'
        )
        # init for phrase repr
        self.apply_phrase = apply_phrase
        # If apply_phrase is set True, we supposed that the key is tokens
        # If apply_phrase is set False, we sepposed that the key is phrase
        if(self.apply_phrase):
            self.phrase_encoder = PhraseGenerator(phrase_args)
            assert self.gaussian_attention

        # original args
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # Note:
        # 1. if self_attention&gaussian_attention = True, apply_phrase should also be True
        # 2. if encoder_decoder_attention=True, apply_phrase should be False
        self.self_attention = self_attention
        if(self.self_attention and self.gaussian_attention):
            assert self.apply_phrase
        self.encoder_decoder_attention = encoder_decoder_attention
        if(self.encoder_decoder_attention):
            assert not self.apply_phrase

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        # projection layers
        if(self.gaussian_attention):
            self.k_proj_gauss = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.q_proj_gauss = nn.Linear(embed_dim, embed_dim, bias=bias)
        if(self.multihead_attention):
            self.k_proj_base = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.q_proj_base = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            if(self.gaussian_attention):
                self.bias_k_gauss = Parameter(torch.Tensor(1, 1, embed_dim))
            if(self.multihead_attention):
                self.bias_k_base = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k_gauss = self.bias_v = self.bias_k_base = None

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            if(self.gaussian_attention):
                nn.init.xavier_uniform_(
                    self.k_proj_gauss.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(
                    self.q_proj_gauss.weight, gain=1 / math.sqrt(2))
            if(self.multihead_attention):
                nn.init.xavier_uniform_(
                    self.k_proj_base.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(
                    self.q_proj_base.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        else:
            if(self.gaussian_attention):
                nn.init.xavier_uniform_(self.k_proj_gauss.weight)
                nn.init.xavier_uniform_(self.q_proj_gauss.weight)
            if(self.multihead_attention):
                nn.init.xavier_uniform_(self.k_proj_base.weight)
                nn.init.xavier_uniform_(self.q_proj_base.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k_gauss is not None:
            nn.init.xavier_normal_(self.bias_k_gauss)
        if self.bias_k_base is not None:
            nn.init.xavier_normal_(self.bias_k_base)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def gauss_builder(self, mus, sigmas, weights, seq_length):
        """
        Generate Gauss attention

        Args:
            mus (Tensor): the mu of the gauss attention for each sequence (phrase_num, bsz)
            sigmas (Tensor): the sigma of the gauss attention for each sequence (phrase_num, bsz)
            seq_length (int): the length of sequences

        Return:
            attention (Tensor): The attention generated by token and phrase repr (seq_length, bsz)
        """

        def gauss_distribution(mu, sigma, x):
            x = x.float()
            base = torch.exp(-(x - mu) * (x - mu) / (2 * sigma * sigma))
            return base / (math.sqrt(2 * math.pi) * sigma)

        bsz = mus.size()[1]
        x = [torch.arange(0, seq_length) for i in range(bsz)]
        y = torch.zeros_like(torch.stack(x)).float()
        for n, (i, m, s, w) in enumerate(zip(x, mus, sigmas, weights)):
            for mu, sigma, weight in zip(m, s, w):
                y[n] += weight * gauss_distribution(mu, sigma, i)
        gauss_attention = y
        return gauss_attention

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        phrase_info: dict = None,
        need_phrase: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.

            query: tokens(source side)
            key: phrase repr
            value: tokens(source/target side)
            phrase_info (dict, optional): used for phrase parsing
            need_phrase (bool, False): return the phrase repr
        """
        if need_head_weights:
            need_weights = True

        key_phrase = None
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        # Here in self_attention, only query is needed
        if self.self_attention:
            if(self.multihead_attention):
                q_base = self.q_proj_base(query)
                k_base = self.k_proj_base(query)
            if(self.gaussian_attention):
                q_gauss = self.q_proj_gauss(query)
                key_phrase = self.phrase_encoder(query, phrase_info)
                k_gauss = self.k_proj_gauss(key_phrase)
            v = self.v_proj(query)

        # In encoder_decoder attention, phrase(k) and token(v) are provided by encoder
        # while token(q) is provided by decoder
        elif self.encoder_decoder_attention:
            # Basic multihead attention's k&v are provided by encoder and k = v
            if(self.multihead_attention):
                q_base = self.q_proj_base(query)
                if key is None:
                    assert value is None
                    k_base = v = None
                else:
                    k_base = self.k_proj_base(key)
                    v = self.v_proj(key)
            # Gaussian attention's key&value are provided by encoder but key!=value
            # Not that there is no need to build phrase in decoder, because it is done by the encoder
            if(self.gaussian_attention):
                q_gauss = self.q_proj_gauss(query)
                if key is None:
                    assert value is None
                    k_gauss = v = None
                else:
                    assert key is not None
                    assert value is not None
                    key_phrase = key
                    k_gauss = self.k_proj_gauss(key)
                    v = self.v_proj(value)
        else:
            # Note:
            # If both key and value are provided, and apply_phrase is set False,
            # we supposed that key is phrase repr,
            # which means no PhraseEncoder will be added here
            assert key is not None and value is not None
            if(self.multihead_attention):
                q_base = self.q_proj_base(query)
                k_base = self.k_proj_base(key)
            if(self.gaussian_attention):
                q_gauss = self.q_proj_gauss(query)
                if(self.apply_phrase):
                    key_phrase = self.phrase_encoder(query, phrase_info)
                    k_gauss = self.k_proj_gauss(key_phrase)
                else:
                    k_gauss = self.k_proj_gauss(key)
            v = self.v_proj(value)

        q_base *= self.scaling
        q_gauss *= self.scaling

        if self.bias_k_base is not None:
            k_base = torch.cat([k_base, self.bias_k_base.repeat(1, bsz, 1)])

        if self.bias_k_gauss is not None:
            k_gauss = torch.cat([k_gauss, self.bias_k_gauss.repeat(1, bsz, 1)])

        if(self.bias_k_base or self.bias_k_gauss):
            assert self.bias_v is not None
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(
                            key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        # embed_dim = head_dim * head_num
        # q: (tgt_len, bsz, embed_dim) -> (bsz * head_num, tgt_len, head_dim)
        # k: (phrase_num, bsz, embed_dim) -> (bsz * head_num, phrase_num, head_dim)
        # v: (src_len, bsz, embed_dim) -> (bsz * head_num, scr_len, head_dim)
        # Now, the implement suppose fixed window~
        # TODO graph based function is not supported yet
        if(self.multihead_attention):
            q_base = (
                q_base.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            if k_base is not None:
                k_base = (
                    k_base.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
                )
        if(self.gaussian_attention):
            q_gauss = (
                q_gauss.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            if k_gauss is not None:
                k_gauss = (
                    k_gauss.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
                )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            # From saved_state get keys
            if "prev_key_base" in saved_state:
                assert self.multihead_attention
                _prev_key_base = saved_state["prev_key_base"]
                assert _prev_key_base is not None
                prev_key_base = _prev_key_base.view(
                    bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k_base = prev_key_base
                else:
                    assert k_base is not None
                    k_base = torch.cat([prev_key_base, k_base], dim=1)
            if "prev_key_gauss" in saved_state:
                assert self.gaussian_attention
                _prev_key_gauss = saved_state["prev_key_gauss"]
                assert _prev_key_gauss is not None
                prev_key_gauss = _prev_key_gauss.view(
                    bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k_gauss = prev_key_gauss
                else:
                    assert k_gauss is not None
                    k_gauss = torch.cat([prev_key_gauss, k_gauss], dim=1)

            # From saved_state get values
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(
                    bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)

            # apply saved mask
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]

            assert v is not None
            assert k_base or k_gauss

            key_padding_mask = MultiPhraseAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k_base.size(1),
                static_kv=static_kv,
            )

            # save the newest state
            if(self.multihead_attention):
                saved_state["prev_key_base"] = k_base.view(
                    bsz, self.num_heads, -1, self.head_dim)
            if(self.gaussian_attention):
                saved_state["prev_key_gauss"] = k_gauss.view(
                    bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(
                incremental_state, saved_state)

        if(self.multihead_attention):
            assert k_base is not None
            src_len = k_base.size(1)
        else:
            assert k_gauss is not None
            src_len = k_gauss.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        # calc multihead attention
        if(self.multihead_attention):
            base_attn = torch.bmm(q_base, k_base.transpose(1, 2))
        else:
            base_attn = None

        # calc gaussian attention
        if(self.gaussian_attention):
            gauss_weight = torch.bmm(q_gauss, k_gauss.transpose(1, 2))
            gauss_attn = self.gauss_builder(
                phrase_info['fixed_mus'], phrase_info['fixed_sigmas'], gauss_weight, tgt_len)
            if(base_attn is None):
                base_attn = torch.zeros_like(gauss_attn)
        else:
            gauss_attn = torch.zeros_like(base_attn)

        # add attention together (maybe add after softmax is better? )
        attn_weights = gauss_attn + base_attn

        attn_weights = MultiPhraseAttention.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [
            bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(
                    2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        # apply softmax and dropout
        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )

        # apply attention
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [
            bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(
                tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        if(need_phrase):
            assert key_phrase is not None
            return attn, attn_weights, key_phrase
        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(
                incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix +
                             "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix +
                             "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix +
                                 "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim: 2 * dim
                    ]
                    items_to_add[prefix +
                                 "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
