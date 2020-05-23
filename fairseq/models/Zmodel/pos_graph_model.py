import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

import fairseq
from fairseq import utils
from fairseq.models import (FairseqEncoder,
                            FairseqDecoder,
                            register_model,
                            register_model_architecture,
                            FairseqEncoderDecoderModel)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (MultiheadAttention,
                             TransformerEncoderLayer,
                             TransformerDecoderLayer,
                             LayerNorm,
                             AdaptiveSoftmax,
                             SinusoidalPositionalEmbedding,
                             PositionalEmbedding)

import torch_geometric as PyG
from torch_geometric.nn import GATConv, GCNConv, GINConv

from ..transformer import TransformerDecoder

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024



class GATLayer(nn.Module):

    def __init__(self, args):
        """GATLayer, A two layers GAT Network

        Arguments:
            args.gnn_layers {int>=2} -- number of gnn layers
            args.gnn_hidden_states {int} -- hidden state of attention(each attention head got hidden_state/heads state)
            args.gnn_heads {int} -- heads of multihead-attention
            args.gnn_dropout_rate {float} -- [dropout rate of GAT layers] (default: {0.6})
        
        Virtual arguments:
            args.gnn_input_features {int} -- input features = args.encoder-embed-dim
            args.gnn_output_features {int} -- output features = args.encoder-embed-dim
        """
        super(GATLayer, self).__init__()

        self.__drop__ = args.gnn_dropout_rate

        self.layers = nn.ModuleList([])
        if(args.gnn_layers > 1):
            for n in range(args.gnn_layers):
                if(n == 0):
                    self.layers.append(GATConv(args.former_encoder_dim, int(args.gnn_hidden_states/args.gnn_heads),
                                            heads=args.gnn_heads, dropout=args.gnn_dropout_rate))
                elif(n == args.gnn_layers-1):
                    self.layers.append(GATConv(args.gnn_hidden_states, args.latter_encoder_dim,
                                            heads=args.gnn_heads, dropout=args.gnn_dropout_rate, concat=False))
                else:
                    self.layers.append(GATConv(args.gnn_hidden_states, int(args.gnn_hidden_states/args.gnn_heads),
                                            heads=args.gnn_heads, dropout=args.gnn_dropout_rate))
        else:
            self.layers.append(GATConv(args.former_encoder_dim, args.latter_encoder_dim,
                                       heads=args.gnn_heads, dropout=args.gnn_dropout_rate, concat=False))

    def forward(self, graphs, x, valid=True):
        """forward

        Arguments:
            graphs List[Tensor] -- Contain a list of tensor which is the COO format of the graph
            x Tensor -- Output of transformer layers, shape=[seq_len, bsz, embed_dim]

        Returns:
            Tensor -- Output of Graph Conv layers, shape=[seq_len, bsz, embed_dim]
        """

        if(not valid):
            return x

        seq_len, bsz, embed_dim = x.shape

        # Form the graph with COO Tensor
        pyg_graphs = []
        for n, graph in enumerate(graphs):
            pyg_graph = PyG.data.Data(x=torch.squeeze(
                x[:, n, :], dim=1), edge_index=graph)
            pyg_graphs.append(pyg_graph)
        pyg_graphs = PyG.data.Batch.from_data_list(pyg_graphs)

        # Calc Conv
        conv_x = F.dropout(pyg_graphs.x, p=self.__drop__,
                           training=self.training)
        for layer in self.layers:
            conv_x = F.elu(layer(conv_x, pyg_graphs.edge_index))

        # Calc outputs
        output = conv_x.view(seq_len, bsz, embed_dim)
        return output

class NoGNN(nn.Module):

    def __init__(self, args):
        """
        A wrapper for nothing
        """
        super(NoGNN, self).__init__()

    def forward(self, graphs, x, valid=True):
        """
        Arguments:
            graphs List[Tensor] -- Contain a list of tensor which is the COO format of the graph
            x Tensor -- Output of transformer layers, shape=[seq_len, bsz, embed_dim]

        Returns:
            Tensor -- Output of Graph Conv layers, shape=[seq_len, bsz, embed_dim]
        """
        return x

class GCNLayer(nn.Module):

    def __init__(self, args):
        """GCNLayer, A two layers GAT Network

        Arguments:
            args.gnn_layers {int>=2} -- number of gnn layers
            args.gnn_hidden_states {int} -- hidden state of attention(each attention head got hidden_state state)
            args.gnn_dropout_rate {float} -- [dropout rate of GAT layers] (default: {0.6})
        
        Virtual arguments:
            args.gnn_input_features {int} -- input features = args.encoder-embed-dim
            args.gnn_output_features {int} -- output features = args.encoder-embed-dim
        """
        super(GCNLayer, self).__init__()

        self.__drop__ = args.gnn_dropout_rate

        self.layers = nn.ModuleList([])
        if(args.gnn_layers > 1):
            for n in range(args.gnn_layers):
                if(n == 0):
                    self.layers.append(GCNConv(args.former_encoder_dim, args.gnn_hidden_states))
                elif(n == args.gnn_layers-1):
                    self.layers.append(GCNConv(args.gnn_hidden_states, args.latter_encoder_dim))
                else:
                    self.layers.append(GCNConv(args.gnn_hidden_states, args.gnn_hidden_states))
        else:
            self.layers.append(GCNConv(args.former_encoder_dim, args.latter_encoder_dim))

    def forward(self, graphs, x, valid=True):
        """forward

        Arguments:
            graphs List[Tensor] -- Contain a list of tensor which is the COO format of the graph
            x Tensor -- Output of transformer layers, shape=[seq_len, bsz, embed_dim]

        Returns:
            Tensor -- Output of Graph Conv layers, shape=[seq_len, bsz, embed_dim]
        """

        if(not valid):
            return x

        seq_len, bsz, embed_dim = x.shape
        assert bsz == len(graphs)

        # Form the graph with COO Tensor
        pyg_graphs = []
        for n, graph in enumerate(graphs):
            pyg_graph = PyG.data.Data(x=torch.squeeze(
                x[:, n, :], dim=1), edge_index=graph)
            pyg_graphs.append(pyg_graph)
        pyg_graphs = PyG.data.Batch.from_data_list(pyg_graphs)

        assert pyg_graphs.x.equal(torch.reshape(x.transpose(0, 1), (-1, embed_dim)))
        assert pyg_graphs.x.view(bsz, seq_len, embed_dim).transpose(0, 1).equal(x)

        # Calc Conv
        conv_x = pyg_graphs.x
        for layer in self.layers:
            conv_x = F.relu(layer(conv_x, pyg_graphs.edge_index))
            conv_x = F.dropout(conv_x, p=self.__drop__, training=self.training)

        # Calc outputs
        output = conv_x.view(bsz, seq_len, embed_dim).transpose(0, 1)
        return output


# TODO:
# 1. Make Encoder Avaliable
# 2. Add Encoder Decoder Attention layer
# 3. Add Decoder
# 4. Add a Final PosGnnModel
# 5. Add a Transformer Model style args support

# 这个地方的self-attention层有三种层次的写法：
# 1.是利用self-attention直接写，灵活性最好，但工程量较大，效果不好说
# 2.是用transformer-layer写，结合了两层全连接层，估计比第一种方便一点
# 3.是利用transformer Encoder写，这种结构最完美，可惜有点难搞（但是可以复制粘贴？我想用这个了~）


@register_model("gnn_transformer")
class TransGnnModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--former-encoder-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--former-encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--latter-encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--latter-encoder-dim', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        # args for GNN
        parser.add_argument('--gnn-type', type=str,
                            help='type of gnn conv layers')
        parser.add_argument('--gnn-layers', type=int,
                            help='layers of gnn')
        parser.add_argument('--gnn-heads', type=int,
                            help='heads of GAT attention head')
        parser.add_argument('--gnn-hidden-states', type=int,
                            help='hidden state of gnn layers')
        parser.add_argument('--gnn-dropout-rate', type=float,
                            help='drop out rate of gnn layers')
        parser.add_argument('--gnn-valid', type=bool,
                            help='whether make gnn layers valid')
        # args for decoder
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError(
                    "--share-all-embeddings requires a joined dictionary")
            if args.former_encoder_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.former_encoder_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.former_encoder_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return GNNTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_anchors = None,
        graphs = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_anchors=src_anchors,
            graphs=graphs,
            cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Dict[str, List[Optional[Tensor]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


# Model support for gnn-transformer
@register_model_architecture("gnn_transformer", "gat_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)

    # Add former/latter encoder argument
    args.former_encoder_layers = getattr(args, "former_encoder_layers", 3)
    args.latter_encoder_layers = getattr(args, "latter_encoder_layers", 3)
    args.former_encoder_dim = getattr(args, "former_encoder_dim", 512)
    args.latter_encoder_dim = getattr(args, "latter_encoder_dim", 512)
    # Add GNN argument
    args.gnn_dropout_rate = getattr(args, 'gnn_dropout_rate', 0.6)
    args.gnn_heads = getattr(args, 'gnn_heads', 8)
    args.gnn_hidden_states = getattr(args, 'gnn_hidden_states', 512)
    args.gnn_layers = getattr(args, 'gnn_layers', 2)
    args.gnn_type = getattr(args, 'gnn_type', 'gcn')

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(
        args, "decoder_embed_dim", args.latter_encoder_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(
        args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

# Model support for transformer
@register_model_architecture("gnn_transformer", "gcn_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)

    # Add former/latter encoder argument
    args.former_encoder_layers = getattr(args, "former_encoder_layers", 3)
    args.latter_encoder_layers = getattr(args, "latter_encoder_layers", 3)
    args.former_encoder_dim = getattr(args, "former_encoder_dim", 512)
    args.latter_encoder_dim = getattr(args, "latter_encoder_dim", 512)
    # Add GNN argument
    args.gnn_dropout_rate = getattr(args, 'gnn_dropout_rate', 0.6)
    args.gnn_heads = getattr(args, 'gnn_heads', 8)
    args.gnn_hidden_states = getattr(args, 'gnn_hidden_states', 512)
    args.gnn_layers = getattr(args, 'gnn_layers', 2)
    args.gnn_type = getattr(args, 'gnn_type', 'gcn')

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(
        args, "decoder_embed_dim", args.latter_encoder_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(
        args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

@register_model_architecture("gnn_transformer", "transformer_only")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)

    # Add former/latter encoder argument
    args.former_encoder_layers = getattr(args, "former_encoder_layers", 3)
    args.latter_encoder_layers = getattr(args, "latter_encoder_layers", 3)
    args.former_encoder_dim = getattr(args, "former_encoder_dim", 512)
    args.latter_encoder_dim = getattr(args, "latter_encoder_dim", 512)
    # Add GNN argument
    args.gnn_dropout_rate = getattr(args, 'gnn_dropout_rate', 0.6)
    args.gnn_heads = getattr(args, 'gnn_heads', 8)
    args.gnn_hidden_states = getattr(args, 'gnn_hidden_states', 512)
    args.gnn_layers = getattr(args, 'gnn_layers', 2)
    args.gnn_type = getattr(args, 'gnn_type', 'none')

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(
        args, "decoder_embed_dim", args.latter_encoder_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(
        args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

class GNNTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layer_wise_attention = getattr(
            args, "layer_wise_attention", False)

        self.former_layers = nn.ModuleList([])
        self.gnn_layers = GNN_TYPE[args.gnn_type](args)
        self.latter_layers = nn.ModuleList([])
        self.former_layers.extend(
            [self.build_encoder_layer(args)
             for i in range(args.former_encoder_layers)]
        )
        self.latter_layers.extend(
            [self.build_encoder_layer(args)
             for i in range(args.latter_encoder_layers)]
        )

        self.num_layers_former = len(self.former_layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_anchors,
        graphs,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # former encoder layers
        for layer in self.former_layers:
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        # GNN layers
        x = self.gnn_layers(graphs, x, True)

        # latter encoder layers
        for layer in self.latter_layers:
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.index_select(0, new_order)
        )

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            # B x T
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],
            # B x T x C
            encoder_embedding=new_encoder_out["encoder_embedding"],
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(
                        self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers_former):
            # update layer norms
            self.former_layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )
        for i in range(self.num_layers_latter):
            # update layer norms
            self.latter_layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

GNN_TYPE={
    'gcn':GCNLayer,
    'gat':GATLayer,
    'none':NoGNN,
}

