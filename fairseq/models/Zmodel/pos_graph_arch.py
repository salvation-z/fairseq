from fairseq.models import register_model_architecture

# Model support for gnn-transformer
@register_model_architecture("gnn_transformer", "gat_transformer")
def gat_architecture(args):
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
def gcn_architecture(args):
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
