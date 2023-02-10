# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass, field
from typing import Optional

from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from fairseq.modules.knn_datastore import KNN_Dstore
import torch.nn.functional as functional
from torch_scatter import scatter


DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class TransformerLanguageModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    character_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, uses character embedding convolutions to produce token embeddings"
        },
    )
    character_filters: str = field(
        default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
        metadata={"help": "size of character embeddings"},
    )
    character_embedding_dim: int = field(
        default=4, metadata={"help": "size of character embeddings"}
    )
    char_embedder_highway_layers: int = field(
        default=2,
        metadata={"help": "number of highway layers for character token embeddder"},
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_input_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    adaptive_input_cutoff: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of adaptive input cutoff points."},
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    # TODO common var add to parent
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("params.common.tpu")


@register_model("transformer_lm", dataclass=TransformerLanguageModelConfig)
class TransformerLanguageModel(FairseqLanguageModel):
    @classmethod
    def hub_models(cls):
        def moses_fastbpe(path):
            return {"path": path, "tokenizer": "moses", "bpe": "fastbpe"}

        return {
            "transformer_lm.gbw.adaptive_huge": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2",
            "transformer_lm.wiki103.adaptive": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2",
            "transformer_lm.wmt19.en": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2"
            ),
            "transformer_lm.wmt19.de": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2"
            ),
            "transformer_lm.wmt19.ru": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2"
            ),
        }

    def __init__(self, decoder):
        super().__init__(decoder)

    def load_state_dict(self, state_dict, strict=True, args=None):
        """we rewrite the load state dict here for only load part of trained model
        add by  
        """
        if self.decoder.knn_lambda_type == 'trainable' or self.decoder.knn_temperature_type == 'trainable' \
                or self.decoder.use_knn_datastore:

            self.upgrade_state_dict(state_dict)
            from fairseq.checkpoint_utils import prune_state_dict
            new_state_dict = prune_state_dict(state_dict, args)

            print('-----------------knn load part of model-----------------')
            model_dict = self.state_dict()

            remove_keys = []
            for k, v in new_state_dict.items():
                if k not in model_dict:
                    remove_keys.append(k)

            for k in remove_keys:
                new_state_dict.pop(k)

            model_dict.update(new_state_dict)
            return super().load_state_dict(model_dict)

        else:
            return super().load_state_dict(state_dict, strict, args)
            
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--load-knn-datastore", default=False, action='store_true')
        parser.add_argument("--dstore-filename", default=None, type=str)
        parser.add_argument("--use-knn-datastore", default=False, action='store_true')

        parser.add_argument("--dstore-fp16", action='store_true', help="if save only fp16")
        parser.add_argument("--dstore-size", metavar="N", default=1, type=int, help="datastore size")
        parser.add_argument("--k", default=8, type=int)
        parser.add_argument("--probe", default=32, type=int)

        parser.add_argument("--faiss-index", default=None, type=str)
        parser.add_argument("--faiss-metric-type", default=None, type=str)
        parser.add_argument("--knn-sim-func", default=None, type=str)

        parser.add_argument("--use-gpu-to-search", default=False, action="store_true")
        parser.add_argument("--use-weights", default=False, action="store_true")
        parser.add_argument("--no-load-keys", default=False, action="store_true")
        parser.add_argument("--move-dstore-to-mem", default=False, action="store_true")
        parser.add_argument("--only-use-max-idx", default=False, action="store_true")

        parser.add_argument("--knn-lambda-type", default="fix", type=str)
        parser.add_argument("--knn-lambda-value", default=0.5, type=float)
        parser.add_argument("--knn-lambda-net-hid-size", default=0, type=int)

        parser.add_argument("--label-count-as-feature", default=False, action="store_true")
        parser.add_argument("--relative-label-count", default=False, action="store_true")
        parser.add_argument("--knn-net-dropout-rate", default=0.5, type=float)

        #
        # parser.add_argument("--knn-lambda-net-input-label-count", default=)
        parser.add_argument("--knn-temperature-type", default="fix", type=str)
        parser.add_argument("--knn-temperature-value", default=10, type=float)
        parser.add_argument("--knn-temperature-net-hid-size", default=0, type=int)

        # we add 4 arguments for trainable k network
        parser.add_argument("--knn-k-type", default="fix", type=str)
        parser.add_argument("--max-k", default=None, type=int)
        parser.add_argument("--knn-k-net-hid-size", default=0, type=int)
        parser.add_argument("--knn-k-net-dropout-rate", default=0, type=float)

        # we add 3 arguments for trainable k_with_lambda network
        parser.add_argument("--k-lambda-net-hid-size", type=int, default=0)
        parser.add_argument("--k-lambda-net-dropout-rate", type=float, default=0.0)
        parser.add_argument("--gumbel-softmax-temperature", type=float, default=1)

        parser.add_argument("--avg-k", default=False, action='store_true')

        parser.add_argument("--only-train-knn-parameter", default=False, action='store_true')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = TransformerLMDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        if args.only_train_knn_parameter:

            for name, param in decoder.named_parameters():
                param.requires_grad = False

            for name, param in decoder.named_parameters():

                if "knn_distance_to_lambda" in name and decoder.knn_lambda_type == "trainable":
                    param.requires_grad = True

                if "knn_distance_to_k" in name and decoder.knn_k_type == "trainable":
                    param.requires_grad = True

                if "retrieve_result_to_k_and_lambda" in name and decoder.knn_lambda_type == "trainable" \
                        and decoder.knn_k_type == "trainable":
                    param.requires_grad = True

                # if "adaptive_softmax" in name and (decoder.knn_lambda_type == "trainable" or decoder.knn_k_type == "trainable"):
                #     param.requires_grad = True

        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens


class TransformerLMDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

        # add by  , trainable (distances to lambda and temperature) knn datastore
        self.fp16 = args.fp16

        if args.use_knn_datastore:
            self.knn_datastore = KNN_Dstore(args, len(dictionary))

        self.use_knn_datastore = args.use_knn_datastore
        self.knn_lambda_type = args.knn_lambda_type
        self.knn_temperature_type = args.knn_temperature_type
        self.knn_k_type = args.knn_k_type
        self.label_count_as_feature = args.label_count_as_feature
        self.relative_label_count = args.relative_label_count
        self.avg_k = args.avg_k

        if self.knn_lambda_type == "trainable" and self.knn_k_type == "trainable":

            # TODO another network to predict k and lambda at the same time without gumbel softmax
            self.retrieve_result_to_k_and_lambda = nn.Sequential(
                nn.Linear(args.max_k if not self.label_count_as_feature else args.max_k * 2,
                          args.k_lambda_net_hid_size),
                nn.Tanh(),
                nn.Dropout(p=args.k_lambda_net_dropout_rate),
                nn.Linear(args.k_lambda_net_hid_size, 2 + int(math.log(args.max_k, 2))),
                nn.Softmax(dim=-1),  # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 4 , 8 , ... , ]
            )

            nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, : args.k], gain=0.01)

            if self.label_count_as_feature:
                nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, args.k:], gain=0.1)

        else:
            if self.knn_lambda_type == 'trainable':
                # TODO, we may update the label count feature here
                self.knn_distances_to_lambda = nn.Sequential(
                    nn.Linear(args.k if not self.label_count_as_feature else args.k * 2, args.knn_lambda_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=args.knn_net_dropout_rate),
                    nn.Linear(args.knn_lambda_net_hid_size, 1),
                    nn.Sigmoid())

                if self.label_count_as_feature:
                    # nn.init.normal_(self.knn_distances_to_lambda[0].weight[:, :args.k], mean=0, std=0.01)
                    # nn.init.normal_(self.knn_distances_to_lambda[0].weight[:, args.k:], mean=0, std=0.1)

                    nn.init.xavier_normal_(self.knn_distances_to_lambda[0].weight[:, : args.k], gain=0.001)
                    nn.init.xavier_normal_(self.knn_distances_to_lambda[0].weight[:, args.k:], gain=0.1)
                    nn.init.xavier_normal_(self.knn_distances_to_lambda[-2].weight)

                else:
                    nn.init.normal_(self.knn_distances_to_lambda[0].weight, mean=0, std=0.01)

            if self.knn_temperature_type == 'trainable':
                # TODO, consider a reasonable function
                self.knn_distance_to_temperature = nn.Sequential(
                    nn.Linear(args.k + 2, args.knn_temperature_net_hid_size),
                    nn.Tanh(),
                    nn.Linear(args.knn_temperature_net_hid_size, 1),
                    nn.Sigmoid())
                # the weight shape is [net hid size, k + 1)
                nn.init.normal_(self.knn_distance_to_temperature[0].weight[:, :-1], mean=0, std=0.01)
                nn.init.normal_(self.knn_distance_to_temperature[0].weight[:, -1:], mean=0, std=0.1)

            # TODO we split the network here for different function, but may combine them in the future
            if self.knn_k_type == "trainable":

                self.knn_distance_to_k = nn.Sequential(
                    nn.Linear(args.max_k * 2 if self.label_count_as_feature else args.max_k,
                              args.knn_k_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=args.knn_k_net_dropout_rate),
                    # nn.Linear(args.knn_k_net_hid_size, args.max_k),
                    nn.Linear(args.knn_k_net_hid_size, args.max_k),
                    nn.Softmax(dim=-1))

                # nn.init.xavier_uniform_(self.knn_distance_to_k[0].weight, gain=0.01)
                # nn.init.xavier_uniform_(self.knn_distance_to_k[-2].weight, gain=0.01)
                # # TODO this maybe change or remove from here
                if self.label_count_as_feature:
                    nn.init.normal_(self.knn_distance_to_k[0].weight[:, :args.max_k], mean=0, std=0.01)
                    nn.init.normal_(self.knn_distance_to_k[0].weight[:, args.max_k:], mean=0, std=0.1)
                else:
                    nn.init.normal_(self.knn_distance_to_k[0].weight, mean=0, std=0.01)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            # use_knn_store: bool = False
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if self.use_knn_datastore:
            last_hidden = x

        if not features_only:
            x = self.output_layer(x)
        return x, extra
            




        #     if self.label_count_as_feature:
        #         return x, extra, knn_prob, knn_lambda, knn_dists, knn_index, label_counts
        #     else:
        #         return x, extra, knn_prob, knn_lambda, knn_dists, knn_index

        # else:
        #     # original situation
        #     return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output. we modify this method to return prob with
        knn result
        """

        logits = net_output[0]

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            network_probs = self.adaptive_softmax.get_log_prob(logits, target=target)

        else:
            network_probs = utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)  # [batch, seq len, vocab size]


        last_hidden = logits
        if self.use_knn_datastore:

            target = sample["target"].contiguous().view(-1)

            non_padding = target != self.padding_idx

            if not non_padding.any():
                return network_probs if log_probs else torch.exp_(network_probs)

            last_hidden = logits.contiguous().view(-1, logits.shape[-1])[target != self.padding_idx]
            last_hidden = last_hidden.unsqueeze(0)

            # we should return the prob of knn search
            knn_search_result = self.knn_datastore.retrieve(last_hidden)
            knn_dists = knn_search_result['distance']  # [batch, seq len, k]  # we need do sort
            knn_index = knn_search_result['knn_index']
            tgt_index = knn_search_result['tgt_index']

            if self.knn_temperature_type == 'trainable':
                knn_temperature = None
            else:
                knn_temperature = self.knn_datastore.get_temperature() 

            if self.label_count_as_feature:
                # TODO, we get the segment label count here, which is conflict with previous experiment
                label_counts = self.knn_datastore.get_label_count_segment(tgt_index, relative=self.relative_label_count)
                network_inputs = torch.cat((knn_dists.detach(), label_counts.detach().float()), dim=-1)
            else:
                network_inputs = knn_dists.detach()

            if self.fp16:
                network_inputs = network_inputs.half()



            if self.knn_lambda_type == "trainable" and self.knn_k_type == 'trainable':
                net_outputs = self.retrieve_result_to_k_and_lambda(network_inputs)

                k_prob = net_outputs  # [B, S, R_K]

                # we add this here only to test the effect of avg prob
                if self.avg_k:
                    k_prob = torch.zeros_like(k_prob).fill_(1. / k_prob.size(-1))

                knn_lambda = 1. - k_prob[:, :, 0: 1]  # [B, S, 1]
                k_soft_prob = k_prob[:, :, 1:]
                decode_result = self.knn_datastore.calculate_select_knn_prob(knn_index, tgt_index, knn_dists,
                                                                             last_hidden,
                                                                             knn_temperature,
                                                                             k_soft_prob,
                                                                             is_test=not self.retrieve_result_to_k_and_lambda.training, 
                                                                             over_vocab=False)

            else:
                if self.knn_lambda_type == 'trainable':
                    # self.knn_distances_to_lambda[2].p = 1.0

                    knn_lambda = self.knn_distances_to_lambda(network_inputs)

                else:
                    knn_lambda = self.knn_datastore.get_lambda() * torch.ones(last_hidden.size(0), last_hidden.size(1), device=last_hidden.device).unsqueeze(-1)

                if self.knn_k_type == "trainable":
                    # we should generate k mask
                    k_prob = self.knn_distance_to_k(network_inputs)

                    if self.knn_distance_to_k.training:
                        # print(k_prob[0])
                        k_log_prob = torch.log(k_prob)
                        k_soft_one_hot = functional.gumbel_softmax(k_log_prob, tau=0.1, hard=False, dim=-1)
                        # print(k_one_hot[0])

                    else:
                        # we get the one hot by argmax
                        _, max_idx = torch.max(k_prob, dim=-1)  # [B, S]
                        k_one_hot = torch.zeros_like(k_prob)
                        k_one_hot.scatter_(-1, max_idx.unsqueeze(-1), 1.)

                        knn_mask = torch.matmul(k_one_hot, self.knn_datastore.mask_for_distance)

                if self.knn_k_type == "trainable" and self.knn_distance_to_k.training:
                    decode_result = self.knn_datastore.calculate_select_knn_prob(knn_index, tgt_index, knn_dists,
                                                                                 last_hidden,
                                                                                 knn_temperature,
                                                                                 k_soft_one_hot, over_vocab=False)

                elif self.knn_k_type == "trainable":
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden,
                                                                          knn_temperature, knn_mask, over_vocab=False)

                else:
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden,
                                                                          knn_temperature, over_vocab=False)

            knn_probs = decode_result['prob']

            # knn_probs = torch.full([logits.shape[0]*logits.shape[1]], 0, dtype = torch.float32, device = logits.device)
            index_mask = torch.eq(tgt_index, target[target != self.padding_idx].unsqueeze(-1)).float()
            knn_probs = torch.mul(knn_probs, index_mask).sum(-1).unsqueeze(-1)

            knn_probs[knn_probs == 0] = 1e-7
            knn_probs = torch.log(knn_probs)

            network_probs = network_probs.view(-1, network_probs.size(-1))

            knn_lambda = knn_lambda.squeeze(0)
            tgt_index = tgt_index.squeeze(0)
            knn_probs = knn_probs.squeeze(0)

            if self.fp16:
                knn_probs = knn_probs.half()
                knn_lambda = knn_lambda.half()
            
            network_probs_target = network_probs[target != self.padding_idx].gather(dim=-1, index=target[non_padding].unsqueeze(-1))


            # if self.knn_lambda_type == "fix":
            #     network_probs[target != self.padding_idx] = network_probs[target != self.padding_idx] * (1 - knn_lambda) + knn_probs * knn_lambda

            # else:
            #     network_probs[target != self.padding_idx] = network_probs[target != self.padding_idx] * (1 - knn_lambda) + knn_probs
            
            combine_probs = torch.stack([network_probs_target, knn_probs], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = torch.log(1 - knn_lambda)

            if self.knn_lambda_type == 'fix':
                coeffs[1] = torch.log(knn_lambda)
            else:
                coeffs[1] = 0
            prob_target = torch.logsumexp(combine_probs + coeffs, dim=0)

            network_probs[non_padding] = torch.scatter(src = prob_target, dim = -1, index = target[non_padding].unsqueeze(-1), input = network_probs[non_padding])
            
            network_probs = network_probs.view(logits.shape[0], logits.shape[1], -1)
                
                
        return network_probs if log_probs else torch.exp(network_probs)


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


@register_model_architecture("transformer_lm", "transformer_lm")
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)


@register_model_architecture("transformer_lm", "transformer_lm_big")
def transformer_lm_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_wiki103")
@register_model_architecture("transformer_lm", "transformer_lm_baevski_wiki103")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)


@register_model_architecture("transformer_lm", "transformer_lm_gbw")
@register_model_architecture("transformer_lm", "transformer_lm_baevski_gbw")
def transformer_lm_baevski_gbw(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    transformer_lm_big(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt")
def transformer_lm_gpt(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_small")
def transformer_lm_gpt2_small(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_medium")
def transformer_lm_gpt2_medium(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1280)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 5120)
    args.decoder_layers = getattr(args, "decoder_layers", 36)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 20)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_big")
def transformer_lm_gpt2_big(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1600)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 6400)
    args.decoder_layers = getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 25)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)
