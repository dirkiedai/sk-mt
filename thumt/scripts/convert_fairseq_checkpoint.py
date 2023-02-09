#!/usr/bin/env python
# coding=utf-8
# created by zhirui zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
import torch
import thumt.utils as utils
import six


def parse_args():
    parser = argparse.ArgumentParser(description="Convert fairseq model")

    parser.add_argument("--path", help="checkpoint directory")
    parser.add_argument("--output", default="average",
                        help="Output path")
    return parser.parse_args()


def export_params(output_dir, name, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)

    with open(filename, "w") as fd:

        fd.write(params.to_json())


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params



def main(args):

    # convert hyper-parameters
    params = utils.HParams(
            pad="<pad>",
            bos="<s>",
            eos="</s>",
            unk="<unk>",
            hidden_size=1024,
            encoder_filter_size=8192,
            decoder_filter_size=4096,
            num_heads=16,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.1,
            residual_dropout=0.2,
            relu_dropout=0.2,
            label_smoothing=0.1,
            normalization="after",
            fastmode=False,
            fast_layernum=6,  
            # Override default parameters
            warmup_steps=4000,
            train_steps=100000,
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
        )

    
    encoder_name_map = {    
        "self_attn.q_proj" : "self_attention.attention.q_transform",
        "self_attn.k_proj" : "self_attention.attention.k_transform",
        "self_attn.v_proj" : "self_attention.attention.v_transform",
        "self_attn.out_proj" : "self_attention.attention.o_transform",
        "self_attn_layer_norm.weight" : "self_attention.layer_norm.weight",
        "self_attn_layer_norm.bias" : "self_attention.layer_norm.bias",
        "fc1" : "feed_forward.ffn_layer.input_transform",
        "fc2" : "feed_forward.ffn_layer.output_transform",
        "final_layer_norm.weight": "feed_forward.layer_norm.weight",
        "final_layer_norm.bias": "feed_forward.layer_norm.bias",
    }

    decoder_name_map = {
        "self_attn.q_proj" : "self_attention.attention.q_transform",
        "self_attn.k_proj" : "self_attention.attention.k_transform",
        "self_attn.v_proj" : "self_attention.attention.v_transform",
        "self_attn.out_proj" : "self_attention.attention.o_transform",
        "self_attn_layer_norm.weight" : "self_attention.layer_norm.weight",
        "self_attn_layer_norm.bias" : "self_attention.layer_norm.bias",
        "encoder_attn.q_proj" : "encdec_attention.attention.q_transform",
        "encoder_attn.k_proj" : "encdec_attention.attention.k_transform",
        "encoder_attn.v_proj" : "encdec_attention.attention.v_transform",
        "encoder_attn.out_proj" : "encdec_attention.attention.o_transform",
        "encoder_attn_layer_norm.weight" : "encdec_attention.layer_norm.weight",
        "encoder_attn_layer_norm.bias" : "encdec_attention.layer_norm.bias",
        "fc1" : "feed_forward.ffn_layer.input_transform",
        "fc2" : "feed_forward.ffn_layer.output_transform",
        "final_layer_norm.weight": "feed_forward.layer_norm.weight",
        "final_layer_norm.bias": "feed_forward.layer_norm.bias",
        "retrieve_result_to_k_and_lambda": "retrieve_result_to_k_and_lambda",
    }

    pt_models = torch.load(args.path, map_location="cpu")
    state, _args = pt_models["model"], pt_models["args"]

    if _args.share_all_embeddings:
        params.add_hparam("shared_embedding_and_softmax_weights", True)
        params.add_hparam("shared_source_target_embedding", True)
        _args.shared_embedding_and_softmax_weights = True
        _args.shared_source_target_embedding = True
    elif _args.share_decoder_input_output_embed:
        params.add_hparam("shared_embedding_and_softmax_weights", True)
        params.add_hparam("shared_source_target_embedding", False)
        _args.shared_embedding_and_softmax_weights = True
        _args.shared_source_target_embedding = False
    else:
        params.add_hparam("shared_embedding_and_softmax_weights", False)
        params.add_hparam("shared_source_target_embedding", False)
        _args.shared_embedding_and_softmax_weights = False
        _args.shared_source_target_embedding = False

    # convert model parameters
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    params.hidden_size = _args.decoder_output_dim
    params.encoder_filter_size = _args.encoder_ffn_embed_dim
    params.decoder_filter_size = _args.decoder_ffn_embed_dim
    params.num_encoder_layers = _args.encoder_layers
    params.num_decoder_layers = _args.decoder_layers
    params.num_heads = _args.encoder_attention_heads
    params.attention_dropout = _args.attention_dropout
    params.label_smoothing = _args.label_smoothing


    export_params(args.output, "params.json", params)


    export_params(args.output, "transformer.json", utils.HParams(**vars(_args)))
    values = collections.OrderedDict()

    print("Old name ----------------------")
    for k, v in state.items():
        print(k)
        print(v.size())
        if k == "encoder.embed_tokens.weight":
            values['weights'] = v
        elif k == "decoder.embed_tokens.weight":
            # values['target_embedding'] = v
            pass

        elif k.startswith("encoder"):
            for map_name, replace_name in encoder_name_map.items():
                if map_name in k: 
                    new_k = k.replace(map_name, replace_name)
                    new_k = new_k.replace('layers', 'layer_stack')
                    values[new_k] = v
                    break
        elif k == "decoder.output_projection.weight":
            # values["softmax_weights"] = v
            pass
        elif k.startswith("decoder"):
            for map_name, replace_name in decoder_name_map.items():
                if map_name in k: 
                    new_k = k.replace(map_name, replace_name)
                    new_k = new_k.replace('layers', 'layer_stack')
                    values[new_k] = v
                    break
    
    print("New name ----------------------")
    for k, v in values.items():
        print(k, v.shape)

    state = {"step": 0, "epoch": 0, "model": values}
    torch.save(state, os.path.join(args.output, "average-0.pt"))


if __name__ == "__main__":
    main(parse_args())
