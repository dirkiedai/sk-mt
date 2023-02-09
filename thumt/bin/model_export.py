# coding=utf-8
# Created by Zhirui Zhang, 2021/12/22


"""
export model with torch.save

file consists of
-config: str
-src_vocab: str
-tgt_vocab: str
-params: dict[str, torch.Tensor]

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import six
import torch
import math

from thumt import data, models, utils

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export well-trained NMT model.",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path to load checkpoints.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save exported model.")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint.")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path to source and target vocabulary.")
    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper-parameters.")

    return parser.parse_args()

def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        pad="<pad>",
        bos="</s>",
        eos="</s>",
        unk="<unk>",
        device_list=[0],
        # decoding
        length_penalty=0.6,
    )

    return params

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

def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(m_name):
        return params

    with open(m_name) as fd:
        print("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params

def override_params(params, args):
    params.parse(args.parameters.lower())

    params.vocabulary = {
        "source": data.Vocabulary(args.vocabulary[0]),
        "target": data.Vocabulary(args.vocabulary[1])
    }

    return params

def get_pe_embedding(channels, maxLen=5000):

    half_dim = channels // 2
    positions = torch.arange(maxLen, dtype=torch.float32)
    dimensions = torch.arange(half_dim, dtype=torch.float32)

    scale = math.log(10000.0) / float(half_dim - 1)
    dimensions.mul_(-scale).exp_()

    scaled_time = positions.unsqueeze(1) * dimensions.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                        dim=1)

    if channels % 2 == 1:
        pad = torch.zeros([signal.shape[0], 1], dtype=torch.float32)
        signal = torch.cat([signal, pad], axis=1)
    
    return torch.reshape(signal, [-1, 1, channels])

def export_obj_string(obj, header=''):
    result = header + (' = ' if header else '')
    if isinstance(obj, str):
        obj = obj.replace('\\', '\\\\')
        obj = obj.replace('\"', '\\\"')
        obj = obj.replace('\'', '\\\'')
        result += '\"' + str(obj) + '\"'
    elif isinstance(obj, bool):
        result += str(obj)
    elif isinstance(obj, (int, float)):
        result += str(obj)
    elif isinstance(obj, list):
        result += '('
        for element in obj:
            result += export_obj_string(element) + ', '
        if len(obj) > 0:
            result = result[:-2]
        result += ')'
    elif isinstance(obj, tuple):
        result += '('
        for element in obj:
            result += export_obj_string(element) + ', '
        if len(obj) > 0:
            result = result[:-2]
        result += ')'
    elif isinstance(obj, dict):
        result += '{ '
        for key, value in obj.items():
            if value is not None:
                result += export_obj_string(value, str(key)) + '; '
        result += '}'
    elif isinstance(obj, torch.Tensor):
        result += export_obj_string({
            'type': 'Tensor',
            'size': [i for i in obj.size()],
            'value': obj.view(-1).tolist()
        })
    return result


"""
Export model from transmart-train code
"""
def export_model_opt(params):
    model_opt = {}
    model_opt["source_voacb_size"] = len(params.vocabulary["source"])
    model_opt["target_vocab_size"] = len(params.vocabulary["target"])
    model_opt["source_embedding_size"] = params.hidden_size
    model_opt["target_embedding_size"] = params.hidden_size
    model_opt["encoder_hidden_size"] = params.hidden_size
    model_opt["decoder_hidden_size"] = params.hidden_size
    model_opt["attention_head"] = params.num_heads
    model_opt["ffn_hidden_size"] = params.filter_size
    model_opt["position_encoding"] = True
    model_opt["encoder_layer_num"] = params.num_encoder_layers
    model_opt["decoder_layer_num"] = params.num_decoder_layers
    model_opt["fast_mode"] = params.fastmode
    model_opt["period "] = params.fast_layernum
    
    model_opt_str = export_obj_string(model_opt, 'model_opt')
    return model_opt_str

def export_dictionary(header, vocab_file):

    vocab = {}
    count = 0

    with open(vocab_file, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            item = line.strip()
            vocab[count] = item
            count += 1

    vocab_out_str = export_obj_string({
        'count': len(vocab),
        'word_id_freq_list': [(vocab[i], i, 1) for i in range(len(vocab))]
    }, header)

    return vocab_out_str

def export_model_from_transmart(args):
    model_cls = models.get_model(args.model)

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = import_params(args.input, args.model, params)
    params = override_params(params, args)

    print("Load Checkpoint.....")

    checkpoint = torch.load(args.checkpoint)
    
    print("Start - Export Model to: %s" % args.output)
    
    export_model = {}

    export_model["config"] = export_model_opt(params)
    export_model["src_vocab"] = export_dictionary("src_vocab", args.vocabulary[0])
    export_model["tgt_vocab"] = export_dictionary("tgt_vocab", args.vocabulary[1])
    
    if params.shared_source_target_embedding:
        checkpoint["model"]["source_embedding"] = checkpoint["model"]["weights"]
        checkpoint["model"]["target_embedding"] = checkpoint["model"]["weights"]
    if params.shared_embedding_and_softmax_weights:
        checkpoint["model"]["softmax_weights"] = checkpoint["model"]["target_embedding"]
    export_model["model"] = checkpoint["model"]
    
    torch.save(export_model, args.output)
    
    print("Finish")

def main(args):
    export_model_from_transmart(args)

if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
