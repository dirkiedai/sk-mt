# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import glob
import logging
import os
import re
import six
import socket
import time
import torch
import math

import torch.distributed as dist
from thumt import data, models, utils, optimizers


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train a neural machine translation model.",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2,
                        help="Path to source and target corpus.")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to load/store checkpoints.")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path to source and target vocabulary.")
    parser.add_argument("--validation", type=str,
                        help="Path to validation file.")
    parser.add_argument("--references", type=str,
                        help="Pattern to reference files.")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint.")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training.")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank of this process.")                  
    parser.add_argument("--half", action="store_true",
                        help="Enable mixed-precision training.")
    parser.add_argument("--hparam_set", type=str,
                        help="Name of pre-defined hyper-parameter set.")
    parser.add_argument("--split", type=int, default=8,
                        help="The total name of spliting piece.")                    

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper-parameters.")

    return parser.parse_args(args)


def default_params():
    params = utils.HParams(
        input=["", ""],
        output="",
        model="transformer",
        vocab=["", ""],
        pad="<pad>",
        bos="</s>",
        eos="</s>",
        unk="<unk>",
        # Dataset
        batch_size=4096,
        fixed_batch_size=False,
        min_length=1,
        max_length=320,
        buffer_size=10000,
        # Initialization
        initializer_gain=1.0,
        initializer="uniform_unit_scaling",
        # Regularization
        scale_l1=0.0,
        scale_l2=0.0,
        # Training
        initial_step=0,
        warmup_steps=4000,
        train_steps=100000,
        update_cycle=1,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
        adadelta_rho=0.95,
        adadelta_epsilon=1e-7,
        pattern="",
        clipping="none",
        clip_grad_norm=5.0,
        learning_rate=1.0,
        initial_learning_rate=1e-7,
        learning_rate_schedule="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        device_list=[0],
        # Checkpoint Saving
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=10,
        save_summary=False,
        save_checkpoint_secs=0,
        save_checkpoint_steps=1000,
        # Validation
        eval_steps=2000,
        eval_secs=0,
        top_beams=1,
        beam_size=5,
        decode_batch_size=32,
        decode_alpha=0.6,
        decode_ratio=1.2,
        decode_length=64,
        validation="",
        references="",
        use_weights = False,
        load_knn_datastore = False,
        use_knn_datastore = False, 
        knn_temperature_type = 'fix',
        knn_temperature_value=10,
        knn_lambda_type = 'fix',
        knn_lambda_value=0.5,
        knn_k_type='fix',
        k= 8,
        dstore_filename = '',
        dstore_size=1,
        probe=32,
        faiss_index='',
        use_gpu_to_search=True,
        no_load_keys=True,
        dstore_fp16=True,
        label_count_as_feature=False,
        relative_label_count=False,
        avg_k = 8,
        fp16=False,
        move_dstore_to_mem = True,
        faiss_metric_type = 'do_not_recomp_l2',
        knn_sim_func = 'do_not_recomp_l2',
        decoder_embed_dim=1024,
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if os.path.exists(p_name):
        with open(p_name) as fd:
            logging.info("Restoring hyper parameters from %s" % p_name)
            json_str = fd.readline()
            params.parse_json(json_str)

    if os.path.exists(m_name):
        with open(m_name) as fd:
            logging.info("Restoring model parameters from %s" % m_name)
            json_str = fd.readline()
            params.parse_json(json_str)

    return params


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


def override_params(params, args):
    params.model = args.model or params.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters.lower())

    params.vocabulary = {
        "source": data.Vocabulary(params.vocab[0]),
        "target": data.Vocabulary(params.vocab[1])
    }

    return params


def collect_params(all_params, params):
    collected = utils.HParams()

    for k in six.iterkeys(params.values()):
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def print_variables(model, pattern, log=True):
    trainable_parameters = []
    trainable_named_parameters = []

    for (name, var) in model.named_parameters():
        if re.search(pattern, name):
            trainable_parameters.append(var)
            trainable_named_parameters.append((name, var))
            # flags.append(True)
        # else:
        #     flags.append(False)

    weights = {v[0]: v[1] for v in model.named_parameters()}
    total_size = 0

    for name in sorted(list(weights)):
        if re.search(pattern, name):
            v = weights[name]
            total_size += v.nelement()

            if log:
                print("%s %s" % (name.ljust(60), str(list(v.shape)).rjust(15)))

    if log:
        print("Total trainable variables size: %d" % total_size)

    return trainable_parameters, trainable_named_parameters


def exclude_variables(flags, grads_and_vars):
    idx = 0
    new_grads = []
    new_vars = []

    for grad, (name, var) in grads_and_vars:
        if flags[idx]:
            new_grads.append(grad)
            new_vars.append((name, var))

        idx += 1

    return zip(new_grads, new_vars)


def save_checkpoint(run_id, step, epoch, model, optimizer, params):
    if run_id == 0:
        state = {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        utils.save(state, params.output, params.keep_checkpoint_max)


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def broadcast(model):
    for var in model.parameters():
        dist.broadcast(var.data, 0)


def get_learning_rate_schedule(params):
    if params.learning_rate_schedule == "linear_warmup_rsqrt_decay":
        schedule = optimizers.LinearWarmupRsqrtDecay(
            params.learning_rate, params.warmup_steps,
            initial_learning_rate=params.initial_learning_rate,
            summary=params.save_summary)
    elif params.learning_rate_schedule == "piecewise_constant_decay":
        schedule = optimizers.PiecewiseConstantDecay(
            params.learning_rate_boundaries, params.learning_rate_values,
            summary=params.save_summary)
    elif params.learning_rate_schedule == "linear_exponential_decay":
        schedule = optimizers.LinearExponentialDecay(
            params.learning_rate, params.warmup_steps,
            params.start_decay_step, params.end_decay_step,
            dist.get_world_size(), summary=params.save_summary)
    elif params.learning_rate_schedule == "constant":
        schedule = params.learning_rate
    else:
        raise ValueError("Unknown schedule %s" % params.learning_rate_schedule)

    return schedule


def get_clipper(params):
    if params.clipping.lower() == "none":
        clipper = None
    elif params.clipping.lower() == "adaptive":
        clipper = optimizers.adaptive_clipper(0.95)
    elif params.clipping.lower() == "global_norm":
        clipper = optimizers.global_norm_clipper(params.clip_grad_norm)
    else:
        raise ValueError("Unknown clipper %s" % params.clipping)

    return clipper


def get_optimizer(params, schedule, clipper):
    if params.optimizer.lower() == "adam":
        optimizer = optimizers.AdamOptimizer(learning_rate=schedule,
                                             beta_1=params.adam_beta1,
                                             beta_2=params.adam_beta2,
                                             epsilon=params.adam_epsilon,
                                             clipper=clipper,
                                             summaries=params.save_summary)
    elif params.optimizer.lower() == "adadelta":
        optimizer = optimizers.AdadeltaOptimizer(
            learning_rate=schedule, rho=params.adadelta_rho,
            epsilon=params.adadelta_epsilon, clipper=clipper,
            summaries=params.save_summary)
    elif params.optimizer.lower() == "sgd":
        optimizer = optimizers.SGDOptimizer(
            learning_rate=schedule, clipper=clipper,
            summaries=params.save_summary)
    else:
        raise ValueError("Unknown optimizer %s" % params.optimizer)

    return optimizer


def load_references(pattern):
    if not pattern:
        return None

    files = glob.glob(pattern)
    references = []

    for name in files:
        ref = []
        with open(name, "rb") as fd:
            for line in fd:
                items = line.strip().split()
                ref.append(items)
        references.append(ref)

    return list(zip(*references))


def main(args):
    model_cls = models.get_model(args.model)

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = default_params()
    params = merge_params(params, model_cls.default_params(args.hparam_set))
    params = import_params(args.output, args.model, params)
    params = override_params(params, args)

    # Initialize distributed utility
    if args.distributed:
        print("using distributed training")
        # print(args.local_rank)
        params.device = args.local_rank
        dist.init_process_group("nccl")
        torch.cuda.set_device(args.local_rank)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        params.device = params.device_list[args.local_rank]
        dist.init_process_group("nccl", init_method=args.url,
                                rank=args.local_rank,
                                world_size=len(params.device_list))
        torch.cuda.set_device(params.device_list[args.local_rank])
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    run_id = dist.get_rank()
    width = int(math.log10(args.split-1)) + 1
    if run_id != 0:
        params.input[0] = params.input[0] + "." + str(run_id % args.split).zfill(width)   
        params.input[1] = params.input[1] + "." + str(run_id % args.split).zfill(width) 
    print("Input Files:")
    print(params.input[0]) 
    print(params.input[1]) 
    
    # Export parameters
    if run_id == 0:
        export_params(params.output, "params.json", params)
        export_params(params.output, "%s.json" % params.model,
                      collect_params(params, model_cls.default_params()))

    model = model_cls(params).cuda()

    if args.half:
        model = model.half()
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model.train()

    schedule = get_learning_rate_schedule(params)
    clipper = get_clipper(params)
    optimizer = get_optimizer(params, schedule, clipper)

    if args.half:
        optimizer = optimizers.LossScalingOptimizer(optimizer)

    optimizer = optimizers.MultiStepOptimizer(optimizer, params.update_cycle)

    trainable_parameters, trainable_named_parameters = print_variables(model, params.pattern, run_id == 0)

    dataset = data.MTPipeline.get_train_dataset(params.input, params)

    if params.validation:
        sorted_key, eval_dataset = data.MTPipeline.get_infer_dataset(
            params.validation, params)
        references = load_references(params.references)
    else:
        sorted_key = None
        eval_dataset = None
        references = None

    # Load checkpoint
    checkpoint = utils.latest_checkpoint(params.output)

    if args.checkpoint is not None:
        # Load pre-trained models
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        step = params.initial_step
        epoch = 0
        broadcast(model)
    elif checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        step = state["step"]
        epoch = state["epoch"]
        model.load_state_dict(state["model"])

        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
    else:
        step = 0
        epoch = 0
        broadcast(model)

    def train_fn(inputs):
        features, labels = inputs
        loss = model(features, labels)
        return loss

    counter = 0

    while True:
        for features in dataset:
            if counter % params.update_cycle == 0:
                t = time.time()
                step += 1
                utils.set_global_step(step)

            counter += 1
            loss = train_fn(features)
            gradients = optimizer.compute_gradients(loss, trainable_parameters)
            # grads_and_vars = exclude_variables(
            #     trainable_flags,
            #     zip(gradients, list(model.named_parameters())))
            grads_and_vars = zip(gradients, trainable_named_parameters)
            optimizer.apply_gradients(grads_and_vars)

            if counter % params.update_cycle == 0:

                if run_id == 0:
                    t = time.time() - t
                    print("epoch = %d, step = %d, loss = %.3f (%.3f sec)" %
                        (epoch + 1, step, float(loss), t))

                if step >= params.train_steps:
                    utils.evaluate(model, sorted_key, eval_dataset,
                                   params.output, references, params)
                    save_checkpoint(run_id, step, epoch, model, optimizer, params)

                    return

                if step % params.eval_steps == 0:
                    utils.evaluate(model, sorted_key, eval_dataset,
                                   params.output, references, params)

                if step % params.save_checkpoint_steps == 0:
                    save_checkpoint(run_id, step, epoch, model, optimizer, params)

        if run_id == 0:
            epoch += 1


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
  
    if not args.distributed:
        local_args.local_rank = rank
 
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    if parsed_args.distributed:
        main(parsed_args)
    else:
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            parsed_args.url = url

        world_size = infer_gpu_num(parsed_args.parameters)

        if world_size > 1:
            # print(world_size)
            # print("using torch.multiprocessing.spawn")
            torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                        nprocs=world_size)
        else:
            process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
