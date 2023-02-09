# coding=utf-8
# Copyright 2017-Present The THUMT Authors

import torch

from thumt.data.dataset import Dataset, ElementSpec, MapFunc, TextLineDataset
from thumt.data.vocab import Vocabulary
from thumt.tokenizers import WhiteSpaceTokenizer


def _sort_input_file(filename, tm_count=0, reverse=True):
    with open(filename, "r", encoding='utf-8') as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.split())) for i, line in enumerate(inputs)]

    sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                               reverse=reverse)

    sorted_keys = {}
    sorted_inputs = []
    if tm_count != 0:
        sorted_tm_source_inputs = [ [] for i in range(tm_count) ]
        sorted_tm_target_inputs = [ [] for i in range(tm_count) ]

    for i, (idx, _) in enumerate(sorted_input_lens):
        if tm_count == 0:
            sorted_keys[idx] = i
            sorted_inputs.append(inputs[idx])
            continue

        items = inputs[idx].split('\t')
        sorted_inputs.append(items[0])
        for tm_idx in range(tm_count):
            if len(items[1 + tm_idx * 2].split(' ')) < 200:
                sorted_tm_source_inputs[tm_idx].append(items[1 + tm_idx * 2])
                sorted_tm_target_inputs[tm_idx].append(items[2 + tm_idx * 2])
            else:
                sorted_tm_source_inputs[tm_idx].append("1")
                sorted_tm_target_inputs[tm_idx].append("1")
        sorted_keys[idx] = i

    if tm_count:
        return sorted_keys, sorted_inputs, sorted_tm_source_inputs, sorted_tm_target_inputs
    else:
        return sorted_keys, sorted_inputs


class MTPipeline(object):

    @staticmethod
    def get_train_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])
        lab_dataset = TextLineDataset(filenames[1])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset, lab_dataset))
        # dataset = dataset.shard(torch.distributed.get_world_size(),
        #                         torch.distributed.get_rank())

        def bucket_boundaries_old(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        def bucket_boundaries(max_length, min_length=8, step=1.4):
            x = min_length
            boundaries = []

            while x < max_length:
                boundaries.append(x)
                x = max(x + 1, (int(x * step) // 8) * 8)
            boundaries.append(max_length)
            
            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // x)
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        dataset = Dataset.bucket_by_sequence_length(
            dataset, boundaries, batch_sizes, pad=src_vocab[params.pad],
            min_length=params.min_length, max_length=params.max_length)

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs
            src_seq = torch.tensor(src_seq)
            tgt_seq = torch.tensor(tgt_seq)
            labels = torch.tensor(labels)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float()
            tgt_mask = tgt_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                tgt_seq = tgt_seq.cuda(params.device)
                tgt_mask = tgt_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_eval_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])
        lab_dataset = TextLineDataset(filenames[1])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset, lab_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs
            src_seq = torch.tensor(src_seq)
            tgt_seq = torch.tensor(tgt_seq)
            labels = torch.tensor(labels)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float()
            tgt_mask = tgt_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                tgt_seq = tgt_seq.cuda(params.device)
                tgt_mask = tgt_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_infer_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        sorted_keys, sorted_data = _sort_input_file(filenames, reverse=False)

        src_dataset = TextLineDataset(sorted_data)

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])

        dataset = Dataset.zip((src_dataset,))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq = inputs[0]
            src_seq = torch.tensor(src_seq)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            src_mask = src_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
            }

            return features

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return sorted_keys, dataset

    @staticmethod
    def get_infer_tm_dataset(filename, params, cpu=False):

        tm_count = params.tm_count
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]
        sorted_keys, sorted_data, sorted_tm_source, sorted_tm_target = _sort_input_file(filename, tm_count)
        src_dataset = TextLineDataset(sorted_data)
        # tm_src_dataset = TextLineDataset(sorted_tm_source)
        # tm_tgt_dataset = TextLineDataset(sorted_tm_target)
        # tm_lab_dataset = TextLineDataset(sorted_tm_target)

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        # tm_src_dataset = tm_src_dataset.tokenize(WhiteSpaceTokenizer(),
        #                                    None, params.eos)
        # tm_tgt_dataset = tm_tgt_dataset.tokenize(WhiteSpaceTokenizer(),
        #                                    params.bos, None)
        # tm_lab_dataset = tm_lab_dataset.tokenize(WhiteSpaceTokenizer(),
        #                                    None, params.eos)

        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        # tm_src_dataset = Dataset.lookup(tm_src_dataset, src_vocab,
        #                              src_vocab[params.unk])
        # tm_tgt_dataset = Dataset.lookup(tm_tgt_dataset, tgt_vocab,
        #                              tgt_vocab[params.unk])
        # tm_lab_dataset = Dataset.lookup(tm_lab_dataset, tgt_vocab,
        #                              tgt_vocab[params.unk])

        tm_src_dataset_list = []
        tm_tgt_dataset_list = []
        tm_lab_dataset_list = []

        for idx in range(tm_count):

            tm_src_dataset = TextLineDataset(sorted_tm_source[idx])
            tm_tgt_dataset = TextLineDataset(sorted_tm_target[idx])
            tm_lab_dataset = TextLineDataset(sorted_tm_target[idx])

            tm_src_dataset = tm_src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
            tm_tgt_dataset = tm_tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.eos, None)
            tm_lab_dataset = tm_lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)

            tm_src_dataset = Dataset.lookup(tm_src_dataset, src_vocab,
                                     src_vocab[params.unk])
            tm_tgt_dataset = Dataset.lookup(tm_tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
            tm_lab_dataset = Dataset.lookup(tm_lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])                          

            tm_src_dataset_list.append(tm_src_dataset)
            tm_tgt_dataset_list.append(tm_tgt_dataset)
            tm_lab_dataset_list.append(tm_lab_dataset)

        all_dataset = []
        all_dataset.append(src_dataset)
        all_dataset.extend(tm_src_dataset_list)
        all_dataset.extend(tm_tgt_dataset_list)
        all_dataset.extend(tm_lab_dataset_list)

        # dataset = Dataset.zip((src_dataset, tm_src_dataset, tm_tgt_dataset, tm_lab_dataset))
        dataset = Dataset.zip(tuple(all_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                    torch.distributed.get_rank())
        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            # src_seq, tm_src_seq, tm_tgt_seq, tm_lab_seq = inputs
            src_seq = inputs[0]
            src_seq = torch.tensor(src_seq)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            src_mask = src_mask.float()

            tm_src_seq_list = []
            tm_tgt_seq_list = []
            tm_lab_seq_list = []
            max_src_seq_len = 0
            max_tgt_seq_len = 0
            max_lab_seq_len = 0

            for idx in range(tm_count):
                tm_src_seq = torch.tensor(inputs[idx + 1])
                tm_tgt_seq = torch.tensor(inputs[tm_count + idx + 1])
                tm_lab_seq = torch.tensor(inputs[tm_count * 2 + idx + 1])
                tm_src_seq_list.append(tm_src_seq)
                tm_tgt_seq_list.append(tm_tgt_seq)
                tm_lab_seq_list.append(tm_lab_seq)
                max_src_seq_len = max(max_src_seq_len, tm_src_seq.size(1))
                max_tgt_seq_len = max(max_tgt_seq_len, tm_tgt_seq.size(1))
                max_lab_seq_len = max(max_lab_seq_len, tm_lab_seq.size(1))

            # [B,Tk,L]
            tm_src_seq = torch.ones((src_seq.size(0), tm_count, max_src_seq_len), dtype=src_seq.dtype) * src_vocab[params.pad]
            tm_tgt_seq = torch.ones((src_seq.size(0), tm_count, max_tgt_seq_len), dtype=src_seq.dtype) * tgt_vocab[params.pad]
            tm_lab_seq = torch.ones((src_seq.size(0), tm_count, max_lab_seq_len), dtype=src_seq.dtype) * tgt_vocab[params.pad]

            for idx in range(tm_count):
                tm_src_seq[:,idx,:tm_src_seq_list[idx].size(1)] = tm_src_seq_list[idx]
                tm_tgt_seq[:,idx,:tm_tgt_seq_list[idx].size(1)] = tm_tgt_seq_list[idx]
                tm_lab_seq[:,idx,:tm_lab_seq_list[idx].size(1)] = tm_lab_seq_list[idx]
            
            tm_src_seq = tm_src_seq.reshape([-1, max_src_seq_len]) # [ B * Tk, L ]
            tm_tgt_seq = tm_tgt_seq.reshape([-1, max_tgt_seq_len])
            tm_lab_seq = tm_lab_seq.reshape([-1, max_lab_seq_len])

            # tm_src_mask = tm_src_seq != params.vocabulary["source"][params.pad]
            # tm_src_mask = tm_src_mask.float()
            # tm_tgt_mask = torch.logical_and(tm_lab_seq != params.vocabulary["target"][params.pad], tm_lab_seq != params.vocabulary["target"][params.eos]) 
            # # tm_tgt_mask = tm_tgt_seq != params.vocabulary["target"][params.pad]
            # tm_tgt_mask = tm_tgt_mask.float()

            tm_src_mask = tm_src_seq != params.vocabulary["source"][params.pad]
            tm_src_mask = tm_src_mask.float()
            tm_tgt_mask = torch.logical_and(tm_lab_seq != params.vocabulary["target"][params.pad], tm_lab_seq != params.vocabulary["target"][params.eos]) 
            tm_tgt_mask = tm_tgt_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                tm_src_seq = tm_src_seq.cuda(params.device)
                tm_src_mask = tm_src_mask.cuda(params.device)
                tm_tgt_seq = tm_tgt_seq.cuda(params.device)
                tm_tgt_mask = tm_tgt_mask.cuda(params.device)
                tm_lab_seq = tm_lab_seq.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "tm_source": tm_src_seq,
                "tm_source_mask": tm_src_mask,
                "tm_target": tm_tgt_seq,
                "tm_target_mask": tm_tgt_mask,
                "tm_labels": tm_lab_seq
            }

            return features

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return sorted_keys, dataset
