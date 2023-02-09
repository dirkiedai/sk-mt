# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils
import thumt.modules as modules
from torch_scatter import scatter
from thumt.modules.knn_datastore import KNN_Dstore
import torch.nn.functional as functional


class AttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(AttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.attention = modules.MultiHeadAttention(
                params.hidden_size, params.num_heads, params.attention_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x, bias, memory=None, state=None, precompute_weights=None):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.training or state is None:
            y, weights = self.attention(y, bias, memory, None, precompute_weights)
        else:
            kv = [state["k"], state["v"]]
            y, weights, k, v = self.attention(y, bias, memory, kv, precompute_weights)
            state["k"], state["v"] = k, v

        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y, weights
        else:
            return self.layer_norm(x + y), weights


class FFNSubLayer(modules.Module):

    def __init__(self, params, filter_size,  dtype=None, name="ffn_layer"):
        super(FFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.ffn_layer = modules.FeedForward(params.hidden_size,
                                                 filter_size,
                                                 dropout=params.relu_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class TransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerEncoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params)
            self.feed_forward = FFNSubLayer(params, params.encoder_filter_size)

    def forward(self, x, bias):
        x, _ = self.self_attention(x, bias)
        x = self.feed_forward(x)
        return x


class TransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerDecoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params,
                                                    name="self_attention")
            self.encdec_attention = AttentionSubLayer(params,
                                                    name="encdec_attention")
            self.feed_forward = FFNSubLayer(params, params.decoder_filter_size)

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        x, _ = self.self_attention(x, attn_bias, state=state)
        x, _ = self.encdec_attention(x, encdec_bias, memory)
        x = self.feed_forward(x)
        return x


class TransformerFastDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerFastDecoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params,
                                                    name="self_attention")
            self.encdec_attention = AttentionSubLayer(params,
                                                    name="encdec_attention")
            self.feed_forward = FFNSubLayer(params)

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None, 
            self_attention_weights=None, pre_encdec_attention=None):

        # share self-attention weight for layer 2/3/4...
        x, weights = self.self_attention(x, attn_bias, state=state, precompute_weights=self_attention_weights)

        # share encoder-decoder attention weight for layer 2/3/4...
        if self.encdec_attention.normalization == "before":  
            if pre_encdec_attention is None:
                y = self.encdec_attention.layer_norm(x)
                encdec_attention_result, _ = self.encdec_attention.attention(y, encdec_bias, memory, None, None)
                encdec_attention_result = nn.functional.dropout(encdec_attention_result, self.encdec_attention.dropout, self.training)
            else:
                encdec_attention_result = pre_encdec_attention
            x = x + encdec_attention_result
        else:
            if pre_encdec_attention is None:
                encdec_attention_result, _ = self.encdec_attention.attention(x, encdec_bias, memory, None, None)
                encdec_attention_result = nn.functional.dropout(encdec_attention_result, self.encdec_attention.dropout, self.training)
            else:
                encdec_attention_result = pre_encdec_attention
            x = self.encdec_attention.layer_norm(x + encdec_attention_result)
        
        x = self.feed_forward(x)

        return x, weights, encdec_attention_result

class TransformerEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(TransformerEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layer_stack = nn.ModuleList([
                TransformerEncoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_encoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, bias):
        for layer in self.layer_stack:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class TransformerDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(TransformerDecoder, self).__init__(name=name)

        self.normalization = params.normalization
        self.fastmode = params.fastmode
        self.fast_layernum = params.fast_layernum



        with utils.scope(name):
            if self.fastmode:
                self.layer_stack = nn.ModuleList([
                    TransformerFastDecoderLayer(params, name="layer_%d" % i)
                    for i in range(params.num_decoder_layers)])
            else:
                self.layer_stack = nn.ModuleList([
                    TransformerDecoderLayer(params, name="layer_%d" % i)
                    for i in range(params.num_decoder_layers)])

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

        self.fp16 = params.fp16

        if params.load_knn_datastore and params.use_knn_datastore:
            self.knn_datastore = KNN_Dstore(params, len(params.vocabulary["target"]))

        self.use_knn_datastore = params.use_knn_datastore
        self.load_knn_datastore = params.load_knn_datastore
        self.knn_lambda_type = params.knn_lambda_type
        self.knn_temperature_type = params.knn_temperature_type
        self.knn_k_type = params.knn_k_type
        self.label_count_as_feature = params.label_count_as_feature
        self.relative_label_count = params.relative_label_count
        self.avg_k = params.avg_k

        if self.knn_lambda_type == "trainable" and self.knn_k_type == "trainable":

            # TODO another network to predict k and lambda at the same time without gumbel softmax
            self.retrieve_result_to_k_and_lambda = nn.Sequential(
                nn.Linear(params.max_k if not self.label_count_as_feature else params.max_k * 2,
                          params.k_lambda_net_hid_size),
                nn.Tanh(),
                nn.Dropout(p=params.k_lambda_net_dropout_rate),
                nn.Linear(params.k_lambda_net_hid_size, 2 + int(math.log(params.max_k, 2))),
                nn.Softmax(dim=-1),  # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 4 , 8 , ... , ]
            )

            nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, : params.k], gain=0.01)

            if self.label_count_as_feature:
                nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, params.k:], gain=0.1)

        else:
            if self.knn_lambda_type == 'trainable':
                # TODO, we may update the label count feature here
                self.knn_distances_to_lambda = nn.Sequential(
                    nn.Linear(params.k if not self.label_count_as_feature else params.k * 2, params.knn_lambda_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=params.knn_net_dropout_rate),
                    nn.Linear(params.knn_lambda_net_hid_size, 1),
                    nn.Sigmoid())

                if self.label_count_as_feature:

                    nn.init.xavier_normal_(self.knn_distances_to_lambda[0].weight[:, : params.k], gain=0.01)
                    nn.init.xavier_normal_(self.knn_distances_to_lambda[0].weight[:, params.k:], gain=0.1)
                    nn.init.xavier_normal_(self.knn_distances_to_lambda[-2].weight)

                else:
                    nn.init.normal_(self.knn_distances_to_lambda[0].weight, mean=0, std=0.01)

            if self.knn_temperature_type == 'trainable':
                # TODO, consider a reasonable function
                self.knn_distance_to_temperature = nn.Sequential(
                    nn.Linear(params.k + 2, params.knn_temperature_net_hid_size),
                    nn.Tanh(),
                    nn.Linear(params.knn_temperature_net_hid_size, 1),
                    nn.Sigmoid())
                # the weight shape is [net hid size, k + 1)
                nn.init.normal_(self.knn_distance_to_temperature[0].weight[:, :-1], mean=0, std=0.01)
                nn.init.normal_(self.knn_distance_to_temperature[0].weight[:, -1:], mean=0, std=0.1)

            # TODO we split the network here for different function, but may combine them in the future
            if self.knn_k_type == "trainable":

                self.knn_distance_to_k = nn.Sequential(
                    nn.Linear(params.max_k * 2 if self.label_count_as_feature else params.max_k,
                              params.knn_k_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=params.knn_k_net_dropout_rate),
                    # nn.Linear(params.knn_k_net_hid_size, params.max_k),
                    nn.Linear(params.knn_k_net_hid_size, params.max_k),
                    nn.Softmax(dim=-1))

                # nn.init.xavier_uniform_(self.knn_distance_to_k[0].weight, gain=0.01)
                # nn.init.xavier_uniform_(self.knn_distance_to_k[-2].weight, gain=0.01)
                # # TODO this maybe change or remove from here
                if self.label_count_as_feature:
                    nn.init.normal_(self.knn_distance_to_k[0].weight[:, :params.max_k], mean=0, std=0.01)
                    nn.init.normal_(self.knn_distance_to_k[0].weight[:, params.max_k:], mean=0, std=0.1)
                else:
                    nn.init.normal_(self.knn_distance_to_k[0].weight, mean=0, std=0.01)

    def forward(self, x, attn_bias, encdec_bias, memory, state=None, features_only=False):

        self_attention_weights = None
        pre_encdec_attention = None

        for i, layer in enumerate(self.layer_stack):
            if self.fastmode:
                if i % self.fast_layernum == 0:
                    x, self_attention_weights, pre_encdec_attention = layer(x, attn_bias, encdec_bias, memory,
                        state["decoder"]["layer_%d" % i] if state is not None else None)
                else:
                    x, _, _ = layer(x, attn_bias, encdec_bias, memory, state["decoder"]["layer_%d" % i] if state is not None else None, 
                        self_attention_weights, pre_encdec_attention)
            else:  
                x = layer(x, attn_bias, encdec_bias, memory,
                            state["decoder"]["layer_%d" % i] if state is not None else None)

        if self.normalization == "before":
            x = self.layer_norm(x)
        
        if features_only:
            return x

        if self.use_knn_datastore and self.load_knn_datastore:
            last_hidden = x
            # we should return the prob of knn search
            knn_search_result = self.knn_datastore.retrieve(last_hidden)
            # knn_probs = knn_search_result['prob']
            knn_dists = knn_search_result['distance']  # [batch, seq len, k]  # we need do sort
            knn_index = knn_search_result['knn_index']
            tgt_index = knn_search_result['tgt_index']

            if self.label_count_as_feature:
                # TODO, we get the segment label count here, which is conflict with previous experiment
                label_counts = self.knn_datastore.get_label_count_segment(tgt_index, relative=self.relative_label_count)
                network_inputs = torch.cat((knn_dists.detach(), label_counts.detach().float()), dim=-1)
            else:
                network_inputs = knn_dists.detach()

            if self.fp16:
                network_inputs = network_inputs.half()

            if self.knn_temperature_type == 'trainable':
                knn_temperature = None
            else:
                knn_temperature = self.knn_datastore.get_temperature() 

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
                                                                             is_test=not self.retrieve_result_to_k_and_lambda.training)

            else:
                if self.knn_lambda_type == 'trainable':
                    # self.knn_distances_to_lambda[2].p = 1.0

                    knn_lambda = self.knn_distances_to_lambda(network_inputs)

                else:
                    knn_lambda = self.knn_datastore.get_lambda() * torch.ones(x.size(0), x.size(1), device=x.device).unsqueeze(-1)

                if self.knn_k_type == "trainable":
                    # we should generate k mask
                    k_prob = self.knn_distance_to_k(network_inputs)

                    if self.knn_distance_to_k.training:
                        k_log_prob = torch.log(k_prob)
                        k_soft_one_hot = functional.gumbel_softmax(k_log_prob, tau=0.1, hard=False, dim=-1)

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
                                                                                 k_soft_one_hot)

                elif self.knn_k_type == "trainable":
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden,
                                                                          knn_temperature, knn_mask)

                else:
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden,
                                                                          knn_temperature)

            knn_prob = decode_result['prob']

            return x, knn_prob, knn_lambda, knn_dists, knn_index

        else:
            # original situation
            return x

class Transformer(modules.Module):

    def __init__(self, params, name="transformer"):
        super(Transformer, self).__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = TransformerEncoder(params)
            self.decoder = TransformerDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
        self.knn_t = params.knn_temperature_value
        self.knn_lambda_t = params.knn_t
        self.knn_k = params.k
        self.tvoc_size = len(params.vocabulary["target"])
        self.reset_parameters()

    def build_embedding(self, params):
        svoc_size = len(params.vocabulary["source"])
        tvoc_size = len(params.vocabulary["target"])

        if params.shared_source_target_embedding and svoc_size != tvoc_size:
            raise ValueError("Cannot share source and target embedding.")

        if not params.shared_embedding_and_softmax_weights:
            self.softmax_weights = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.softmax_weights, "softmax_weights")

        if not params.shared_source_target_embedding:
            self.source_embedding = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.target_embedding = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.source_embedding, "source_embedding")
            self.add_name(self.target_embedding, "target_embedding")
        else:
            self.weights = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.add_name(self.weights, "weights")

        # self.bias = torch.nn.Parameter(torch.zeros([params.hidden_size]))
        # self.add_name(self.bias, "bias")

    @property
    def src_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.source_embedding

    @property
    def tgt_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.target_embedding

    @property
    def softmax_embedding(self):
        if not self.params.shared_embedding_and_softmax_weights:
            return self.softmax_weights
        else:
            return self.tgt_embedding

    def reset_parameters(self):
        nn.init.normal_(self.src_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.tgt_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)

        if not self.params.shared_embedding_and_softmax_weights:
            nn.init.normal_(self.softmax_weights, mean=0.0,
                            std=self.params.hidden_size ** -0.5)
    
    def get_tm_datastore(self, features, batch_size):
        tm_src_seq = features["tm_source"]
        tm_src_mask = features["tm_source_mask"]
        tm_tgt_seq = features["tm_target"]
        tm_tgt_mask = features["tm_target_mask"]
        tm_lab_seq = features["tm_labels"]

        # tm-encoder
        enc_attn_bias = self.masking_bias(tm_src_mask)
        inputs = torch.nn.functional.embedding(tm_src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout, self.training)
        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output = self.encoder(inputs, enc_attn_bias)

        # tm-decoder
        dec_attn_bias = self.causal_bias(tm_tgt_seq.shape[1])
        targets = torch.nn.functional.embedding(tm_tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)
        decoder_input = nn.functional.dropout(self.encoding(targets),
                                              self.dropout, self.training)
        dec_attn_bias = dec_attn_bias.to(targets)
        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      enc_attn_bias, encoder_output, None, features_only=True)
                                      
        keys = decoder_output.reshape([batch_size, -1, self.hidden_size])  # [B, Tk * L, H]
        keys_mask = ((1.0 - tm_tgt_mask) * -1e9).to(keys).reshape([batch_size, -1]) # [B, Tk * L]
        values = tm_lab_seq.reshape([batch_size, -1])    # [B, L]

        # self.decoder.knn_datastore.add_entry(keys, values) 
        init_knn_prob = torch.zeros([values.size(0), self.knn_k if keys.size(1) > self.knn_k else keys.size(1), self.tvoc_size],
                    dtype=torch.float32, device=keys.device)  # [B, K, V]

        return keys, keys_mask, values, init_knn_prob

    def get_knn_prob(self, queries, keys, keys_mask, values, init_knn_prob):
        # queries [B, H]
        dists = ((keys - queries.unsqueeze(1)) ** 2).sum(-1) # [B, L]
        scaled_dists = -1.0 / self.knn_t * dists  + keys_mask
        top_dists, top_indices = torch.topk(scaled_dists, 
                self.knn_k if keys.size(1) >= self.knn_k else keys.size(1)) # [B, K]
        top_values = torch.gather(values, 1, top_indices)
        knn_weight = torch.softmax(top_dists, dim=-1).unsqueeze(-1)  # [B, K, 1]
        knn_mask = torch.nn.functional.relu(1.0 + top_dists[:,0])

        # init knn-prob
        knn_tgt_prob = 0 * init_knn_prob

        # implemented with pytorch_scatter
        if queries.dtype == torch.float16:
            scatter(src=knn_weight.float(), out=knn_tgt_prob, index=top_values, dim=-1)
            prob = knn_tgt_prob.sum(dim=-2).half()  # [B, V]
            knn_mask = knn_mask.half()
        else:
            scatter(src=knn_weight, out=knn_tgt_prob, index=top_values, dim=-1)
            prob = knn_tgt_prob.sum(dim=-2)
            knn_mask = knn_mask.float()

        return prob, knn_mask

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]

        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout,
                                       self.training)


        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output = self.encoder(inputs, enc_attn_bias)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        if not self.decoder.load_knn_datastore and "tm_source" in features.keys():
            keys, keys_mask, values, init_knn_prob = self.get_tm_datastore(features, src_seq.size()[0])
            state["ds_keys"] = keys
            state["ds_keys_mask"] = keys_mask
            state["ds_values"] = values
            state["ds_init_knn_prob"] = init_knn_prob
            
        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        # decoder_input = torch.cat(
        #     [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
        #      targets[:, 1:, :]], dim=1)

        decoder_input = nn.functional.dropout(self.encoding(targets),
                                              self.dropout, self.training)

        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets)

        if mode == "infer":
            decoder_input = decoder_input[:, -1:, :]
            dec_attn_bias = dec_attn_bias[:, :, -1:, :]

        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      enc_attn_bias, encoder_output, state)

        if self.decoder.load_knn_datastore and self.decoder.use_knn_datastore:
            decoder_output, knn_prob, knn_lambda, knn_dists, knn_index = decoder_output
            
            # knn_lambda = torch.nn.functional.relu(1.0 - knn_dists[:,:,0:1] / self.knn_lambda_t)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output_T = torch.transpose(decoder_output, -1, -2)

        logits = torch.matmul(self.softmax_embedding, decoder_output_T)
        # logits = torch.transpose(logits, 0, 1).unsqueeze(1)
        logits = torch.transpose(logits, 0, 1)

        if self.decoder.use_knn_datastore:
            if self.decoder.load_knn_datastore:
                pass
                #knn-mt & ak-mt
                logits = logits.unsqueeze(1)
                
            else:
                #sk-mt
                keys = state["ds_keys"]
                keys_mask = state["ds_keys_mask"]
                values = state["ds_values"]
                init_knn_prob = state["ds_init_knn_prob"]
                knn_prob, knn_mask = self.get_knn_prob(decoder_output, keys, keys_mask, values, init_knn_prob) 
                knn_lambda = knn_mask.unsqueeze(-1)
                

            final_prob = knn_lambda * knn_prob + (1.0 - knn_lambda) * torch.softmax(logits, dim=-1)
        else:
            final_prob = torch.softmax(logits, dim=-1)

        logits = torch.log(final_prob + 1e-9).squeeze(1)

        return logits, state

    def forward(self, features, labels, mode="train", level="sentence"):
        mask = features["target_mask"]

        state = self.empty_state(features["target"].shape[0],
                                 labels.device)
        state = self.encode(features, state)
        logits, _ = self.decode(features, state, mode=mode)
        loss = self.criterion(logits, labels)
        mask = mask.to(torch.float32)

        # Prevent FP16 overflow
        if loss.dtype == torch.float16:
            loss = loss.to(torch.float32)

        if mode == "eval":
            if level == "sentence":
                return -torch.sum(loss * mask, 1)
            else:
                return torch.exp(-loss) * mask - (1 - mask)

        return (torch.sum(loss * mask) / torch.sum(mask)).to(logits)

    def get_decoder_feature(self, features, labels):

        src_seq = features["source"]
        src_mask = features["source_mask"]
        tgt_seq = features["target"]
        
        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout,
                                       self.training)

        enc_attn_bias = self.masking_bias(src_mask)
        enc_attn_bias = enc_attn_bias.to(inputs)

        encoder_output = self.encoder(inputs, enc_attn_bias)

        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)
        decoder_input = nn.functional.dropout(self.encoding(targets),
                                              self.dropout, self.training)

        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])
        dec_attn_bias = dec_attn_bias.to(targets)

        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      enc_attn_bias, encoder_output, None)

        return decoder_output

    def empty_state(self, batch_size, device):
        state = {
            "decoder": {
                "layer_%d" % i: {
                    "k": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device)
                } for i in range(self.num_decoder_layers)
            }
        }

        return state

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="</s>",
            eos="</s>",
            unk="<unk>",
            hidden_size=512,
            encoder_filter_size=2048,
            decoder_filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            normalization="after",
            fastmode=False,
            fast_layernum=6,  
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
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
            knn_k = 1,
            knn_t = 10,
        )

        return params

    @staticmethod
    def base_params_v2():
        params = Transformer.base_params()
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 8000
        params.normalization = "before"
        params.adam_beta2 = 0.997

        return params

    @staticmethod
    def deep_params():
        params = Transformer.base_params()
        params.num_encoder_layers = 50
        params.num_decoder_layers = 2
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.residual_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 20000
        params.normalization = "before"
        params.shared_embedding_and_softmax_weights = True

        return params

    @staticmethod
    def big_params():
        params = Transformer.base_params()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 5e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def big_params_v2():
        params = Transformer.base_params_v2()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 7e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def transmart_zh2en():
        params = Transformer.base_params()
        params.num_encoder_layers = 40
        params.num_decoder_layers = 6
        params.hidden_size = 1024
        params.filter_size = 2048
        params.warmup_steps = 20000
        params.normalization = "before"
        params.train_steps = 1000000
        params.learning_rate = 12e-4
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.residual_dropout = 0.1
        params.shared_embedding_and_softmax_weights = True

        return params

    @staticmethod
    def transmart_zh2en_big():
        params = Transformer.transmart_zh2en()
        params.num_heads = 16
        params.filter_size = 4096
        
        return params

    @staticmethod
    def default_params(name=None):
        if name == "base":
            return Transformer.base_params()
        elif name == "base_v2":
            return Transformer.base_params_v2()
        elif name == "big":
            return Transformer.big_params()
        elif name == "big_v2":
            return Transformer.big_params_v2()
        elif name == "transmart_zh2en":
            return Transformer.transmart_zh2en()
        elif name == "transmart_zh2en_big":
            return Transformer.transmart_zh2en_big()
        elif name == "deep":
            return Transformer.deep_params()
        else:
            return Transformer.base_params()
