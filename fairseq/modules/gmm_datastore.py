import torch
import faiss
import numpy as np
from torch_scatter import scatter
import time
import math
import faiss.contrib.torch_utils
from tqdm import tqdm


class GMM_Dstore(object):

    def __init__(self, args, trg_vocab_size):

        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        # self.temperature = args.knn_temperature
        self.use_gpu_to_search = args.use_gpu_to_search
        self.vocab_size = trg_vocab_size
        self.no_load_keys = args.no_load_keys
        self.move_dstore_to_mem = args.move_dstore_to_mem
        # self.only_use_max_idx = args.only_use_max_idx
    

        # self.index = self.setup_dstore(args)
        self.setup_distribution(args)
        self.time_for_retrieve = 0.
        self.retrieve_count = 0.
        self.time_for_setup_prob = 0.

        # set lambda
        self.set_lambda(args)

        # set temperature
        self.temperature_type = args.knn_temperature_type
        if self.temperature_type == 'fix':
            self.temperature = args.knn_temperature_value
        elif self.temperature_type == 'trainable':
            self.temperature = None
        else:
            self.temperature = None

        self.k_type = args.knn_k_type
        if self.k_type == 'fix':
            self.k = args.k

        elif self.k_type == 'trainable':

            assert args.max_k is not None

            self.max_k = args.max_k
            self.k = args.max_k
            # we first need to generate the mask for different k

            self.mask_for_distance = self.generate_neighbor_mask(args.max_k if args.max_k is not None else args.k)
            self.reduce_k = self.mask_for_distance.size(0)

        self.mask_for_label_count = self.generate_label_count_mask(args.max_k if args.max_k is not None else args.k)

    def generate_neighbor_mask(self, max_k):

        # [1, 1000, 1000]
        # [1, 1,    1000]
        # [1, 1,    1   ]
        k_mask = torch.empty((max_k, max_k)).fill_(999.)
        k_mask = torch.triu(k_mask, diagonal=1) + 1

        # we only select 2's power here
        # [1 - 1, 2 - 1, 4 - 1, 8 - 1, ...]
        power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(self.max_k, 2)) + 1)])
        k_mask = k_mask[power_index]

        k_mask.requires_grad = False
        if torch.cuda.is_available():
            k_mask = k_mask.cuda()

        # random_idx = torch.randperm(self.k_mask.size(0))
        # print(random_idx)
        # random_idx = [4, 3, 0, 1, 2]
        # self.k_mask = self.k_mask[random_idx]

        return k_mask

    def generate_label_count_mask(self, max_k):

        # [0, 1, 1]
        # [0, 0, 1]
        # [0, 0, 0]
        mask_for_label_count = torch.empty((max_k, max_k)).fill_(1)
        mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()

        if torch.cuda.is_available():
            mask_for_label_count = mask_for_label_count.cuda()

        mask_for_label_count.requires_grad = False

        return mask_for_label_count

    def get_label_count_segment(self,
                                tgt_idx: torch.Tensor,
                                relative=False):  # [B, S, K]
        """
        This function return the label counts for different range of k nearest neighbor
        [[0:0], [0:1], [0:2], ..., [0:K-1]]

        """

        B, S, K = tgt_idx.size()

        expand_tgt_idx = tgt_idx.unsqueeze(-2).expand(B, S, K, K)
        expand_tgt_idx = expand_tgt_idx.masked_fill(self.mask_for_label_count, value=-1)

        labels_sorted, _ = expand_tgt_idx.sort(dim=-1)  # [B, S, K, K]
        labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, :, :-1]) != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)  # [B, S, K]
        retrieve_label_counts[:, :, :-1] -= 1

        # if we want relative label count, i.e [1, 2, 3, 3, 4] -> [1, 1, 1, 0, 1]
        if relative:
            retrieve_label_counts[:, :, 1:] = retrieve_label_counts[:, :, 1:] - retrieve_label_counts[:, :, :-1]

        return retrieve_label_counts

    def get_label_count(self, tgt_idx: torch.Tensor):
        """
        This only return total label count for all neighbors
        """
        tgt_sorted, _ = tgt_idx.sort(dim=-1)
        tgt_sorted[:, :, 1:] *= ((tgt_sorted[:, :, 1:] - tgt_sorted[:, :, :-1]) != 0).long()
        retrieve_label_counts = tgt_sorted.ne(0).sum(-1).unsqueeze(-1)  # [B, S, 1]

        return retrieve_label_counts

    def set_lambda(self, args):

        if not hasattr(args, 'knn_lambda_type'):
            return

        self.lambda_type = args.knn_lambda_type

        if self.lambda_type == 'fix':
            self.lambda_value = args.knn_lambda_value

        if self.lambda_type == 'trainable':
            self.lambda_value = None  # not generate lambda value in this class

    def get_lambda(self, step=None, distance=None):

        if self.lambda_type == 'fix':

            return self.lambda_value

        elif self.lambda_type == 'trainable':

            return None

    def get_temperature(self):

        if self.temperature_type == 'fix':
            return self.temperature
        else:
            return None

    def setup_distribution(self, args):

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int32')   
            self.dist_keys = np.memmap(args.dstore_filename + '/dist_keys.npy', dtype=np.float16, mode='r',
                                    shape=(self.vocab_size, self.dimension * 2))
        else:
            print('Keys are fp32 and vals are int32')
            self.dist_keys = np.memmap(args.dstore_filename + '/dist_keys.npy', dtype=np.float32, mode='r',
                                    shape=(self.vocab_size, self.dimension * 2))

        self.dist_vals = np.memmap(args.dstore_filename + '/dist_vals.npy', dtype=np.int, mode='r',
                                shape=(self.vocab_size, 1))
        self.dist_counts = np.memmap(args.dstore_filename + '/dist_counts.npy', dtype=np.int, mode='r',
                                shape=(self.vocab_size, 1))

        self.indice = np.where(self.dist_counts > 1)[0]


        self.dist_vals = self.dist_vals[self.indice]
        self.dist_keys = self.dist_keys[self.indice]

        self.dist_keys[:, self.dimension: ][self.dist_keys[:, self.dimension: ] == 0] = 0.0001
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            # if not args.no_load_dist_keys:
            #     del self.dist_keys
            #     self.keys_from_memmap = np.memmap(args.dstore_filename + '/dist_keys.npy',
            #                                     dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
            #                                     shape=(self.vocab_size, self.dimension))
            #     self.dist_keys = np.zeros((self.vocab_size, self.dimension),
            #                         dtype=np.float16 if args.dstore_fp16 else np.float32)
            #     self.dist_keys = self.keys_from_memmap[:]
            #     self.dist_keys = self.dist_keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.dist_keys
            self.keys_from_memmap = np.memmap(args.dstore_filename + '/dist_keys.npy',
                                            dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                            shape=(self.vocab_size, self.dimension * 2))
            self.dist_keys = np.zeros((self.vocab_size, self.dimension * 2),
                                dtype=np.float16 if args.dstore_fp16 else np.float32)
            self.dist_keys = self.keys_from_memmap[self.indice]
            
            self.dist_keys[:, self.dimension: ][self.dist_keys[:, self.dimension: ] == 0] = 0.0001
            self.dist_keys = self.dist_keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.dist_vals
            self.vals_from_memmap = np.memmap(args.dstore_filename + '/dist_vals.npy',
                                            dtype=np.int, mode='r',
                                            shape=(self.vocab_size, 1))
            self.dist_vals = np.zeros((self.vocab_size, 1), dtype=np.int)
            self.dist_vals = self.vals_from_memmap[self.indice]
            self.dist_vals = self.dist_vals.astype(np.int)

            del self.dist_counts
            self.counts_from_memmap = np.memmap(args.dstore_filename + '/dist_counts.npy',
                                            dtype=np.int, mode='r',
                                            shape=(self.vocab_size, 1))
            self.dist_counts = np.zeros((self.vocab_size, 1), dtype=np.int)
            self.dist_counts = self.counts_from_memmap[self.indice]
            self.dist_counts = self.dist_counts.astype(np.int)

            if self.use_gpu_to_search:
                print('put vals and counts to gpu')
                self.dist_counts = torch.from_numpy(self.dist_counts)
                self.dist_vals = torch.from_numpy(self.dist_vals)
                if torch.cuda.is_available():
                    self.dist_counts = self.dist_counts.cuda()
                    self.dist_vals = self.dist_vals.cuda()

                print('put keys to gpu')
                self.dist_keys = torch.from_numpy(self.dist_keys)
                if torch.cuda.is_available():
                    self.dist_keys = self.dist_keys.cuda()

                self.indice = torch.from_numpy(self.indice)
                if torch.cuda.is_available():
                    self.indice = self.indice.cuda()

    def setup_dstore(self, args):

        if args.load_knn_datastore:
            if not args.dstore_filename:
                raise ValueError('Cannot build a datastore without the data.')

            if args.dstore_fp16:
                print('Keys are fp16 and vals are int32')
                if not args.no_load_keys:
                    self.keys = np.memmap(args.dstore_filename + '/keys.npy', dtype=np.float16, mode='r',
                                        shape=(self.dstore_size, self.dimension))
                self.vals = np.memmap(args.dstore_filename + '/vals.npy', dtype=np.int, mode='r',
                                    shape=(self.dstore_size, 1))
            else:
                print('Keys are fp32 and vals are int32')
                if not args.no_load_keys:
                    self.keys = np.memmap(args.dstore_filename + '/keys.npy', dtype=np.float32, mode='r',
                                        shape=(self.dstore_size, self.dimension))

                self.vals = np.memmap(args.dstore_filename + '/vals.npy', dtype=np.int, mode='r',
                                    shape=(self.dstore_size, 1))
            self.dstore_idx = None
                    # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
            if args.move_dstore_to_mem:
                print('Loading to memory...')
                start = time.time()

                if not args.no_load_keys:
                    del self.keys
                    self.keys_from_memmap = np.memmap(args.dstore_filename + '/keys.npy',
                                                    dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                                    shape=(self.dstore_size, self.dimension))
                    self.keys = np.zeros((self.dstore_size, self.dimension),
                                        dtype=np.float16 if args.dstore_fp16 else np.float32)
                    self.keys = self.keys_from_memmap[:]
                    self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

                del self.vals
                self.vals_from_memmap = np.memmap(args.dstore_filename + '/vals.npy',
                                                dtype=np.int, mode='r',
                                                shape=(self.dstore_size, 1))
                self.vals = np.zeros((self.dstore_size, 1), dtype=np.int)
                self.vals = self.vals_from_memmap[:]
                self.vals = self.vals.astype(np.int)

                if self.use_gpu_to_search:
                    self.vals = torch.from_numpy(self.vals)
                    if torch.cuda.is_available():
                        print('put vals to gpu')
                        self.vals = self.vals.cuda()

                print('Loading to memory took {} s'.format(time.time() - start))

    def get_prior(self, queries):
        if not hasattr(self, 'prior'):
            self.prior_prob = self.dist_counts / self.dist_counts.sum()
            if isinstance(self.prior_prob, np.ndarray):
                self.prior_prob = torch.from_numpy(self.prior_prob)
        self.prior_prob = self.prior_prob.view(1,1,-1)

        if self.use_gpu_to_search:
            self.prior_prob = self.prior_prob.cuda()
        
        return torch.log(self.prior_prob)
        
        
    def get_posterior(self, queries: torch.Tensor): # [Beam, Batch, Hidden]

        queries = queries.unsqueeze(-2)

        if not self.move_dstore_to_mem:
            queries = queries.detach().cpu().numpy()
        

        power = - (queries - self.dist_keys[:, :self.dimension]) ** 2 / (2 * self.dist_keys[:, self.dimension: ] ** 2) 
        power = power.sum(-1)
        
        log_coef = - torch.log(self.dist_keys[:, self.dimension:]).sum(-1)

        posterior = power + log_coef - self.dimension * np.log(2 * np.pi)

        return posterior
    
        
    def calculate_prob(self, queries, temperature):
        beam_size, batch_size, hidden_size = queries.size()
        prob = self.get_prior(queries) + self.get_posterior(queries)

        scaled_prob = torch.softmax(prob / temperature, dim=-1).to(queries.device)

        del prob
        prob = torch.zeros(beam_size, batch_size, self.vocab_size).to(queries.device)  # [B, S, K, Vocab Size]

        # # implemented with pytorch_scatter
        scatter(src=scaled_prob.float(), out=prob, index=self.indice.view(1,1,-1), dim=-1)
        
        return prob


if __name__ == "__main__":
    class ARGS:
        fp16 = False
        decoder_embed_dim = 1024
        k = 2
        dstore_size = 3613350
        faiss_metric_type = 'do_not_recomp_l2'
        knn_sim_func = 'do_not_recomp_l2'
        dstore_fp16 = True
        knn_temperature = 1.0
        indexfile = ''
        dstore_filename = 'interactive-mt/datastore/it/distribute'
        no_load_keys = False
        probe = 32
        move_dstore_to_mem = True
        use_gpu_to_search = True
        trg_vocab_size = 42024
        load_knn_datastore = True
        knn_temperature_type = 'fix'
        knn_temperature_value = 10
        knn_k_type = 'fix'
        knn_k_value = 4
        max_k = 32
        filter_once = True
        log_prob = True
        no_load_dist_keys = False
        


    args = ARGS()
    knn_store = GMM_Dstore(args=args, trg_vocab_size=args.trg_vocab_size)

    query = torch.randn(3, 2, args.decoder_embed_dim).cuda()

    target = torch.arange(15)
    print('query size is {}', query.size())
    knn_store.get_posterior(query)
    knn_store.calculate_prob(query, args.knn_temperature_value)
    # print(dist.shape)  # [10000, 64]
    # print(knn_idx.shape)  # [10000, 64]
    knn_store.add_entry(args, query, target)
    knn_store.get_knns(torch.tensor([4,0.5,0.6]).view(1,-1))

    query = torch.randn(15, args.decoder_embed_dim)
    query[:, 0] += torch.arange(query.size(0))

    target = torch.arange(15)
    knn_store.add_entry(args, query, target)
    knn_store.get_knns(torch.tensor([4,0.5,0.6]).view(1,-1))

    # print(prob.max(dim=-1)[0])
    # print(prob.max(dim=-1)[1])

    print('average time for retrieve neighbors, {} s'.format(knn_store.time_for_retrieve / knn_store.retrieve_count))
    print('average time for set the target prob for each neighbor'
          ' (need do scatter operation for (batch size * beam size * k, vocab size) tensor), {} s'
          .format(knn_store.time_for_setup_prob / knn_store.retrieve_count))