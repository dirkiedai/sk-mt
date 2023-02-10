import argparse
import os
import numpy as np
import faiss
import time
import torch



def train_faiss_index(args, keys, vals, dstore_size, use_gpu, save=True):

 
    if not os.path.exists(args.faiss_index + ".trained") or not save:
        # Initialize faiss index

        index_dim = args.pca if args.pca > 0 else args.dimension

        quantizer = faiss.IndexFlatL2(index_dim)
        index = faiss.IndexIVFPQ(quantizer, index_dim,
                                args.ncentroids, args.code_size, 8)
        index.nprobe = args.probe

        if args.pca > 0:
            pca_matrix = faiss.PCAMatrix(args.dimension, args.pca, 0, True)
            index = faiss.IndexPreTransform(pca_matrix, index)

        # TODO, we may remove useFloat16 when the GPU satisfy the condition
        if use_gpu:
            print('Start put index to gpu')
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

        print('Training Index')
        np.random.seed(args.seed)
        random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, dstore_size)], replace=False)
        start = time.time()
        # Faiss does not handle adding keys in fp16 as of writing this.
        # index.train(keys[random_sample].astype(np.float32))

        index.train(keys[random_sample].astype(np.float32))
        print('Training took {} s'.format(time.time() - start))

        index = faiss.index_gpu_to_cpu(index) if use_gpu else index
        
        if save:
            print('Writing index after training')
            start = time.time()
            faiss.write_index(index, args.faiss_index + ".trained")
            print('Writing index took {} s'.format(time.time() - start))

    print('Adding Keys')
    if os.path.exists(args.faiss_index + ".trained"):
        index = faiss.read_index(args.faiss_index + ".trained")

    if use_gpu:
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

    start = args.starting_point
    start_time = time.time()
    while start < dstore_size:
        end = min(dstore_size, start + args.num_keys_to_add_at_a_time)
        to_add = keys[start:end].copy()
        index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
        start += args.num_keys_to_add_at_a_time

        if (start % 1000000) == 0:
            print('Added %d tokens so far' % start)
            if save:
                print('Writing Index', start)
                faiss.write_index(faiss.index_gpu_to_cpu(index) if use_gpu else index, args.faiss_index)


    print("Adding total %d keys" % end)
    print('Adding took {} s'.format(time.time() - start_time))
    
    if save:
        print('Writing Index')
        start = time.time()
        faiss.write_index(faiss.index_gpu_to_cpu(index) if use_gpu else index, args.faiss_index)
        print('Writing index took {} s'.format(time.time() - start))

    return index

if __name__ == "__main__":
    # the implementation refers to knnlm

    parser = argparse.ArgumentParser()
    parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
    parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
    parser.add_argument('--store_dstore_mmap', type=str, help='memmap where keys and vals are stored')
    parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
    parser.add_argument('--dstore_fp16', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for sampling the subset of vectors to train the cache')
    parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
    parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
    parser.add_argument('--probe', type=int, default=32, help='number of clusters to query')
    parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
    parser.add_argument('--num_keys_to_add_at_a_time', default=500000, type=int,
                        help='can only load a certain amount of data to memory at a time.')
    parser.add_argument('--starting_point', type=int, default=0, help='index to start adding keys at')
    parser.add_argument('--load-multiple-files', default=False, action='store_true')
    parser.add_argument('--multiple-key-files', type=str, default=None)
    parser.add_argument('--multiple-val-files', type=str, default=None)
    parser.add_argument('--multiple-files-size', type=str, default=None)
    parser.add_argument('--concat-file-path', type=str, default=None)

    parser.add_argument('--use-gpu', default=False, action='store_true')
    parser.add_argument('--load-to-mem', default=False, action='store_true')

    parser.add_argument('--cutoff', type=int, default=0, help='top cut off')
    parser.add_argument('--topk', default=False, action='store_true')
    parser.add_argument('--pca', type=int, default=0, help='pca dimension')
    args = parser.parse_args()

    print(args)

    # load the saved keys and values
    if args.dstore_fp16:
        if args.load_multiple_files:
            assert args.multiple_key_files is not None and args.multiple_val_files is not None
            key_files = args.multiple_key_files.split(':')
            val_files = args.multiple_val_files.split(':')
            sizes = [int(size) for size in args.multiple_files_size.split(':')]
            print(sizes)
            key_list = [np.memmap(key_file, dtype=np.float16, mode='r', shape=(sizes[idx], args.dimension)) for
                        idx, key_file in enumerate(key_files)]
            val_list = [np.memmap(val_file, dtype=np.int, mode='r', shape=(sizes[idx], 1)) for idx, val_file in
                        enumerate(val_files)]
            concat_size = np.sum(sizes)

            keys = np.memmap(args.concat_file_path + '/keys.npy', dtype=np.float16, mode='w+',
                            shape=(concat_size, args.dimension))
            vals = np.memmap(args.concat_file_path + '/vals.npy', dtype=np.int, mode='w+', shape=(concat_size, 1))

            cur_size = 0
            for idx, size in enumerate(sizes):
                print('write {} to {}'.format(cur_size, cur_size + size))
                keys[cur_size: cur_size + size, :] = key_list[idx][:, :]
                vals[cur_size: cur_size + size, :] = val_list[idx][:, :]
                cur_size += size

            exit()

    if args.dstore_fp16:
        print('load dstore fp16', args.dstore_size, args.dimension)

    if args.load_to_mem:
        print('load keys and vals to mem')
        keys_from_memmap = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                            shape=(args.dstore_size, args.dimension))
        vals_from_memmap = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

        keys = np.zeros(keys_from_memmap.shape, dtype = keys_from_memmap.dtype)
        vals = np.zeros(vals_from_memmap.shape, dtype = vals_from_memmap.dtype)

        keys[:] = keys_from_memmap[:]
        vals[:] = vals_from_memmap[:]

        del keys_from_memmap
        del vals_from_memmap
    else:

        keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                shape=(args.dstore_size, args.dimension))
        vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

    print('done.')

    if args.cutoff:
        unique_vals = np.unique(vals[vals != 0])
        counts = np.array([np.sum(vals == val) for val in unique_vals])

        if not args.topk:
            cutoff_vals = unique_vals[np.where(counts < args.cutoff)]
            print("dictionary overlap: ", cutoff_vals.shape[0] / unique_vals.shape[0])
            print("datastore overlap: ", vals[np.isin(vals, cutoff_vals)].shape[0] / vals.shape[0])
        else:
            #filter special token, e.g. <bos>,<eos>,<unk>
            filter_vals = unique_vals[np.where(unique_vals > 4)]
            filter_counts = counts[np.where(unique_vals > 4)]
            cutoff_vals = filter_vals[filter_counts.argsort()[-args.cutoff:]]

            print("dictionary overlap: ", cutoff_vals.shape[0] / unique_vals.shape[0])
            print("datastore overlap: ", vals[np.isin(vals, cutoff_vals)].shape[0] / vals.shape[0])

        indice = ~np.isin(vals, cutoff_vals).flatten() & (vals != 0).flatten()
        vals = vals[indice]

        dstore_size = vals.shape[0]
        print("datastore size: ", dstore_size)

        if args.store_dstore_mmap:
            if args.dstore_fp16:

                vals_map = np.memmap(args.store_dstore_mmap+'/vals.npy', dtype=np.int, mode='w+', shape=vals.shape)
                vals_map[:] = vals
                vals_map.flush()
    else:
        dstore_size = args.dstore_size

    use_gpu = args.use_gpu and torch.cuda.is_available()


    train_faiss_index(args, keys, vals, dstore_size, use_gpu)