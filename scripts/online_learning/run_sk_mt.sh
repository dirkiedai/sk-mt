run_path=.
data_path=
model_path=
output_path=
mkdir -p $output_path

device_id=0
export PYTHONPATH=$PYTHONPATH:${run_path}

bs=4
lpv=0.6
tm=32
knn_T=100
knn_k=2

CUDA_VISIBLE_DEVICES=$device_id python3 $run_path/thumt/bin/translator.py \
    --input $data_path/50/dev.input.clean \
    --output $data_path/outputs/50.dev.sknn.tm$tm.k$topk.T$knn_T.output  \
    --vocabulary $model_path/dict.de.txt $model_path/dict.en.txt  \
    --models transformer --checkpoints $model_path --half  \
    --parameters=beam_size=${bs},device_list=[0],decode_batch_size=32,decode_alpha=${lpv},tm_count=${tm},knn_k=${knn_k},knn_t=${knn_T}
