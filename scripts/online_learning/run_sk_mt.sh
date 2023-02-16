PROJECT_PATH=.
DATA_PATH=/path/to/data
MODEL_PATH=/path/to/model

device_id=0
export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}

bs=4
lpv=0.6
tm=32
knn_T=100
knn_k=2
bucket=50

OUTPUT_PATH=$PROJECT_PATH/output/online_learning/$bucket
mkdir -p $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=$device_id python3 $PROJECT_PATH/thumt/bin/translator.py \
    --input $DATA_PATH/online_learning/$bucket/dev.input.clean \
    --output $output_path/$bucket.dev.sknn.tm$tm.k$topk.T$knn_T.output  \
    --vocabulary $MODEL_PATH/dict.de.txt $MODEL_PATH/dict.en.txt  \
    --models transformer --checkpoints $MODEL_PATH --half  \
    --parameters=beam_size=${bs},device_list=[0],decode_batch_size=32,decode_alpha=${lpv},load_knn_datastore=False,use_knn_datastore=True,tm_count=${tm},knn_k=${knn_k},knn_t=${knn_T}

detok_path=$PROJECT_PATH/scripts
ref_data=$DATA_PATH/online_learning/dev.input.ref

data_input=$output_path/$bucket.dev.sknn.tm$tm.k$topk.T$knn_T.output
perl $detok_path/detokenizer.perl -l en < $data_input > $data_input.post
sacrebleu $ref_data < $data_input.post
