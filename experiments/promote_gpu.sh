
DSTORE_SIZE=19070000
PROJECT_PATH=/apdcephfs/share_916081/dirkiedai/knn-mt-research
MODEL_PATH=$PROJECT_PATH/pretrain/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/law
DATASTORE_PATH=$PROJECT_PATH/datastore/law/de-en

OUTPUT_PATH=$PROJECT_PATH/save_output_result

mkdir -p "$OUTPUT_PATH"



while true
do
    CUDA_VISIBLE_DEVICES=1  python3 $PROJECT_PATH/adaptive-knn-mt/experimental_generate.py $DATA_PATH \
    --gen-subset test \
    --path $MODEL_PATH --arch transformer_wmt19_de_en_with_datastore \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --quiet \
    --batch-size 64 \
    --tokenizer moses --remove-bpe \
    --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
    'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'k': 4, 'probe': 32,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_lambda_type': 'fix', 'knn_lambda_value': 0.8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10,
     }" 

    sleep 1m

done


