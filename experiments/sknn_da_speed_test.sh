
MODEL_PATH=/data/dirkiedye/knn-mt-research/pretrain/wmt19_pretrain/wmt19.de-en.ffn8192.pt

BATCH_SIZE=32
for DOMAIN in koran it
do
    OUTPUT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt/output/domain_adaptation/$DOMAIN
    DATA_PATH=/data/dirkiedye/knn-mt-research/data-bin/$DOMAIN

    mkdir -p "$OUTPUT_PATH"

    if test -f $OUTPUT_PATH/skmt-bsz$BATCH_SIZE-generate.txt
    then rm -f $OUTPUT_PATH/skmt-bsz$BATCH_SIZE-generate.txt
    fi

    for TM in 2 4 8 16 32
    do
    
    CUDA_VISIBLE_DEVICES=1 python3 /data/dirkiedye/knn-mt-research/sknn-mt/experimental_generate.py $DATA_PATH \
        --gen-subset test \
        --path $MODEL_PATH --arch transformer_wmt19_de_en_with_datastore \
        --task translation_tm \
        --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
        --scoring sacrebleu \
        --batch-size $BATCH_SIZE \
        --tm-counts $TM \
        --quiet \
        --fp16 \
        --tokenizer moses --remove-bpe \
        --model-overrides "{'load_knn_datastore': False, 'use_knn_datastore': True, 'dstore_fp16': True, 'k': 2, 'probe': 32,
        'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
        'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_lambda_temperature_value': 100,
        }" \
        | tee -a "$OUTPUT_PATH"/skmt-bsz$BATCH_SIZE-generate.txt
    done
done