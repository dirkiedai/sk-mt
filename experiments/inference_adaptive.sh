declare -A DSTORE_SIZES_dict
DSTORE_SIZES_dict=([it]="3613350" [medical]="6903320" [koran]="524400" [law]="19070000" [wiki]="47987250" [subtitles]="153604142")

PROJECT_PATH=/apdcephfs/share_916081/dirkiedai/adaptive-knn-mt

MODEL_PATH=/data/dirkiedye/knn-mt-research/pretrain/aknn_it_hid32_maxk8_full/checkpoint_20_2500.pt

BATCH_SIZE=32
DOMAINS=(koran it medical law)

for DOMAIN in ${DOMAINS[*]}
do
    DATASTORE_PATH=/data/dirkiedye/knn-mt-research/datastore/$DOMAIN
    DSTORE_SIZE=${DSTORE_SIZES_dict[$DOMAIN]}

    OUTPUT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt/output/domain_adaptation/$DOMAIN
    DATA_PATH=/data/dirkiedye/knn-mt-research/data-bin/$DOMAIN
    mkdir -p "$OUTPUT_PATH"

    if test -f "$OUTPUT_PATH"/akmt-bsz$BATCH_SIZE-generate.txt 
    then rm -f "$OUTPUT_PATH"/akmt-bsz$BATCH_SIZE-generate.txt 
    fi

    CUDA_VISIBLE_DEVICES=2 python3 /data/dirkiedye/knn-mt-research/sknn-mt/experimental_generate.py $DATA_PATH \
        --gen-subset test \
        --path "$MODEL_PATH" --arch transformer_wmt19_de_en_with_datastore \
        --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
        --scoring sacrebleu \
        --batch-size $BATCH_SIZE \
        --quiet \
        --tokenizer moses --remove-bpe \
        --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
        'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'probe': 32,
        'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
        'knn_temperature_type': 'fix', 'knn_temperature_value': 10,}" \
        | tee -a "$OUTPUT_PATH"/akmt-bsz$BATCH_SIZE-generate.txt 
done




