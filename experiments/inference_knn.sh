declare -A DSTORE_SIZES_dict
declare -A KNN_TEMPERATURE_dict
declare -A KNN_LAMBDA_dict
declare -A KNN_K_dict

DSTORE_SIZES_dict=([it]="3613350" [medical]="6903320" [koran]="524400" [law]="19070000")
KNN_TEMPERATURE_dict=([it]="10" [medical]="10" [koran]="100" [law]="10")
KNN_LAMBDA_dict=([it]="0.7" [medical]="0.8" [koran]="0.8" [law]="0.8")
KNN_K_dict=([it]="8" [medical]="4" [koran]="16" [law]="4")

MODEL_PATH=/data/dirkiedye/knn-mt-research/pretrain/wmt19_pretrain/wmt19.de-en.ffn8192.pt

BATCH_SIZE=8
DOMAINS=(it medical koran law)

for DOMAIN in ${DOMAINS[*]}
do
    
    DSTORE_SIZE=${DSTORE_SIZES_dict[$DOMAIN]}

    LAMBDA=${KNN_LAMBDA_dict[$DOMAIN]}
    K=${KNN_K_dict[$DOMAIN]}
    TEMP=${KNN_TEMPERATURE_dict[$DOMAIN]}

    DATA_PATH=/data/dirkiedye/knn-mt-research/data-bin/$DOMAIN
    DATASTORE_PATH=/data/dirkiedye/knn-mt-research/datastore/$DOMAIN

    OUTPUT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt/output/domain_adaptation/$DOMAIN
    mkdir -p "$OUTPUT_PATH"

    if test -f "$OUTPUT_PATH"/knnmt-bsz$BATCH_SIZE-generate.txt 
    then rm -f "$OUTPUT_PATH"/knnmt-bsz$BATCH_SIZE-generate.txt 
    fi

    CUDA_VISIBLE_DEVICES=2 python3 /data/dirkiedye/knn-mt-research/sknn-mt/experimental_generate.py $DATA_PATH \
        --gen-subset test\
        --path "$MODEL_PATH" --arch transformer_wmt19_de_en_with_datastore \
        --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
        --scoring sacrebleu \
        --batch-size $BATCH_SIZE \
        --quiet \
        --tokenizer moses --remove-bpe \
        --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
        'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'probe': 32, 'k': $K,
        'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
        'knn_temperature_type': 'fix', 'knn_temperature_value': $TEMP,'knn_lambda_type': 'fix', 'knn_lambda_value': $LAMBDA}" \
        | tee -a "$OUTPUT_PATH"/knnmt-bsz$BATCH_SIZE-generate.txt 
done



