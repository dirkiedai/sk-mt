declare -A DSTORE_SIZES_dict
DSTORE_SIZES_dict=([0.2]="3566700" [0.4]="7124000" [0.6]="10713000" [0.8]="14318000")

MODEL_PATH=/data/dirkiedye/knn-mt-research/pretrain/wmt19_pretrain/wmt19.de-en.ffn8192.pt

for RATIO in 0.2 0.4 0.6 0.8
do
    DATASTORE_PATH=/data/dirkiedye/knn-mt-research/datastore/law/$RATIO
    DATA_PATH=/data/dirkiedye/knn-mt-research/data-bin/law/$RATIO

    mkdir -p $DATASTORE_PATH
    DSTORE_SIZE=${DSTORE_SIZES_dict[$RATIO]}

    CUDA_VISIBLE_DEVICES=2 python3 /data/dirkiedye/knn-mt-research/sknn-mt/save_datastore.py $DATA_PATH \
        --dataset-impl mmap \
        --task translation \
        --valid-subset train \
        --path $MODEL_PATH \
        --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
        --decoder-embed-dim 1024 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH

    CUDA_VISIBLE_DEVICES=2 python3 /data/dirkiedye/knn-mt-research/sknn-mt/train_datastore.py \
        --dstore_mmap $DATASTORE_PATH \
        --dstore_size $DSTORE_SIZE \
        --dstore_fp16 \
        --faiss_index ${DATASTORE_PATH}/knn_index \
        --ncentroids 4096 \
        --probe 32 \
        --dimension 1024 \
        --load-to-mem --use-gpu
 
done

# declare -A DSTORE_SIZES_dict
# DSTORE_SIZES_dict=([it]="3613350" [medical]="6903320" [koran]="524400" [law]="19070000" [wiki]="47987250" [subtitles]="153604142")

# MODEL_PATH=/data/dirkiedye/knn-mt-research/pretrain/wmt19_pretrain/wmt19.de-en.ffn8192.pt

# for DOMAIN in koran law
# do
#     DATASTORE_PATH=/data/dirkiedye/knn-mt-research/datastore/$DOMAIN
#     DATA_PATH=/data/dirkiedye/knn-mt-research/data-bin/$DOMAIN

#     mkdir -p $DATASTORE_PATH
#     DSTORE_SIZE=${DSTORE_SIZES_dict[$DOMAIN]}

#     CUDA_VISIBLE_DEVICES=0 python3 /data/dirkiedye/knn-mt-research/sknn-mt/save_datastore.py $DATA_PATH \
#         --dataset-impl mmap \
#         --task translation \
#         --valid-subset train \
#         --path $MODEL_PATH \
#         --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
#         --decoder-embed-dim 1024 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH
 
# done
# 4096 and 1024 depend on your device and model separately


