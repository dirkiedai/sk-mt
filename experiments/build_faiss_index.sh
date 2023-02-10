
# declare -A DSTORE_SIZES_dict
# DSTORE_SIZES_dict=([it]="3613350" [medical]="6903320" [koran]="524400" [law]="19070000" [wiki]="47987250" [subtitles]="153604142")
# PROJECT_PATH=/apdcephfs/share_916081/dirkiedai/adaptive-knn-mt
# MODEL_PATH=/apdcephfs/share_916081/dirkiedai/pretrain/wmt19.de-en.ffn8192.pt

# DOMAINS=(it medical law)
# VERSIONS=(1 2 3)


# for DOMAIN in ${DOMAINS[*]}
# do
#   DSTORE_SIZE=${DSTORE_SIZES_dict[$DOMAIN]}
#   DATASTORE_PATH=/apdcephfs/share_916081/dirkiedai/datastore/$DOMAIN/de-en

#   for VERSION in ${VERSIONS[*]}
#   do
#     CUDA_VISIBLE_DEVICES=0 python3 $PROJECT_PATH/train_datastore.py \
#       --dstore_mmap $DATASTORE_PATH \
#       --dstore_size $DSTORE_SIZE \
#       --dstore_fp16 \
#       --faiss_index ${DATASTORE_PATH}/knn_index_v$VERSION \
#       --ncentroids 4096 \
#       --probe 32 \
#       --dimension 1024 \
#       --use_gpu \
#       --seed $VERSION
#   done
# done


declare -A DSTORE_SIZES_dict
DSTORE_SIZES_dict=([it]="3613350" [medical]="6903320" [koran]="524400" [law]="19070000" [wiki]="47987250" [subtitles]="153604142")
for DOMAIN in koran law
do 
    DATASTORE_PATH=/data/dirkiedye/knn-mt-research/datastore/$DOMAIN
    DSTORE_SIZE=${DSTORE_SIZES_dict[$DOMAIN]}

    CUDA_VISIBLE_DEVICES=1 python3 /data/dirkiedye/knn-mt-research/sknn-mt/train_datastore.py\
        --dstore_mmap $DATASTORE_PATH \
        --dstore_size $DSTORE_SIZE \
        --dstore_fp16 \
        --faiss_index ${DATASTORE_PATH}/knn_index \
        --ncentroids 4096 \
        --probe 32 \
        --dimension 1024 \
        --use-gpu --load-to-mem 

done






