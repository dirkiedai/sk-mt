declare -A DSTORE_SIZES_dict
declare -A KNN_TEMPERATURE_dict
declare -A KNN_LAMBDA_dict
declare -A KNN_K_dict

DSTORE_SIZES_dict=([it]="3613350" [medical]="6903320" [koran]="524400" [law]="19070000")
KNN_TEMPERATURE_dict=([it]="10" [medical]="100" [koran]="100" [law]="100")
KNN_LAMBDA_dict=([it]="0.7" [medical]="0.8" [koran]="0.8" [law]="0.8")
KNN_K_dict=([it]="8" [medical]="4" [koran]="16" [law]="4")

PROJECT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt-thumt
DATA_PATH=$PROJECT_PATH/data/domain_adaptation
MODEL_PATH=/data/dirkiedye/knn-mt-research/sknn-mt-thumt-copy/model/pretrain_nmt/thumt_version
DSTORE_PATH=/data/dirkiedye/knn-mt-research/datastore

export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}

lpv=0.6
tm=0
bsz=32
device_id=1
type=test


for domain in  medical law 
do 
    for knn_t in 500 1000 2000
    do
    dstore_size=${DSTORE_SIZES_dict[$domain]}
    
    LAMBDA=${KNN_LAMBDA_dict[$domain]}
    K=${KNN_K_dict[$domain]}
    TEMP=${KNN_TEMPERATURE_dict[$domain]}
    OUTPUT_PATH=$PROJECT_PATH/output/domain_adaptation/$domain
    mkdir -p $OUTPUT_PATH

    CUDA_VISIBLE_DEVICES=$device_id python3 $PROJECT_PATH/thumt/bin/translator.py \
        --input $DATA_PATH/$domain/$type.src.bpe \
        --output $OUTPUT_PATH/$domain.$type.knnmt.adapter.lpv$lpv.k$K.lambda$LAMBDA.temp$TEMP.output  \
        --vocabulary $MODEL_PATH/dict.de.txt $MODEL_PATH/dict.en.txt  \
        --models transformer --checkpoints $MODEL_PATH \
        --parameters=beam_size=4,device_list=[0],knn_t=$knn_t,decode_batch_size=$bsz,k=$K,decode_alpha=$lpv,tm_count=$tm,load_knn_datastore=True,use_knn_datastore=True,dstore_filename=$DSTORE_PATH/$domain,dstore_size=$dstore_size,dstore_fp16=True,knn_temperature_type='fix',knn_temperature_value=$TEMP,knn_lambda_type='fix',knn_lambda_value=$LAMBDA,use_gpu_to_search=True 

    detok_path=$PROJECT_PATH/scripts
    ref_data=$DATA_PATH/$domain/$type.en

    data_input=$OUTPUT_PATH/$domain.$type.knnmt.adapter.lpv$lpv.k$K.lambda$LAMBDA.temp$TEMP.output
    perl $detok_path/detokenizer.perl -l en < $data_input > $data_input.post
    sacrebleu $ref_data < $data_input.post 
    done
done
