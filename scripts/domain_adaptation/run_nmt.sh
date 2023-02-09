PROJECT_PATH=.
DATA_PATH=$PROJECT_PATH/data/domain_adaptation
MODEL_PATH=$PROJECT_PATH/model/pretrain/thumt_version

export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH
lpv=0.6
tm=0
bsz=8
device_id=0
type=test

for domain in it medical koran law
do 
    OUTPUT_PATH=$PROJECT_PATH/output/domain_adaptation/$domain
    mkdir -p $OUTPUT_PATH

    CUDA_VISIBLE_DEVICES=$device_id python3 $PROJECT_PATH/thumt/bin/translator.py \
        --input $DATA_PATH/$domain/$type.src.bpe \
        --output $OUTPUT_PATH/$domain.$type.nmt.lpv$lpv.output  \
        --vocabulary $MODEL_PATH/dict.de.txt $MODEL_PATH/dict.en.txt  \
        --models transformer --checkpoints $MODEL_PATH \
        --parameters=beam_size=4,device_list=[0],decode_batch_size=$bsz,decode_alpha=$lpv,tm_count=$tm,load_knn_datastore=False,use_knn_datastore=False 
    detok_path=$PROJECT_PATH/scripts
    ref_data=$DATA_PATH/$domain/test.en

    data_input=$OUTPUT_PATH/$domain.$type.nmt.lpv$lpv.output 
    perl $detok_path/detokenizer.perl -l en < $data_input > $data_input.post
    sacrebleu $ref_data < $data_input.post
done
