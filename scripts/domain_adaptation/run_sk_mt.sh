PROJECT_PATH=.
DATA_PATH=/path/to/data
MODEL_PATH=/path/to/model

export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}

device_id=1

type=test
lpv=0.6
tm=16
data=bm25
knn_k=2
knn_T=100
bsz=8

for domain in it medical koran law
do 
    OUTPUT_PATH=$PROJECT_PATH/output/domain_adaptation/$domain
    mkdir -p $OUTPUT_PATH

    CUDA_VISIBLE_DEVICES=$device_id python3 $PROJECT_PATH/thumt/bin/translator.py \
        --input $DATA_PATH/${data}_all/$domain.$type.clean \
        --output $OUTPUT_PATH/$domain.$type.$data.sknn.tm$tm.k$knn_k.T$knn_T.lpv$lpv.output  \
        --vocabulary $MODEL_PATH/dict.de.txt $MODEL_PATH/dict.en.txt  \
        --models transformer --checkpoints $MODEL_PATH --half  \
        --parameters=beam_size=4,device_list=[0],decode_batch_size=${bsz},decode_alpha=${lpv},load_knn_datastore=False,use_knn_datastore=True,tm_count=${tm},k=${knn_k},knn_temperature_value=${knn_T}

    detok_path=$PROJECT_PATH/scripts
    ref_data=$DATA_PATH/$domain/test.en

    data_input=$OUTPUT_PATH/$domain.$type.$data.sknn.tm$tm.k$knn_k.T$knn_T.lpv$lpv.output
    perl $detok_path/detokenizer.perl -l en < $data_input > $data_input.post
    sacrebleu $ref_data < $data_input.post
done
