declare -A DSTORE_SIZES_dict
declare -A KNN_TEMPERATURE_dict
declare -A KNN_LAMBDA_dict
declare -A KNN_K_dict

project_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt
data_path=$project_path/data/domain_adaptation
model_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt-copy/model/pretrain_nmt/thumt_version
dstore_path=/data/dirkiedye/knn-mt-research/datastore/


DSTORE_SIZES_dict=([0.2]="3566700" [0.4]="7124000" [0.6]="10713000" [0.8]="14318000" [1]="19070000")


type=test

export PYTHONPATH=$PYTHONPATH:${project_path}
lpv=0.6
device_id=1
data=bm25
knn_k=1
knn_T=100
bsz=16
domain=law
results_file=$project_path/output/domain_adaptation/law.dsize.m.results.k.test.skmt
output_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt/output/domain_adaptation/law

if test -f $results_file
then rm -f $results_file
fi

for RATIO in 0.2 0.4 0.6 0.8 1
do
    echo "ratio $RATIO" >> $results_file
    # echo "kNN-MT" >> $results_file
    # K=4
    # TEMP=10
    # LAMBDA=0.8
    # tm=0
    # dstore_size=${DSTORE_SIZES_dict[$RATIO]}
    # dstore_path=/data/dirkiedye/knn-mt-research/datastore/$domain/
    # CUDA_VISIBLE_DEVICES=$device_id python3 $project_path/thumt/bin/translator.py \
    #     --input $data_path/$domain/$type.src.bpe \
    #     --output $output_path/$domain.$type.knnmt.lpv$lpv.k$K.lambda$LAMBDA.temp$TEMP.r$RATIO.output  \
    #     --vocabulary $model_path/dict.de.txt $model_path/dict.en.txt  \
    #     --models transformer --checkpoints $model_path \
    #     --parameters=beam_size=4,device_list=[0],decode_batch_size=$bsz,k=$K,decode_alpha=$lpv,tm_count=0,load_knn_datastore=True,use_knn_datastore=True,dstore_filename=$dstore_path/$RATIO,dstore_size=$dstore_size,dstore_fp16=True,knn_temperature_type='fix',knn_temperature_value=$TEMP,knn_lambda_type='fix',knn_lambda_value=$LAMBDA,use_gpu_to_search=True \
    #     | tail -2 >>$results_file 

    # detok_path=$project_path/scripts
    # ref_data=$data_path/$domain/$type.en

    # data_input=$output_path/$domain.$type.knnmt.lpv$lpv.k$K.lambda$LAMBDA.temp$TEMP.r$RATIO.output
    # perl $detok_path/detokenizer.perl -l en < $data_input > $data_input.post
    # sacrebleu $ref_data < $data_input.post | tee -a $results_file

    # echo "AK-MT" >> $results_file

    # CUDA_VISIBLE_DEVICES=$device_id python3 $project_path/thumt/bin/translator.py \
    #     --input $data_path/$domain/$type.src.bpe \
    #     --output $output_path/$domain.$type.akmt.lpv$lpv.r$RATIO.output  \
    #     --vocabulary $model_path/dict.de.txt $model_path/dict.en.txt  \
    #     --models transformer --checkpoints /data/dirkiedye/knn-mt-research/sknn-mt-thumt-copy/model/pretrain_nmt/thumt_knn_version \
    #     --parameters=beam_size=4,device_list=[0],decode_batch_size=$bsz,decode_alpha=$lpv,tm_count=0,load_knn_datastore=True,use_knn_datastore=True,dstore_filename=$dstore_path/$RATIO,dstore_size=$dstore_size,dstore_fp16=True,use_gpu_to_search=True,label_count_as_feature=True \
    #     | tail -2 >>$results_file
    
    # detok_path=$project_path/scripts
    # ref_data=$data_path/$domain/$type.en

    # data_input=$output_path/$domain.$type.akmt.lpv$lpv.r$RATIO.output
    # perl $detok_path/detokenizer.perl -l en < $data_input > $data_input.post
    # sacrebleu $ref_data < $data_input.post | tee -a $results_file


    tm=2
    knn_k=1
    CUDA_VISIBLE_DEVICES=$device_id python3 $project_path/thumt/bin/translator.py \
        --input $data_path/${data}_all/law-ratio/law.$RATIO.$type.clean \
        --output $output_path/$domain.$type.$data.sknn.tm$tm.k$knn_k.T$knn_T.lpv$lpv.r$RATIO.output  \
        --vocabulary $model_path/dict.de.txt $model_path/dict.en.txt  \
        --models transformer --checkpoints $model_path --half  \
        --parameters=beam_size=4,device_list=[0],decode_batch_size=${bsz},decode_alpha=${lpv},load_knn_datastore=False,use_knn_datastore=True,tm_count=${tm},k=${knn_k},knn_temperature_value=${knn_T} \
        | tail -2 >>$results_file

    detok_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt/scripts/
    ref_data=$data_path/$domain/$type.en

    data_input=$output_path/$domain.$type.$data.sknn.tm$tm.k$knn_k.T$knn_T.lpv$lpv.r$RATIO.output
    perl $detok_path/detokenizer.perl -l en < $data_input > $data_input.post
    sacrebleu $ref_data < $data_input.post | tee -a $results_file


    for tm in 4 8 16
    do
        for knn_k in 2
        do

        echo "SK-MT, tm: $tm" >> $results_file

        CUDA_VISIBLE_DEVICES=$device_id python3 $project_path/thumt/bin/translator.py \
            --input $data_path/${data}_all/law-ratio/law.$RATIO.$type.clean \
            --output $output_path/$domain.$type.$data.sknn.tm$tm.k$knn_k.T$knn_T.lpv$lpv.r$RATIO.output  \
            --vocabulary $model_path/dict.de.txt $model_path/dict.en.txt  \
            --models transformer --checkpoints $model_path --half  \
            --parameters=beam_size=4,device_list=[0],decode_batch_size=${bsz},decode_alpha=${lpv},load_knn_datastore=False,use_knn_datastore=True,tm_count=${tm},k=${knn_k},knn_temperature_value=${knn_T} \
            | tail -2 >>$results_file

        detok_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt/scripts/
        ref_data=$data_path/$domain/$type.en

        data_input=$output_path/$domain.$type.$data.sknn.tm$tm.k$knn_k.T$knn_T.lpv$lpv.r$RATIO.output
        perl $detok_path/detokenizer.perl -l en < $data_input > $data_input.post
        sacrebleu $ref_data < $data_input.post | tee -a $results_file
        done
    done
done




