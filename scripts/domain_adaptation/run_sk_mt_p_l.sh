project_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt
data_path=$project_path/data/domain_adaptation
model_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt-copy/model/pretrain_nmt/thumt_version
dstore_path=/data/dirkiedye/knn-mt-research/datastore/

type=test

export PYTHONPATH=$PYTHONPATH:${project_path}
lpv=0.6
tm=2
device_id=1
data=bm25
knn_k=2
knn_T=100

for bsz in 8
do

for tm in 16 8 4 2
do
results_file=$project_path/output/domain_adaptation/skmt.$type.results.bsz$bsz.tm$tm.k$knn_k.T$knn_T.speed

if test -f $results_file
then rm -r $results_file
fi

for domain in law
do 
    dstore_size=${DSTORE_SIZES_dict[$domain]}
    output_path=$project_path/output/domain_adaptation/$domain
    mkdir -p $output_path

    CUDA_VISIBLE_DEVICES=$device_id python3 $project_path/thumt/bin/translator.py \
        --input $data_path/${data}_all/$domain.$type.clean \
        --output $output_path/$domain.$type.$data.sknn.tm$tm.k$knn_k.T$knn_T.lpv$lpv.output  \
        --vocabulary $model_path/dict.de.txt $model_path/dict.en.txt  \
        --models transformer --checkpoints $model_path --half  \
        --parameters=beam_size=4,device_list=[0],decode_batch_size=${bsz},decode_alpha=${lpv},load_knn_datastore=False,use_knn_datastore=True,tm_count=${tm},k=${knn_k},knn_temperature_value=${knn_T} \
        | tail -2 >>$results_file

    detok_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt/scripts/
    ref_data=$data_path/$domain/$type.en

    data_input=$output_path/$domain.$type.$data.sknn.tm$tm.k$knn_k.T$knn_T.lpv$lpv.output
    perl $detok_path/detokenizer.perl -l en < $data_input > $data_input.post
    sacrebleu $ref_data < $data_input.post | tee -a $results_file
done
done
done