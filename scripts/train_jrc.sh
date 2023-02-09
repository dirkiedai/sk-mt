run_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt
data_path=/data/dirkiedye/knn-mt-research/raw-data/jrc-gu/deen
model_path=/data/dirkiedye/knn-mt-research/sknn-mt-thumt/model

export PYTHONPATH=$PYTHONPATH:${run_path}

source_lang=de
target_lang=en
split_num=8

# transformer version
CUDA_VISIABLE_DEVICES=5 python3 ${run_path}/thumt/bin/trainer.py  \
    --input ${data_path}/train/jrc.train.src ${data_path}/train/jrc.train.tgt  \
    --vocabulary ${data_path}/thumt.dict.${source_lang}.txt ${data_path}/thumt.dict.${target_lang}.txt  \
    --model transformer --validation ${data_path}/dev/jrc.dev.src   \
    --references ${data_path}/dev/jrc.dev.tgt --half  \
    --output ${model_path}/transformer_${source_lang}2${target_lang}_h512_f2048_e6_d6_a8 \
    --parameters=batch_size=4096,device_list=[0],encoder_filter_size=2048,decoder_filter_size=2048,update_cycle=1,hidden_size=512,num_encoder_layers=6,num_decoder_layers=6,num_heads=8,train_steps=100000,save_checkpoint_steps=5000,eval_steps=5000,decode_batch_size=4096,save_summary=false,beam_size=4,decode_alpha=0.6