# Simple and Scalable Nearest Neighbor Machine Translation

Official Code for our paper "Simple and Scalable Nearest Neighbor Machine Translation" (ICLR 2023).

This branch provides the implementation of SK-MT(short for **S**imple and Scalable **k**NN-**MT**) under fairseq framework.

## Requirements and Installation
* pytorch version >= 1.1.0
* python version >= 3.6
* faiss-gpu >= 1.6.5
* sacrebleu == 1.5.1
* numpy == 1.23
* pytorch_scatter

You can install this project by running
```
pip install faiss-gpu
pip install sacrebleu==1.5.1
pip install numpy==1.23
pip install --editable ./
```

## Instructions
Please follow our [official guide](https://github.com/dirkiedai/sk-mt/tree/thumt) to prepare textual data and pre-trained model, and we also provide corresponding scripts below.
### Domain Adaptation
#### Retrieval
```
PROJECT_PATH=./

domain=it
type=test

DATA_PATH=/path/to/data/$domain

# we should tokenize the data with the bpecodes provided by the pretrained model first
for split in train dev test
do
    paste -d '\t' $DATA_PATH/$split.bpe.de $DATA_PATH/$split.bpe.en > $DATA_PATH/$split.txt
done

python $PROJECT_PATH/bm25_retrieval.py \
        --build_index --search_index \ 
        --index_file $DATA_PATH/train.txt --search_file $DATA_PATH/$type.txt \
        --output_file $DATA_PATH/$domain.$type \
        --index_name $domain --topk 64 
```
#### Data Preprocessing

```
PROJECT_PATH=./
DATA_PATH=/path/to/retrieved_samples
tmp_dir=/path/to/tmp

type=test
max_t=64

python $PROJECT_PATH/data_clean.py \
        --input $DATA_PATH/$domain.$type \
        --output $tmp_dir --subset $type \
        --max-t $max_t

DEST_PATH=/path/to/data-bin/$type_tm
DICT_PATH=/path/to/dictionary_provided_by_the_pretrained_model
for i in $(seq 1 $max_t)
do
    if [ $type == 'dev' ]
    then
        fairseq-preprocess --validpref $tmp_dir/${type}${i} -s de -t en --destdir $DEST_PATH/$i --srcdict $DICT_PATH/dict.de.txt --tgtdict $DICT_PATH/dict.en.txt --workers 20
    else
        fairseq-preprocess --testpref $tmp_dir/${type}${i} -s de -t en --destdir $DEST_PATH/$i --srcdict $DICT_PATH/dict.de.txt --tgtdict $DICT_PATH/dict.en.txt --workers 20
    fi
    
done
```
#### Inference with SK-MT
```
MODEL_PATH=/path/to/model
DOMAIN=it
OUTPUT_PATH=/path/to/output
DATA_PATH=/path/to/data

mkdir -p "$OUTPUT_PATH"

CUDA_VISIBLE_DEVICES=0 python3 experimental_generate.py $DATA_PATH \
    --gen-subset test \
    --path $MODEL_PATH --arch transformer_wmt19_de_en_with_datastore \
    --task translation_tm \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --batch-size 16 \
    --tm-counts 2 \
    --fp16 \
    --tokenizer moses --remove-bpe \
    --model-overrides "{'load_knn_datastore': False, 'use_knn_datastore': True, 'dstore_fp16': True, 'k': 1, 'probe': 32,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_lambda_temperature_value': 100,
     }" \
    | tee "$OUTPUT_PATH"/generate.txt
```


#### Inference with NMT
```
CUDA_VISIBLE_DEVICES=0 python3 experimental_generate.py $DATA_PATH \
    --gen-subset test \
    --path $MODEL_PATH --arch transformer_wmt19_de_en \
    --task translation \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --batch-size 16 \
    --fp16 \
    --tokenizer moses --remove-bpe \
    | tee "$OUTPUT_PATH"/generate.txt
```

## Citation
If you find this repo helpful for your research, please cite the following paper:

## Contact
If you have questions, suggestions and bug reports, please email <dirkiedye@gmail.com> or <zrustc11@gmail.com>.

