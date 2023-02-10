# Simple and Scalable Nearest Neighbor Machine Translation

Official Code for our paper "Simple and Scalable Nearest Neighbor Machine Translation" (ICLR 2023).


This project impliments our SK-MT(short for **S**imple and Scalable **k**NN-**MT**) as well as vanilla kNN-MT. The implementation is build upon [THUMT](https://github.com/THUNLP-MT/THUMT/tree/pytorch) and heavily inspired by [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt) and [KoK](https://github.com/wangqi1996/KoK).
Many thanks to the authors for making their code avaliable.

## Requirements and Installation
* pytorch version >= 1.1.0
* python version >= 3.6
* faiss-gpu >= 1.6.5
* pytorch_scatter = 2.0.5
You can install this project by
```
pip install --editable ./
```

## Instructions
Please follow our [official guide](https://github.com/dirkiedai/sk-mt/tree/thumt) to prepare data and pre-trained model.
### Domain Adaptation
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
    --model-overrides "{'load_knn_datastore': False, 'use_knn_datastore': True, 'dstore_fp16': True, 'k': 2, 'probe': 32,
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
If you have questions, suggestions and bug reports, please email <dirkiedye@gmail.com>.

