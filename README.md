# Simple and Scalable Nearest Neighbor Machine Translation

Official Code for our paper "Simple and Scalable Nearest Neighbor Machine Translation" (ICLR 2023).

This project impliments our SK-MT(short for **S**imple and Scalable **k**NN-**MT**) as well as vanilla kNN-MT. The implementation is built upon [THUMT](https://github.com/THUNLP-MT/THUMT/tree/pytorch) and heavily inspired by [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt) and [KoK](https://github.com/wangqi1996/KoK). Many thanks to the authors for making their code avaliable.

We also provide the implementation built upon [fairseq](https://github.com/facebookresearch/fairseq), which can be found in [fairseq branch](https://github.com/dirkiedai/sk-mt/tree/fairseq). The scores we reported in our paper is evaluated based on THUMT framework.


## Requirements and Installation
* pytorch version >= 1.1.0
* python version >= 3.6

You need to install pytorch and cuda based on your hardware.
```
pip install --upgrade pip

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install tensorboardX cffi cython dataclasses hydra-core numpy regex sacremoses sacrebleu tqdm nltk>=3.2 matplotlib absl-py fairseq sklearn tensorboard bitarray

pip install -U git+https://github.com/pltrdy/pyrouge
```

## Instructions

### Pre-trained Model
The pre-trained translation model can be downloaded from this [site](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md). We use the De-En Single Model and follow [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt) to evaluate the performance of the kNN-MT and adaptive kNN-MT.
We provide the Transformer-based model we use in our experiments under those two frameworks: [fairseq Model](https://drive.google.com/file/d/1cVf6TbZxj59o12HRIgZgtYFq_zAViR8e/view?usp=sharing) and [THUMT Model](https://drive.google.com/file/d/18zhbv-JXeSL802OsUL0wZUgSNcjjE8L1/view?usp=sharing).

### Data
The raw data can be downloaded in this [site](https://github.com/roeeaharoni/unsupervised-domain-clusters), and you should preprocess them with [moses](https://github.com/moses-smt/mosesdecoder) toolkits and the bpe-codes provided by pre-trained model. 
To implement SK-MT,  we recommend to follow [copyisallyouneed](https://github.com/jcyk/copyisallyouneed) to perform text retrieval using BM25. The obtained textual data can be used in THUMT framework. Moreover, if you favor fairseq, you are required to follow its instruction to preprocess and binarize the textual data.
For convenience, We also provide pre-processed [textual data](https://drive.google.com/file/d/1KJ3jxXN7ziM9ZlIRHK-2XM5g9yVf9NaG/view?usp=sharing) for THUMT and [binarized data](https://drive.google.com/file/d/1Xg4DVgZpOltME76RyZ8rKQ6xjVOHbolX/view?usp=sharing) for fairseq.

### Domain Adaptation
This section provides instructions to perform SK-MT based on THUMT framework. More information about the implementations on fairseq framework can be found in the [fairseq branch](https://github.com/dirkiedai/sk-mt/tree/fairseq).
#### Inference with SK-MT
```
bash scripts/domain_adaptation/run_sk_mt.sh
```
The batch size and other parameters should be adjusted by yourself depending on the hardware condition. We recommend to adopt the following hyper-parameters to replicate good SK-MT results.
| |  tm counts   | $k$  | $\tau$ 
| :----:|  :----:  | :----:  | :----:  | 
|SK-MT$_{1}$ | 2  | 1 | 100 | 
| SK-MT$_{2}$ | 16  | 2 |  100| 


#### Inference with NMT
```
bash scripts/domain_adaptation/run_nmt.sh
```

## Online Learning
### Inference with SK-MT
```
bash scripts/online_learning/run_sk_mt.sh
```
The recommeded hyper-parameters are the same as what used in Domain Adaptation.


## Citation
If you find this repo helpful for your research, please cite the following paper:

## Contact
If you have questions, suggestions and bug reports, please email <dirkiedye@gmail.com>.

