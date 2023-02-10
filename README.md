# Simple and Scalable Nearest Neighbor Machine Translation

Official Code for our paper "Simple and Scalable Nearest Neighbor Machine Translation" (ICLR 2023).

This project impliments our SK-MT(short for **S**imple and Scalable **k**NN-**MT**) as well as vanilla kNN-MT. The implementation is build upon [THUMT](https://github.com/THUNLP-MT/THUMT/tree/pytorch) and heavily inspired by [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt) and [KoK](https://github.com/wangqi1996/KoK).
Many thanks to the authors for making their code avaliable.

## Requirements and Installation
* pytorch version >= 1.1.0
* python version >= 3.6

You need to install pytorch and cuda based on your hardware.
```
pip3 install --upgrade pip

pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install tensorboardX cffi cython dataclasses hydra-core numpy regex sacremoses sacrebleu tqdm nltk>=3.2 matplotlib absl-py fairseq sklearn tensorboard bitarray

pip3 install -U git+https://github.com/pltrdy/pyrouge
```
You can also refer to Dockerfile provided in `docker`.

## Instructions
We use an example to show how to use our codes. We have implemented our proposed SK-MT under two frameworks, [fairseq](https://github.com/facebookresearch/fairseq) and [THUMT](https://github.com/THUNLP-MT/THUMT/tree/pytorch). The scores we reported in our paper is evaluated based on THUMT framework.

### Pre-trained Model
The pre-trained translation model can be downloaded from this [site](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md). We use the De-En Single Model and follow [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt) to evaluate the performance of the kNN-MT and adaptive kNN-MT.
We provide the Transformer-based model we use in our experiments under those two frameworks: [Fairseq Model]() and [THUMT Model]().

### Data
The raw data can be downloaded in this [site](), and you should preprocess them with [moses](https://github.com/moses-smt/mosesdecoder) toolkits and the bpe-codes provided by pre-trained model. 
To implement SK-MT,  we recommend to follow [copyisallyouneed](https://github.com/jcyk/copyisallyouneed) to perform text retrieval using BM25. The obtained textual data can be used in THUMT framework. Moreover, if you favor Fairseq, you are required to follow its instruction to preprocess and binarize the textual data.
For convenience, We also provide pre-processed [textual data](https://drive.google.com/drive/folders/1q1XXouhLP-CpW6j44UPukdlB3nqRmwGP?usp=sharing) for THUMT and [binarized data]() for Fairseq.

### Domain Adaptation
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

