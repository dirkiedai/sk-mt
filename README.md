# Simple and Scalable Nearest Neighbor Machine Translation

Official Code for our paper "Simple and Scalable Nearest Neighbor Machine Translation".

This project impliments our SK-MT(short for **S**imple and Scalable **k**NN-**MT**) as well as vanilla kNN-MT. The implementation is build upon [THUMT](https://github.com/THUNLP-MT/THUMT/tree/pytorch), and heavily inspired by [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt) and [KoK](https://github.com/wangqi1996/KoK).
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
We use an example to show how to use our codes.
### Pre-trained Model and Data
The pre-trained translation model can be downloaded from this [site](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md). We use the De-En Single Model for all experiments. Note that this pre-trained model cannot be applied to THUMT directly. To make it compatiable to THUMT framework, we can run the following script to make use of this model. And also, it is necessary to add '\<s>','\<pad>','\</s>','\<unk>' to the vocabulary. For convenience, we provide [converted model]() used in our experiments.
```
bash model/convert_model.sh
```
### Data
The raw data can be downloaded in this [site](), and you should preprocess them with [moses]() toolkits and the bpe-codes provided by pre-trained model. 
To implement SK-MT, you are required to perform text retrieval using BM25 at first, and we recommend to follow [copyisallyouneed](https://github.com/jcyk/copyisallyouneed) to obtain serveral reference samples.
After that, we recommend to clean the dataset using these scripts for differenct datasets, respectively.
```
bash scripts/preprocess/data_clean.sh
bash scripts/preprocess/data_clean_ol.sh
```
For convenience, We also provide pre-processed textual data.
This data is organized as

```
-data
--domain adaptation
---original
----it/medical/koran/law
---bm25  
----it/medical/koran/law
--online learning
---original
----50/100/200/500/1000
---bm25
----50/100/200/500/1000
```
### Domain Adaptation
#### Inference with SK-MT
```
bash scripts/domain_adaptation/run_sk_mt.sh
```
The 'batch size' and parameters should be adjust by yourself depends on your GPU. project_path
model_path, data_path should be determined on your own. We recommend you to use below hyper-parameters to replicate our SK-MT results.
| |  tm count   | $k$  | $\tau$ 
| :----:|  :----:  | :----:  | :----:  | 
|SK-MT$_{1}$ | 2  | 1 | 100 | 
| SK-MT$_{2}$ | 16  | 2 |  100| 


#### Inference with NMT
```
bash scripts/domain_adaptation/run_nmt.sh
```

#### Inference with kNN-MT
Before inference, please follow [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt) to create datastore and build Faiss index. Once finished, you can refer to this script for kNN-MT inference.
```
bash scripts/domain_adaptation/run_knnmt.sh
```
Following [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt), we recommend you to use below hyper-parameters to replicate good vanilla kNN-mt results.
| |  IT   | Medical  | Koran | Law
| :----:|  :----:  | :----:  |   :----:  | :----:  | 
|$k$ | 8  | 4 | 16 | 4 |
| $\lambda$ | 0.7  | 0.8 |  0.8|  0.8|
| $\tau$ | 10  | 10 | 100 |  10 |

## Online Learning
### Inference with SK-MT
```
bash scripts/online_learning/run_sk_mt.sh
```
The recommeded hyper-parameters are the same as what used in Domain Adaptation.

### Inference with KoK and kNN-MT
Please refer to [KoK](https://github.com/wangqi1996/KoK) for implementation details.

## Citation
If you find this repo helpful for your research, please cite the following paper:

## Contact
If you have questions, suggestions and bug reports, please email <dirkiedye@gmail.com>.

