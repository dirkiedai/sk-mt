FROM mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda11.0-cudnn8.1:latest

ENV PYTHONIOENCODING=utf-8

RUN pip3 install --upgrade pip

RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install tensorboardX cffi cython dataclasses hydra-core numpy regex sacremoses sacrebleu tqdm nltk>=3.2 matplotlib absl-py fairseq sklearn tensorboard bitarray

RUN pip3 install -U git+https://github.com/pltrdy/pyrouge