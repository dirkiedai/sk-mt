
PROJECT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt

DATA_PATH=/data/dirkiedye/knn-mt-research/sknn-mt-thumt/data/domain_adaptation/bm25_all/law-ratio

for ratio in 0.2 0.4 0.6 0.8 
do
DEST_PATH=/data/dirkiedye/knn-mt-research/data-bin/law
DICT_PATH=/data/dirkiedye/knn-mt-research/data-bin/law
python3 $PROJECT_PATH/fairseq_cli/preprocess.py \
        --trainpref $DATA_PATH/law.$ratio.train \
        --source-lang de --target-lang en \
        --destdir $DEST_PATH/$ratio  \
        --srcdict $DICT_PATH/dict.de.txt --tgtdict $DICT_PATH/dict.en.txt \
        --workers 20 
done

# PROJECT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt

# DATA_PATH=/data/dirkiedye/knn-mt-research/raw-data/WMT14/de-en
# DEST_PATH=/data/dirkiedye/knn-mt-research/data-bin/wmt14 
# DICT_PATH=/data/dirkiedye/knn-mt-research/raw-data/WMT14/de-en
# python3 $PROJECT_PATH/fairseq_cli/preprocess.py \
#         --testpref $DATA_PATH/test \
#         --validpref $DATA_PATH/dev \
#         --source-lang de --target-lang en \
#         --destdir $DEST_PATH  \
#         --srcdict $DICT_PATH/dict.de.txt --tgtdict $DICT_PATH/dict.en.txt \
#         --workers 20 


# PROJECT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt

# DATA_PATH=/data/dirkiedye/knn-mt-research/raw-data/wmt19/de-en
# DEST_PATH=/data/dirkiedye/knn-mt-research/data-bin/wmt19 
# DICT_PATH=/data/dirkiedye/knn-mt-research/pretrain/wmt19_pretrain/
# python3 $PROJECT_PATH/fairseq_cli/preprocess.py \
#         --trainpref $DATA_PATH/train.de-en.tok.bpe.length \
#         --source-lang de --target-lang en \
#         --destdir $DEST_PATH  \
#         --srcdict $DICT_PATH/dict.de.txt --tgtdict $DICT_PATH/dict.en.txt \
#         --workers 20 
