PROJECT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt-thumt
MODEL_PATH=/data/dirkiedye/knn-mt-research/pretrain/wmt14model/wmt14_en2de.pt
OUTPUT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt-thumt/model/wmt14_en2de

export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH

python3 $PROJECT_PATH/thumt/scripts/convert_fairseq_checkpoint.py --path $MODEL_PATH --output $OUTPUT_PATH
