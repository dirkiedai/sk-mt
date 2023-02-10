



# MODEL_PATH=/data/dirkiedye/knn-mt-research/pretrain/wmt19_pretrain/wmt19.de-en.ffn8192.pt

# DATA_PATH=/data/dirkiedye/knn-mt-research/data-bin/newstest2019

# CUDA_VISIBLE_DEVICES=4 python3 /data/dirkiedye/knn-mt-research/sknn-mt/experimental_generate.py $DATA_PATH \
#         --gen-subset test \
#         --path $MODEL_PATH \
#         --task translation --arch transformer \
#         --beam 4 --lenpen 2 --max-len-a 1.5 --max-len-b 30 --source-lang de --target-lang en \
#         --scoring sacrebleu \
#         --quiet \
#         --batch-size 32 \
#         --tokenizer moses --remove-bpe


MODEL_PATH=/data/dirkiedye/knn-mt-research/pretrain/wmt14model/wmt14_de2en.pt
DATA_PATH=/data/dirkiedye/knn-mt-research/data-bin/wmt14

OUTPUT_PATH=/data/dirkiedye/knn-mt-research/output/wmt14
mkdir -p $OUTPUT_PATH
CUDA_VISIBLE_DEVICES=5 python3 /data/dirkiedye/knn-mt-research/sknn-mt/experimental_generate.py $DATA_PATH \
        --gen-subset test \
        --path $MODEL_PATH \
        --task translation --arch transformer \
        --beam 5 --lenpen 1 --max-len-a 1.5 --max-len-b 10 --source-lang de --target-lang en \
        --scoring sacrebleu \
        --batch-size 32 \
        --tokenizer moses --remove-bpe \
        | tee "$OUTPUT_PATH"/generate.txt

grep ^S "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/src
grep ^T "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/ref
grep ^H "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp
grep ^D "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp.detok


# for DOMAIN in koran it medical law
# do
#     OUTPUT_PATH=/data/dirkiedye/knn-mt-research/sknn-mt/output/domain_adaptation/$DOMAIN
#     DATA_PATH=/data/dirkiedye/knn-mt-research/data-bin/$DOMAIN

#     if test -f $OUTPUT_PATH/nmt-generate.txt
#     then rm -f $OUTPUT_PATH/nmt-generate.txt
#     fi
#     CUDA_VISIBLE_DEVICES=2 python3 /data/dirkiedye/knn-mt-research/sknn-mt/experimental_generate.py $DATA_PATH\
#         --gen-subset test \
#         --path $MODEL_PATH \
#         --task translation --arch transformer \
#         --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
#         --scoring sacrebleu \
#         --quiet \
#         --batch-size 8 \
#         --tokenizer moses --remove-bpe \
#         | tee -a $OUTPUT_PATH/nmt-generate.txt
# done

