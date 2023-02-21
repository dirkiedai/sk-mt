PROJECT_PATH=./

domain=it
type=test

DATA_PATH=/path/to/data/$domain

# we should tokenize the data with the bpecodes provided by the pretrained model first
for split in train dev test
do
    paste -d '\t' $DATA_PATH/$split.bpe.de $DATA_PATH/$split.bpe.en > $DATA_PATH/$split.txt
done

python $PROJECT_PATH/scripts/bm25_retrieval.py \
        --build_index --search_index \ 
        --index_file $DATA_PATH/train.txt --search_file $DATA_PATH/$type.txt \
        --output_file $DATA_PATH/$domain.$type \
        --index_name $domain --topk 64 

python3 $PROJECT_PATH/scripts/data_clean.py \
        --input $DATA_PATH/$domain.$type \
        --output $DATA_PATH/$domain.$type.clean \ 
        --max-t 64