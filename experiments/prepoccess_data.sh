data_path=/data/dirkiedye/knn-mt-research/raw-data

dict_path=/data/dirkiedye/knn-mt-research/pretrain/wmt19_pretrain
tmp=/data/dirkiedye/knn-mt-research/raw-data/bm25_all/tmp

# if test -d $tmp;
#     then rm -r $tmp 
# fi 

mkdir -p $tmp

max_t=64

for folder_name in "bm25_all"
do 
    for domain in 'medical'
    do 
        for type in 'test'
        do 
            mkdir -p $tmp/$domain

            dest_path=/data/dirkiedye/knn-mt-research/data-bin/$domain/${type}_tm
            python3 data_clean.py --input $data_path/$folder_name/$domain.$type --output $tmp/$domain/ --subset $type --max-t $max_t

            for i in $(seq 1 $max_t)
            do
                if [ $type == 'dev' ]
                then
                fairseq-preprocess --validpref $tmp/$domain/${type}${i} -s de -t en --destdir $dest_path/$i --srcdict $dict_path/dict.de.txt --tgtdict $dict_path/dict.en.txt --workers 20
                else
                fairseq-preprocess --testpref $tmp/$domain/${type}${i} -s de -t en --destdir $dest_path/$i --srcdict $dict_path/dict.de.txt --tgtdict $dict_path/dict.en.txt --workers 20
                fi
                
            done
            
        done 
    done
done 
