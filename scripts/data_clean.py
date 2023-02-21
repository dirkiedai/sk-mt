import argparse
import sys
import os 


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing Data for All-in-One Model Training")

    parser.add_argument("--input", type=str, help="input data")
    parser.add_argument("--output", type=str, help="output data")
    parser.add_argument("--max-t", type=int, help="the maximum number of tm")

    return parser.parse_args()

def data_process(line, max_t):
    new_line = ""
    items = line.split('\t')
    new_line += items[0]
    tm_count = 0
    for index in range(max_t):
        src_tm_index = index * 2 + 2
        tgt_tm_index = index * 2 + 3
        if src_tm_index < len(items):
            src_tm_len = len(items[src_tm_index])
            tgt_tm_len = len(items[tgt_tm_index])
            if tgt_tm_len > 0.5 * src_tm_len and tgt_tm_len < 2 * src_tm_len and src_tm_len != 1 and tgt_tm_len != 1:
                new_line += "\t" + items[src_tm_index] + "\t" + items[tgt_tm_index]
                tm_count += 1
        else:
            new_line += "\t1\t1"
            tm_count += 1

    if tm_count < max_t:
        for _ in range(max_t - tm_count):
            new_line += "\t1\t1"

    return new_line

def main(args):

    with open(args.input, 'r') as fr:
        with open(args.output, 'w') as fw:
            for line in fr.readlines():
                new_line = data_process(line.strip(), args.max_t)
                fw.write(new_line + '\n')                

if __name__ == '__main__':
    main(parse_args())
